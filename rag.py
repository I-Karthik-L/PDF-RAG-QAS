import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


load_dotenv()
logger = logging.getLogger("multi_pdf_rag")

# --------------------------------------------------------------------------- #
# Paths
# Read-only app code lives next to this file. Anything the app WRITES goes in
# DATA_DIR: next to rag.py locally, /tmp on Vercel (the only writable place
# there, and it is wiped between invocations), or wherever DATA_DIR points.
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "NIPS-2017-attention-is-all-you-need-Paper.pdf"  # optional bundled sample
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"

DATA_DIR = Path(os.getenv("DATA_DIR") or ("/tmp/pdf_rag" if os.getenv("VERCEL") else BASE_DIR))
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
MANIFEST_PATH = DATA_DIR / "documents.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "local_multi_pdf_rag"
MAX_PDF_SIZE_BYTES = 25 * 1024 * 1024
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOTAL_CHUNKS = 8          # target number of chunks sent to the LLM
MIN_CHUNKS_PER_DOC = 3

NOT_FOUND = "I couldn't find that information in the selected documents."

# filename -> {"path": Path, "removable": bool}
documents: dict[str, dict] = {}
indexed_documents: set[str] = set()
embeddings = None
vectorstore = None
llm = None
rag_error = None
rag_lock = Lock()


class QueryRequest(BaseModel):
    question: str = Field(..., max_length=2000)
    selected_documents: list[str] = Field(default_factory=list)


PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only the document context below. Each context block "
    "starts with its source in square brackets. If the answer is not contained in "
    'the context, say exactly: "' + NOT_FOUND + '" '
    "Do not make up facts.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)


# --------------------------------------------------------------------------- #
# Lazy singletons
# --------------------------------------------------------------------------- #
def get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return embeddings


def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
    return vectorstore


def get_llm():
    global llm
    if llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            max_tokens=1500,  # reasoning models spend part of this on thinking
            api_key=api_key,
        )
    return llm


# --------------------------------------------------------------------------- #
# Manifest: remembers uploaded documents across restarts
# --------------------------------------------------------------------------- #
def save_manifest():
    data = {
        name: {"file": doc["path"].name}
        for name, doc in documents.items()
        if doc["removable"]  # the bundled sample is re-registered at startup
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MANIFEST_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(MANIFEST_PATH)


def load_manifest():
    if not MANIFEST_PATH.exists():
        return
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read %s; starting with no uploaded documents.", MANIFEST_PATH)
        return
    for name, info in data.items():
        path = UPLOAD_DIR / info["file"]
        if path.exists():
            documents[name] = {"path": path, "removable": True}


def register_bundled_pdf():
    if PDF_PATH.exists() and PDF_PATH.name not in documents:
        documents[PDF_PATH.name] = {"path": PDF_PATH, "removable": False}


def sync_index():
    """Make sure every known document has chunks in Chroma (index it if not)."""
    store = get_vectorstore()
    for filename, document in list(documents.items()):
        try:
            if not store.get(where={"source": filename}, limit=1)["ids"]:
                logger.info("Indexing %s", filename)
                load_and_index_pdf(document["path"], filename)
            indexed_documents.add(filename)
        except Exception:
            logger.exception("Could not index %s; removing it from the list.", filename)
            del documents[filename]
    save_manifest()


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_manifest()
    register_bundled_pdf()
    if documents:
        sync_index()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def document_payload(filename: str):
    return {"filename": filename, "removable": documents[filename]["removable"]}


def load_and_index_pdf(pdf_path: Path, filename: str):
    loaded_pages = PyPDFLoader(str(pdf_path)).load()
    if not loaded_pages or not any(page.page_content.strip() for page in loaded_pages):
        raise ValueError("This PDF does not contain readable text.")
    for page in loaded_pages:
        page.metadata["source"] = filename
        if "page" in page.metadata:
            page.metadata["page"] = page.metadata["page"] + 1
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(loaded_pages)
    if not chunks:
        raise ValueError("This PDF does not contain text that can be indexed.")
    get_vectorstore().add_documents(chunks)


def save_upload(upload: UploadFile) -> tuple[Path, str]:
    if not upload.filename or not upload.filename.lower().endswith(".pdf"):
        upload.file.close()
        raise HTTPException(status_code=400, detail="Please upload PDF files only.")
    filename = Path(upload.filename).name
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / f"{uuid4().hex}_{filename}"
    bytes_written = 0
    try:
        with destination.open("wb") as saved_file:
            while chunk := upload.file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written == len(chunk) and b"%PDF-" not in chunk[:1024]:
                    raise HTTPException(status_code=400, detail=f"{filename} is not a valid PDF.")
                if bytes_written > MAX_PDF_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail=f"{filename} is larger than 25 MB.")
                saved_file.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not save {filename}.") from exc
    finally:
        upload.file.close()
    if bytes_written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"{filename} is empty.")
    return destination, filename


def format_context(found_documents):
    blocks = []
    for document in found_documents:
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page")
        label = f"{source}, p.{page}" if page is not None else source
        blocks.append(f"[{label}]\n{document.page_content}")
    return "\n\n".join(blocks)


def source_payload(found_documents):
    sources, seen = [], set()
    for document in found_documents:
        source = document.metadata.get("source")
        if not source:
            continue
        page = document.metadata.get("page")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append({"filename": source, "page": page})
    return sources


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(documents), "last_error": rag_error}


@app.get("/documents")
def list_documents():
    return {"documents": [document_payload(name) for name in documents]}


@app.post("/upload-pdf")
def upload_pdfs(files: list[UploadFile] = File(...)):
    global rag_error
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one PDF to upload.")

    uploaded, failed = [], []
    for upload in files:
        display_name = Path(upload.filename or "unknown").name
        try:
            if display_name in documents:
                upload.file.close()
                raise HTTPException(status_code=409, detail=f"{display_name} has already been uploaded.")
            saved_path, filename = save_upload(upload)
            with rag_lock:
                if filename in documents:  # lost a race with another request
                    saved_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=409, detail=f"{filename} has already been uploaded.")
                try:
                    load_and_index_pdf(saved_path, filename)
                except Exception as exc:
                    saved_path.unlink(missing_ok=True)
                    rag_error = str(exc)
                    logger.exception("Indexing failed for %s", filename)
                    raise HTTPException(
                        status_code=422,
                        detail=f"{filename} could not be processed. Make sure it is a readable, text-based PDF.",
                    ) from exc
                documents[filename] = {"path": saved_path, "removable": True}
                indexed_documents.add(filename)
                save_manifest()
                uploaded.append(document_payload(filename))
        except HTTPException as exc:
            failed.append({"filename": display_name, "status": exc.status_code, "reason": exc.detail})

    if not uploaded:  # everything failed: keep the old error behaviour for the UI
        first = failed[0]
        raise HTTPException(status_code=first["status"], detail=first["reason"])

    rag_error = None
    return {
        "message": "PDFs are indexed and ready for questions.",
        "documents": uploaded,
        "failed": failed,
    }


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    with rag_lock:
        if filename not in documents:
            raise HTTPException(status_code=404, detail="Document not found.")
        if not documents[filename]["removable"]:
            raise HTTPException(status_code=400, detail="The bundled document cannot be removed.")
        document = documents[filename]
        try:
            # langchain-chroma forwards `where` to Chroma. The old
            # langchain_community.Chroma wrapper silently ignored it.
            get_vectorstore().delete(where={"source": filename})
            document["path"].unlink(missing_ok=True)
            del documents[filename]
            indexed_documents.discard(filename)
            save_manifest()
        except Exception as exc:
            logger.exception("Could not remove %s", filename)
            raise HTTPException(status_code=500, detail="The document could not be removed.") from exc
    return {"message": f"{filename} was removed."}


@app.post("/ask")
def ask(request: QueryRequest):
    global rag_error
    question = request.question.strip()
    selected_documents = list(dict.fromkeys(request.selected_documents))
    if not question:
        raise HTTPException(status_code=400, detail="Enter a question before sending it.")
    if not selected_documents:
        raise HTTPException(status_code=400, detail="Select at least one document first.")
    if set(selected_documents) - documents.keys():
        raise HTTPException(status_code=400, detail="One or more selected documents are unavailable.")

    # 1) Retrieval: a fair share of chunks from each selected document
    per_doc_k = max(MIN_CHUNKS_PER_DOC, TOTAL_CHUNKS // len(selected_documents))
    try:
        found_documents = []
        with rag_lock:
            store = get_vectorstore()
            for filename in selected_documents:
                found_documents += store.similarity_search(
                    question, k=per_doc_k, filter={"source": filename}
                )
    except Exception as exc:
        rag_error = str(exc)
        logger.exception("Retrieval failed")
        raise HTTPException(
            status_code=500,
            detail="Searching the documents failed. Check the server logs and try again.",
        ) from exc

    if not found_documents:
        return {"answer": NOT_FOUND, "sources": []}

    # 2) Generation
    try:
        answer = (PROMPT | get_llm() | StrOutputParser()).invoke(
            {"context": format_context(found_documents), "question": question}
        )
    except Exception as exc:
        rag_error = str(exc)
        logger.exception("LLM call failed")
        raise HTTPException(
            status_code=500,
            detail="The answer could not be generated. Check the Groq API key and try again.",
        ) from exc

    rag_error = None
    answer = answer.strip()
    if answer.startswith(NOT_FOUND):
        return {"answer": NOT_FOUND, "sources": []}
    return {"answer": answer, "sources": source_payload(found_documents)}
