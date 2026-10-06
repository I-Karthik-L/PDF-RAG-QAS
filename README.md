# 📄 Multi-PDF RAG Question Answering System

A **Retrieval-Augmented Generation (RAG)** application that lets you upload multiple PDFs, choose which ones to search, and ask questions through a local web interface. Answers are grounded in the retrieved document text and include the source PDF and page number.

---

## 🚀 Features

* 📄 Upload and manage multiple PDFs (up to 25 MB each, text-based PDFs only)
* ☑ Choose exactly which PDFs are searched for each question
* ⚖️ Fair retrieval: each selected PDF contributes its own top chunks, so one document can't crowd out the others
* 🔍 Semantic search using sentence-transformer embeddings
* 🗂️ Every chunk is tagged with its source PDF and page number
* 🔗 Answers return the retrieved sources (filename + page)
* 💾 Persistent storage: uploaded PDFs and their embeddings survive server restarts
* 🤖 LLM-powered answers via the Groq API
* 🧠 Strict prompt: the model answers only from the retrieved context and says so when the answer isn't there
* 📦 Batch uploads: one bad file doesn't block the others, and failures are reported per file
* ⚡ Lazy loading: the embedding model and LLM client are created on first use
* 🌐 FastAPI backend with a simple HTML + JS frontend
* 🧪 Unit tests for upload, retrieval filtering, and deletion

---

## 🧠 How it Works

1. Upload one or more PDFs. Each file is validated, saved to `uploads/`, and indexed once
2. Pages are loaded with `PyPDFLoader` and split into chunks (800 characters, 100 overlap)
3. Chunks are embedded with `all-MiniLM-L6-v2` and stored in a persistent Chroma collection with `source` and `page` metadata
4. A small `documents.json` manifest records the uploaded files so the list is restored on startup (any missing embeddings are rebuilt automatically)
5. On each question:
   * The top chunks are retrieved **separately from each selected PDF**
   * Each chunk is labelled with its filename and page and sent to the LLM with the question
   * The answer is returned together with the source filenames and pages

---

## 🏗️ Tech Stack

* Python
* LangChain
* FastAPI
* ChromaDB (`langchain-chroma`)
* HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
* Groq LLM (`openai/gpt-oss-120b`)

---

## 📂 Project Structure

```
project_rag/
│
├── rag.py                  # Backend (FastAPI + RAG pipeline)
├── api/
│   └── index.py            # Entry point for Vercel (imports the app)
├── static/                 # CSS / JS files
├── templates/
│   └── index.html          # Frontend UI
├── tests/
│   └── test_multi_pdf.py   # Unit tests
├── requirements.txt
├── vercel.json
├── .env                    # API key (ignored in git)
├── .gitignore
│
├── uploads/                # Uploaded PDFs (created automatically, ignored in git)
├── chroma_db/              # Vector store (created automatically, ignored in git)
└── documents.json          # Uploaded-document list (created automatically, ignored in git)
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

---

### 2. Create virtual environment

Python 3.11 or 3.12 is recommended; newer releases may not yet be supported by `chromadb` and `torch`.

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

---

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

### 4. Add API key

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

### 5. Run the app

```bash
uvicorn rag:app --reload
```

---

### 6. Open in browser

```
http://127.0.0.1:8000
```

Upload a PDF, tick it in the document list, and start asking questions.

---

### 7. Run the tests

```bash
python -m unittest tests.test_multi_pdf
```

---

## 📡 API Endpoints

### 🔹 Ask Question

```
POST /ask
```

**Request:**

```json
{
  "question": "What optimizer is used?",
  "selected_documents": ["your-uploaded-file.pdf"]
}
```

`selected_documents` must contain at least one uploaded filename. Questions are limited to 2000 characters.

**Response:**

```json
{
  "answer": "The paper uses the Adam optimizer...",
  "sources": [
    { "filename": "your-uploaded-file.pdf", "page": 7 }
  ]
}
```

If nothing relevant is found, the answer is `"I couldn't find that information in the selected documents."` with an empty `sources` list.

---

### 🔹 Upload PDFs

```
POST /upload-pdf
```

Send a multipart form with one or more `files` fields containing `.pdf` files (maximum 25 MB each). Each file is processed and embedded once.

**Response:**

```json
{
  "message": "PDFs are indexed and ready for questions.",
  "documents": [{ "filename": "a.pdf", "removable": true }],
  "failed": [{ "filename": "b.txt", "status": 400, "reason": "Please upload PDF files only." }]
}
```

Files that fail (wrong type, too large, duplicate name, unreadable) are listed in `failed` while the rest are still indexed. If every file fails, the request returns the first error as a normal HTTP error.

---

### 🔹 List documents

```
GET /documents
```

---

### 🔹 Remove an uploaded document

```
DELETE /documents/{filename}
```

Removes the document's chunks from the vector store and deletes the saved file. A bundled sample PDF, if present, cannot be removed.

---

### 🔹 Health Check

```
GET /health
```

Returns the server status, the number of loaded documents, and the last error (if any).

---

## 🔐 Security

* API keys are stored in `.env`
* `.env` is excluded via `.gitignore`
* Uploaded filenames are sanitised and saved with a random prefix
* Uploads are checked for a PDF header and a 25 MB size limit

---

## ⚙️ Configuration

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY` | Groq API key (required) |
| `DATA_DIR` | Where `uploads/`, `chroma_db/` and `documents.json` are stored. Defaults to the project folder locally, and to `/tmp` when running on Vercel |

---

## ⚠️ Notes

* The first upload or question may take time because the embedding model is downloaded on first use (internet required for initial setup)
* This is designed as a local, single-process application
* Optionally place `NIPS-2017-attention-is-all-you-need-Paper.pdf` in the project folder and it is registered as a bundled sample on startup
* **Deployment:** serverless hosts such as Vercel have a read-only filesystem apart from a temporary `/tmp` that is cleared between requests, and PyTorch is large for function size limits. For a public deployment, use a host with a persistent disk (Render, Railway, a Hugging Face Space), or move to a hosted vector database and hosted embeddings

---

## 📌 Future Improvements

* Chat-style UI (like ChatGPT)
* Inline source citations inside the answer text
* Hosted vector database and embeddings for serverless deployment
* Authentication & deployment

---

## 🙌 Acknowledgements

* LangChain ecosystem
* Groq for fast LLM inference
* Hugging Face sentence-transformers

---

## 📧 Contact

Feel free to reach out for collaboration or improvements!
