import unittest
from pathlib import Path
from tempfile import SpooledTemporaryFile

from fastapi import HTTPException, UploadFile

import rag


class FakeVectorStore:
    def __init__(self, results=None):
        self.results = results or []
        self.searches = []
        self.deleted_where = None

    def similarity_search(self, question, k, filter):
        self.searches.append({"question": question, "k": k, "filter": filter})
        return self.results

    def delete(self, where):
        self.deleted_where = where


class MultiPdfTests(unittest.TestCase):
    def setUp(self):
        self.original_upload_dir = rag.UPLOAD_DIR
        self.original_manifest_path = rag.MANIFEST_PATH
        self.original_documents = rag.documents
        self.original_indexer = rag.load_and_index_pdf
        self.original_vectorstore = rag.vectorstore
        self.original_indexed_documents = rag.indexed_documents
        rag.UPLOAD_DIR = Path(rag.BASE_DIR) / "uploads-test"
        rag.UPLOAD_DIR.mkdir(exist_ok=True)
        # never touch the real documents.json while testing
        rag.MANIFEST_PATH = rag.UPLOAD_DIR / "documents-test.json"
        rag.documents = {rag.PDF_PATH.name: {"path": rag.PDF_PATH, "removable": False}}
        rag.indexed_documents = {rag.PDF_PATH.name}
        self.indexed = []
        rag.load_and_index_pdf = lambda path, filename: self.indexed.append((path, filename))

    def tearDown(self):
        for path, _ in self.indexed:
            path.unlink(missing_ok=True)
        rag.MANIFEST_PATH.unlink(missing_ok=True)
        rag.UPLOAD_DIR = self.original_upload_dir
        rag.MANIFEST_PATH = self.original_manifest_path
        rag.documents = self.original_documents
        rag.load_and_index_pdf = self.original_indexer
        rag.vectorstore = self.original_vectorstore
        rag.indexed_documents = self.original_indexed_documents

    def test_multiple_pdfs_are_indexed_once_and_listed(self):
        result = rag.upload_pdfs([
            self.upload("first.pdf", b"%PDF-1.4\n"),
            self.upload("second.pdf", b"%PDF-1.4\n"),
        ])

        self.assertEqual([item["filename"] for item in result["documents"]], ["first.pdf", "second.pdf"])
        self.assertEqual(result["failed"], [])
        self.assertEqual([name for _, name in self.indexed], ["first.pdf", "second.pdf"])
        self.assertEqual([item["filename"] for item in rag.list_documents()["documents"]], [rag.PDF_PATH.name, "first.pdf", "second.pdf"])

    def test_one_bad_file_does_not_block_the_others(self):
        result = rag.upload_pdfs([
            self.upload("good.pdf", b"%PDF-1.4\n"),
            self.upload("notes.txt", b"not a pdf"),
        ])

        self.assertEqual([item["filename"] for item in result["documents"]], ["good.pdf"])
        self.assertEqual(result["failed"][0]["filename"], "notes.txt")
        self.assertEqual(result["failed"][0]["status"], 400)

    def test_duplicate_filename_is_rejected(self):
        rag.upload_pdfs([self.upload("dup.pdf", b"%PDF-1.4\n")])
        with self.assertRaises(HTTPException) as error:
            rag.upload_pdfs([self.upload("dup.pdf", b"%PDF-1.4\n")])
        self.assertEqual(error.exception.status_code, 409)

    def test_ask_filters_search_to_selected_document(self):
        store = FakeVectorStore()
        rag.vectorstore = store

        result = rag.ask(rag.QueryRequest(question="What is attention?", selected_documents=[rag.PDF_PATH.name]))

        self.assertIn("couldn't find", result["answer"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(store.searches[0]["filter"], {"source": rag.PDF_PATH.name})

    def test_ask_searches_each_selected_document(self):
        rag.upload_pdfs([self.upload("second.pdf", b"%PDF-1.4\n")])
        store = FakeVectorStore()
        rag.vectorstore = store

        rag.ask(rag.QueryRequest(question="Compare them", selected_documents=[rag.PDF_PATH.name, "second.pdf"]))

        self.assertEqual(
            [search["filter"] for search in store.searches],
            [{"source": rag.PDF_PATH.name}, {"source": "second.pdf"}],
        )
        self.assertTrue(all(search["k"] >= rag.MIN_CHUNKS_PER_DOC for search in store.searches))

    def test_removing_uploaded_document_deletes_its_chunks(self):
        rag.upload_pdfs([self.upload("remove-me.pdf", b"%PDF-1.4\n")])
        store = FakeVectorStore()
        rag.vectorstore = store

        rag.delete_document("remove-me.pdf")

        self.assertEqual(store.deleted_where, {"source": "remove-me.pdf"})
        self.assertNotIn("remove-me.pdf", rag.documents)
        self.assertNotIn("remove-me.pdf", rag.indexed_documents)

    def test_bundled_document_cannot_be_removed(self):
        with self.assertRaises(HTTPException) as error:
            rag.delete_document(rag.PDF_PATH.name)
        self.assertEqual(error.exception.status_code, 400)

    def test_invalid_extension_is_rejected(self):
        with self.assertRaises(HTTPException) as error:
            rag.upload_pdfs([self.upload("notes.txt", b"not a pdf")])
        self.assertEqual(error.exception.status_code, 400)

    def test_question_requires_a_selected_document(self):
        with self.assertRaises(HTTPException) as error:
            rag.ask(rag.QueryRequest(question="What is this?"))
        self.assertEqual(error.exception.status_code, 400)

    @staticmethod
    def upload(filename, content):
        file = SpooledTemporaryFile()
        file.write(content)
        file.seek(0)
        return UploadFile(file=file, filename=filename)


if __name__ == "__main__":
    unittest.main()
