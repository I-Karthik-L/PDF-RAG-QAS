const form = document.getElementById("question-form");
const questionInput = document.getElementById("question");
const submitButton = document.getElementById("submit-button");
const fileInput = document.getElementById("pdf-files");
const documentList = document.getElementById("document-list");
const documentCount = document.getElementById("document-count");
const chatHistory = document.getElementById("chat-history");
const statusMessage = document.getElementById("status-message");

let documents = [];
let selectedDocuments = new Set();

function setStatus(message) {
    statusMessage.textContent = message;
}

function escapeHtml(value) {
    const element = document.createElement("span");
    element.textContent = value;
    return element.innerHTML;
}

function renderDocuments() {
    documentCount.textContent = `${documents.length} document${documents.length === 1 ? "" : "s"} available`;
    if (!documents.length) {
        documentList.innerHTML = '<p class="empty-state">Upload a PDF to get started.</p>';
        return;
    }

    documentList.innerHTML = documents.map((document) => `
        <div class="document-row">
            <label class="document-select">
                <input type="checkbox" data-document="${escapeHtml(document.filename)}" ${selectedDocuments.has(document.filename) ? "checked" : ""}>
                <span>${escapeHtml(document.filename)}</span>
            </label>
            ${document.removable ? `<button class="remove-button" data-remove="${escapeHtml(document.filename)}" aria-label="Remove ${escapeHtml(document.filename)}">Remove</button>` : '<span class="bundled-tag">Bundled</span>'}
        </div>
    `).join("");

    documentList.querySelectorAll("input[data-document]").forEach((checkbox) => {
        checkbox.addEventListener("change", () => {
            const filename = checkbox.dataset.document;
            checkbox.checked ? selectedDocuments.add(filename) : selectedDocuments.delete(filename);
            setStatus(`${selectedDocuments.size} document${selectedDocuments.size === 1 ? "" : "s"} selected.`);
        });
    });
    documentList.querySelectorAll("button[data-remove]").forEach((button) => {
        button.addEventListener("click", () => removeDocument(button.dataset.remove));
    });
}

async function loadDocuments() {
    const response = await fetch("/documents");
    if (!response.ok) throw new Error("Could not load documents.");
    const data = await response.json();
    documents = data.documents;
    const available = new Set(documents.map((document) => document.filename));
    selectedDocuments = new Set([...selectedDocuments].filter((name) => available.has(name)));
    renderDocuments();
}

function appendMessage(kind, content, sources = []) {
    const sourceList = sources.length ? `<ul class="sources">${sources.map((source) => `<li>${escapeHtml(source.filename)}${source.page ? `, p. ${source.page}` : ""}</li>`).join("")}</ul>` : "";
    const message = document.createElement("article");
    message.className = `message ${kind}`;
    message.innerHTML = `<p class="message-label">${kind === "user" ? "You" : "Assistant"}</p><div class="message-text">${escapeHtml(content)}</div>${sourceList}`;
    chatHistory.appendChild(message);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

async function removeDocument(filename) {
    try {
        const response = await fetch(`/documents/${encodeURIComponent(filename)}`, {method: "DELETE"});
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Could not remove the document.");
        selectedDocuments.delete(filename);
        await loadDocuments();
        setStatus(data.message);
    } catch (error) {
        setStatus(error.message || "Could not remove the document.");
    }
}

fileInput.addEventListener("change", async () => {
    const files = [...fileInput.files];
    if (!files.length) return;
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    fileInput.disabled = true;
    setStatus("Uploading and indexing PDFs…");
    try {
        const response = await fetch("/upload-pdf", {method: "POST", body: formData});
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "The upload failed.");
        data.documents.forEach((document) => selectedDocuments.add(document.filename));
        await loadDocuments();
        setStatus(data.message);
    } catch (error) {
        setStatus(error.message || "The upload failed.");
    } finally {
        fileInput.value = "";
        fileInput.disabled = false;
    }
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return setStatus("Enter a question before sending it.");
    if (!selectedDocuments.size) return setStatus("Select at least one document first.");

    appendMessage("user", question);
    questionInput.value = "";
    submitButton.disabled = true;
    setStatus("Searching selected documents…");
    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({question, selected_documents: [...selectedDocuments]}),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "The answer could not be generated.");
        appendMessage("assistant", data.answer, data.sources || []);
        setStatus("Answer ready.");
    } catch (error) {
        appendMessage("assistant", error.message || "The answer could not be generated.");
        setStatus("Something went wrong.");
    } finally {
        submitButton.disabled = false;
    }
});

loadDocuments()
    .then(() => {
        selectedDocuments = new Set(documents.map((document) => document.filename));
        renderDocuments();
        setStatus("Select documents to begin.");
    })
    .catch((error) => setStatus(error.message));
