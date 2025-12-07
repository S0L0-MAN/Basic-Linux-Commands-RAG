# rag_classic_corrected.py

from langchain_classic.chains import RetrievalQA
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS

# --- Configuration ---
OLLAMA_MODEL = "llama3.1:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
DOCS_FOLDER = "docs_to_load"  # Folder containing your .txt files

# --- 1️⃣ Set up the Ollama LLM ---
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0
)

# --- 2️⃣ Load documents from folder ---
loader = DirectoryLoader(DOCS_FOLDER, glob="**/*.txt", show_progress=True)
loaded_docs = loader.load()
print(f"Loaded {len(loaded_docs)} documents!")

# --- 3️⃣ Create embeddings using Ollama ---
embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

# --- 4️⃣ Create FAISS vector store from documents and embeddings ---
vector_store = FAISS.from_documents(loaded_docs, embeddings)

# --- 5️⃣ Create retriever ---
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# --- 6️⃣ Create RetrievalQA chain with source documents returned ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True  # important to get sources
)

# --- 7️⃣ Run a sample query using .invoke() ---
query = "How do I list files in Linux?"
result = qa_chain.invoke({"query": query})

print("\nAnswer:")
print(result["result"])

print("\nSource documents:")
for i, doc in enumerate(result["source_documents"]):
    print(f"Doc {i+1}: {doc.page_content[:300]}...\n")  # print first 300 chars of each doc
