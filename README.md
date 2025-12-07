# LangChain + Ollama RAG Demo

## Overview

This repository contains a simple Retrieval-Augmented Generation (RAG) demo script using:

- **LangChain Classic API** (via `langchain-classic` package)  
- **Ollama local LLM** (`llama3.1` model)  
- **FAISS vector store** for efficient document retrieval  
- **Text files as knowledge base**

The script (`main.py`) loads text documents from a folder (`docs_to_load`), embeds them using Ollama embeddings, indexes them with FAISS, and runs queries to retrieve relevant information augmented by the LLM.

---

## Purpose

This project is primarily a **learning and experimentation sandbox** for understanding how to build RAG applications using LangChain and Ollama models locally.

The goal is to explore:

- Setting up an LLM with Ollama locally  
- Creating embeddings and vector stores with LangChain  
- Building a RetrievalQA chain for question answering over custom documents  

---

## How to Run
### 1. Ensure Ollama local server is running:

   
   ``` ollama serve```
   
Add your .txt documents inside the docs_to_load folder.

### 2.Install dependencies:

``` pip install -r requirements.txt```
```pip install langchain-classic langchain-ollama langchain-community faiss-cpu```

### 3.Run the script:

  ```python main.py```

### 4.The script will print an answer to the sample question along with snippets of the source documents.

## Future Scope and Plans

- Support for **more document formats** such as PDFs, Word docs, etc.
- Integration of a **web frontend** (using FastAPI, Streamlit, or similar) for interactive querying.
- Implementation of **custom prompt templates** to improve answer relevance.
- Experimentation with **different Ollama models** and parameter tuning.
- Extending to **multi-turn conversations** with context retention.
- Adding **logging, monitoring, and evaluation** for production readiness.
- Exploring **hybrid retrieval** combining keyword-based and semantic search.
- Packaging as a **reusable Python library or CLI tool**.

> ### Notes
> - This is an experimental demo for **educational purposes only**.
> - Uses `langchain-classic` due to recent API changes in the main LangChain package.
> - Requires Ollama to be installed and running locally.
