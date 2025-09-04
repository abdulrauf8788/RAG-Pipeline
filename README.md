# 🧠 RAG Demo (Chroma, Pinecone, FAISS)

This repo showcases how to build a **Retrieval-Augmented Generation (RAG)** pipeline 
with three different vector stores: **ChromaDB**, **Pinecone**, and 
**FAISS**.

## 🚀 Features
- Load documents from `documents/` (PDF, TXT, DOCX).
- Split into chunks with metadata & deduplication.
- Sync with vector DB (add new docs, remove deleted docs).
- Simple chat interface in terminal with history (type `clear` or `quit`).

## ⚡ Quickstart

1. #### Clone repo & install deps
```bash
git clone https://github.com/abdulrauf8788/RAG-Pipeline.git
cd RAG-Pipeline

pip install -r requirements.txt
```

2. #### Setup your API Keys in .env
```bash
cp .env.example .env
# Add GOOGLE_API_KEY, PINECONE_API_KEY, etc.
```

3. #### Run and start chatting 🎉
```bash

# Run with Chroma 
python -m scripts.run_chroma

# Run with Pinecone 
python -m scripts.run_pinecone

# Run with FAISS 
python -m scripts.run_faiss

```
