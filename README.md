# Production RAG AI Assistant

A production-ready Retrieval-Augmented Generation (RAG) AI Assistant designed for portfolio demonstration. It features document upload, hybrid search, AI answers with citations, a beautiful glassmorphic Streamlit UI, and a scalable FastAPI backend, all fully Dockerized.

## Features
- **PDF Upload:** Automatically parse and chunk PDF documents.
- **Hybrid Search:** Combines vector similarity search (ChromaDB) with BM25 keyword search for robust retrieval.
- **Citations:** Every AI response includes direct citations back to the source documents.
- **FastAPI Backend:** High-performance async API with Prometheus monitoring endpoints.
- **Streamlit Frontend:** A sleek, dark-mode, glassmorphic chat interface.
- **Dockerized:** One command to spin up the entire stack, including monitoring (Prometheus + Grafana).

## Quickstart

1. Provide your OpenAI API key in the `.env` file (or set it in your environment). If you don't provide one, the backend will default to a local HuggingFace embedding model to test the processing, but generation requires the key.
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. Run the stack using Docker Compose:
   ```bash
   docker-compose up --build
   ```

3. Access the services:
   - **Frontend (Streamlit):** http://localhost:8501
   - **Backend API (FastAPI):** http://localhost:8000/docs
   - **Prometheus:** http://localhost:9090
   - **Grafana:** http://localhost:3000

## Architecture
- **Backend:** Python 3.11, FastAPI, Langchain, PyPDF2, ChromaDB, Rank-BM25
- **Frontend:** Streamlit, Custom CSS
- **Monitoring:** Prometheus, Grafana

### Data Flow
```text
User Question
     ↓
Frontend (Streamlit)
     ↓
FastAPI Backend
     ↓
Retriever
 ┌─────────────┐
 │ Vector DB   │
 │ BM25 Search │
 └─────────────┘
     ↓
Relevant Chunks
     ↓
LLM
     ↓
Answer + Citations
```
