import os
from typing import List, Tuple, AsyncGenerator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory

from config import settings

# Global state
vector_store = None
bm25_retriever = None


def get_embeddings():
    """Always use local HuggingFace embeddings — free, no API key needed."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)


def get_llm():
    """Priority: Groq → Ollama → OpenAI. Raises if none configured."""
    # 1. Groq (free tier, fast)
    if settings.GROQ_API_KEY:
        return ChatGroq(
            model="llama3-8b-8192",
            api_key=settings.GROQ_API_KEY,
            temperature=0,
        )
    # 2. Ollama (local, offline)
    if settings.OLLAMA_BASE_URL:
        from langchain_community.llms import Ollama
        return Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
    # 3. OpenAI
    if settings.OPENAI_API_KEY:
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
    raise ValueError(
        "No LLM configured. Set GROQ_API_KEY, OLLAMA_BASE_URL, or OPENAI_API_KEY in .env"
    )


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def init_vector_store():
    global vector_store
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )


def rebuild_bm25():
    """Rebuild BM25 index from all documents stored in Chroma."""
    global bm25_retriever, vector_store
    if vector_store is None:
        return
    result = vector_store.get()
    if result and result.get("documents"):
        docs = [
            Document(page_content=result["documents"][i], metadata=result["metadatas"][i])
            for i in range(len(result["documents"]))
        ]
        bm25_retriever = BM25Retriever.from_documents(docs)


# ---------------------------------------------------------------------------
# Document processing  (supports multiple PDFs)
# ---------------------------------------------------------------------------

def process_and_store_document(file_path: str, original_filename: str) -> int:
    """Process a single PDF: load → chunk → embed → store. Returns chunk count."""
    global vector_store

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = original_filename

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    if vector_store is None:
        init_vector_store()

    vector_store.add_documents(chunks)
    vector_store.persist()
    rebuild_bm25()

    return len(chunks)


# ---------------------------------------------------------------------------
# Chat memory  (SQLite‑backed, persists across restarts)
# ---------------------------------------------------------------------------

def get_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=settings.MEMORY_DB_URL,
    )


# ---------------------------------------------------------------------------
# Synchronous chat  (returns full answer at once)
# ---------------------------------------------------------------------------

def chat(session_id: str, message: str) -> Tuple[str, List[dict]]:
    global vector_store, bm25_retriever

    if vector_store is None:
        init_vector_store()
        rebuild_bm25()

    # Build retriever list
    retriever_list = []
    if vector_store:
        retriever_list.append(vector_store.as_retriever(search_kwargs={"k": 4}))
    if bm25_retriever:
        retriever_list.append(bm25_retriever)

    if not retriever_list:
        return "Please upload a document first before asking questions.", []

    if len(retriever_list) == 2:
        ensemble = EnsembleRetriever(retrievers=retriever_list, weights=[0.5, 0.5])
    else:
        ensemble = retriever_list[0]

    docs = ensemble.invoke(message)

    # Build context string
    context_text = "\n\n".join(
        f"[Source: {d.metadata.get('source','Unknown')} | Page {d.metadata.get('page','?')}]\n{d.page_content}"
        for d in docs
    )

    citations = [
        {
            "source": str(d.metadata.get("source", "Unknown")),
            "page": d.metadata.get("page"),
            "content": d.page_content[:250],
        }
        for d in docs
    ]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful AI assistant. Answer using ONLY the context below. "
            "Cite sources with [Source: filename | Page N]. "
            "If you cannot answer, say so.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    try:
        llm = get_llm()
        chain = prompt | llm
        history = get_history(session_id)

        response = chain.invoke({
            "context": context_text,
            "history": history.messages,
            "question": message,
        })

        answer = response if isinstance(response, str) else response.content

        history.add_user_message(message)
        history.add_ai_message(answer)

    except Exception as e:
        answer = f"Error generating response: {e}"
        citations = []

    return answer, citations


# ---------------------------------------------------------------------------
# Streaming chat  (yields tokens one‑by‑one)
# ---------------------------------------------------------------------------

async def chat_stream(session_id: str, message: str) -> AsyncGenerator[str, None]:
    """Async generator that yields answer tokens and appends to memory when done."""
    global vector_store, bm25_retriever

    if vector_store is None:
        init_vector_store()
        rebuild_bm25()

    retriever_list = []
    if vector_store:
        retriever_list.append(vector_store.as_retriever(search_kwargs={"k": 4}))
    if bm25_retriever:
        retriever_list.append(bm25_retriever)

    if not retriever_list:
        yield "Please upload a document first."
        return

    if len(retriever_list) == 2:
        ensemble = EnsembleRetriever(retrievers=retriever_list, weights=[0.5, 0.5])
    else:
        ensemble = retriever_list[0]

    docs = ensemble.invoke(message)

    context_text = "\n\n".join(
        f"[Source: {d.metadata.get('source','Unknown')} | Page {d.metadata.get('page','?')}]\n{d.page_content}"
        for d in docs
    )

    # Yield citations header
    citations_header = "\n\n---\n**📚 Sources:**\n"
    for i, d in enumerate(docs):
        citations_header += f"- [{i+1}] {d.metadata.get('source','Unknown')} (Page {d.metadata.get('page','?')})\n"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful AI assistant. Answer using ONLY the context below. "
            "Cite sources with [Source: filename | Page N]. "
            "If you cannot answer, say so.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    try:
        llm = get_llm()
        chain = prompt | llm
        history = get_history(session_id)

        full_answer = ""
        async for chunk in chain.astream({
            "context": context_text,
            "history": history.messages,
            "question": message,
        }):
            token = chunk if isinstance(chunk, str) else chunk.content
            full_answer += token
            yield token

        # Yield citations at the end
        yield citations_header

        # Persist to memory
        history.add_user_message(message)
        history.add_ai_message(full_answer)

    except Exception as e:
        yield f"\n\nError: {e}"
