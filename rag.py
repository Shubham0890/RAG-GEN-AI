"""
rag.py
Retrieval and grounded generation for Cost-Efficient RAG
"""

import time
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from config import (
    GOOGLE_API_KEY,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    TOP_K
)

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

vectordb = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embedding
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=GOOGLE_API_KEY
)

PROMPT = PromptTemplate.from_template("""
You are a helpful assistant.

Use ONLY the supplied context.

If the answer is not present, reply:
"I don't have enough information from the provided documents."

Context:
{context}

Question:
{question}
""")


def retrieve(question: str, k: int = TOP_K, metadata_filter=None):
    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": k,
            "filter": metadata_filter
        }
    )
    return retriever.invoke(question)


def answer(question: str, k: int = TOP_K, metadata_filter=None):
    start = time.time()

    try:
        docs = retrieve(question, k, metadata_filter)

        if not docs:
            return {
                "answer": "I don't have enough information from the provided documents.",
                "sources": [],
                "retrieved_chunks": 0,
                "latency": round(time.time() - start, 3)
            }

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = PROMPT.format(
            context=context,
            question=question
        )

        response = llm.invoke(prompt)

        sources = []

        for d in docs:
            sources.append({
                "source": d.metadata.get("source"),
                "chunk_id": d.metadata.get("chunk_id")
            })

        return {
            "answer": response.content,
            "sources": sources,
            "retrieved_chunks": len(docs),
            "latency": round(time.time() - start, 3)
        }

    except Exception as e:
        print("ERROR:", e)
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "retrieved_chunks": 0,
            "latency": round(time.time() - start, 3)
        }