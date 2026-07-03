"""
app.py
FastAPI service for Cost-Efficient RAG
"""
import os
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag import answer
from config import TOP_K, LOG_FILE

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag import answer
from config import TOP_K, LOG_FILE

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = FastAPI(
    title="Cost Efficient RAG API",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str
    k: int = TOP_K
    metadata_filter: dict | None = None

@app.get("/")
def health():
    return {"status": "ok", "service": "Cost Efficient RAG"}

@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.time()
    result = answer(
        question=req.question,
        k=req.k,
        metadata_filter=req.metadata_filter
    )
    print("RESULT:", result)
    total_latency = round(time.time() - start, 3)

    logging.info(
        "question=%s latency=%s retrieved=%s",
        req.question,
        total_latency,
        result.get("retrieved_chunks", 0)
    )

    return {
        "question": req.question,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "retrieved_chunks": result.get("retrieved_chunks", 0),
        "latency_seconds": total_latency
    }

# Run:
# uvicorn app:app --reload
