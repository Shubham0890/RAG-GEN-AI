# Cost-Efficient RAG Application

## Overview
This project implements a Retrieval-Augmented Generation (RAG) application using **ChromaDB** as a low-cost vector store.

## Features
- PDF / HTML / Markdown ingestion
- Configurable chunk size and overlap
- SHA-256 idempotent re-ingestion
- ChromaDB persistent vector database
- Metadata filtering
- Top-k retrieval
- Grounded LLM responses with source citations
- FastAPI REST API
- Retrieval evaluation (Recall@k, Hit Rate, MRR, nDCG@k)
- Query logging
- Environment-based configuration

## Project Structure
```
cost-efficient-rag/
├── app.py
├── config.py
├── ingest.py
├── rag.py
├── evaluate.py
├── requirements.txt
├── .env
├── documents/
├── chroma_db/
├── logs/
└── evaluation/
```

## Installation
```bash
python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

## Configure
Copy `.env.example` to `.env` and set:
```
OPENAI_API_KEY=your_key
```

## Ingest Documents
```bash
python ingest.py
```

## Run API
```bash
uvicorn app:app --reload
```

## Example Request
POST /query

```json
{
  "question":"What is RAG?",
  "k":5
}
```

## Evaluation
```bash
python evaluate.py
```

Outputs:
- evaluation/results.csv

Metrics:
- Recall@k
- Hit Rate
- MRR
- nDCG@k

## Cost Comparison (Illustrative)

| Vectors | Managed DB | ChromaDB |
|----------|------------|----------|
|100K|~$80/month|Disk only|
|1M|~$300/month|Disk only|
|10M|~$1800/month|Disk only|

Assumptions:
- 384-dimensional embeddings
- 4 bytes per float
- Local persistent storage

## Trade-offs

Advantages
- Low cost
- Easy deployment
- Persistent storage
- Metadata filtering

Limitations
- Single-node deployment
- Manual backups
- Limited horizontal scaling

## Deliverables
- Runnable FastAPI service
- ChromaDB vector store
- Evaluation pipeline
- Cost comparison
- README
