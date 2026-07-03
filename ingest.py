
"""
Production-style ingest.py for Cost-Efficient RAG
"""
import os, hashlib, logging, argparse, time
from pathlib import Path
# from dotenv import load_dotenv
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,TextLoader)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# load_dotenv()

DOC_DIR=os.getenv("DOCUMENT_DIR","documents")
DB_DIR=os.getenv("CHROMA_DB_DIR","chroma_db")
CHUNK_SIZE=int(os.getenv("CHUNK_SIZE",500))
CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP",100))
MODEL=os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

emb=HuggingFaceEmbeddings(model_name=MODEL)
db=Chroma(persist_directory="./chroma_db", embedding_function=emb)

splitter=RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

def sha256(text:str)->str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def loader(path):
    ext=Path(path).suffix.lower()
    if ext==".pdf":
        return PyPDFLoader(path)
    elif ext in [".md",".txt"]:
        return TextLoader(path,encoding="utf-8")
    else:
     raise ValueError(f"Unsupported file: {path}")

def existing_hashes():
    hashes=set()
    data=db.get(include=["metadatas"])
    for m in data.get("metadatas",[]):
        if m and "chunk_hash" in m:
            hashes.add(m["chunk_hash"])
    return hashes

def ingest_file(path):
    docs=loader(path).load()
    chunks=splitter.split_documents(docs)
    seen=existing_hashes()
    texts,metas=[],[]
    added=skipped=0
    for i,ch in enumerate(tqdm(chunks,desc=Path(path).name)):
        h=sha256(ch.page_content)
        if h in seen:
            skipped+=1
            continue
        meta=dict(ch.metadata)
        meta.update({
            "chunk_hash":h,
            "chunk_id":i,
            "source":Path(path).name
        })
        texts.append(ch.page_content)
        metas.append(meta)
        seen.add(h)
        added+=1
    if texts:
        db.add_texts(texts=texts, metadatas=metas)
        
    logging.info("%s added=%d skipped=%d",path,added,skipped)

def ingest_directory(folder):
    for p in Path(folder).iterdir():
        if p.suffix.lower() in [".pdf",".md",".html",".htm"]:
            try:
                ingest_file(str(p))
            except Exception as e:
                logging.exception("Failed %s : %s",p,e)

def rebuild():
    import shutil
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR,exist_ok=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--directory",default=DOC_DIR)
    parser.add_argument("--rebuild",action="store_true")
    args=parser.parse_args()

    if args.rebuild:
        rebuild()

    t=time.time()
    if args.file:
        ingest_file(args.file)
    else:
        ingest_directory(args.directory)
    logging.info("Completed in %.2fs",time.time()-t)

if __name__=="__main__":
    main()
