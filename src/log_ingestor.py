import os
import json
import time
import logging
from glob import glob
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load env
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s")
logger = logging.getLogger(__name__)

# Configs
PROJECT_ROOT = Path(__file__).parent
STATIC_LOG_FILES = PROJECT_ROOT.parent / "static-logs" / "*.json"
STREAM_LOG_FILE = PROJECT_ROOT.parent / "stream-logs" / "stream_logs.jsonl"
INGEST_TRACKER_FILE = PROJECT_ROOT.parent / "ingest-tracker" / "ingested_files.json"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 8))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 10))
BATCH_SLEEP_TIME = float(os.getenv("BATCH_SLEEP_TIME", 2))
LOG_BATCH = int(os.getenv("LOG_BATCH", 20))
STREAM_LOG_INTERVAL = int(os.getenv("STREAM_LOG_INTERVAL", 10))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
QDRANT_CLOUD_API_KEY = os.getenv("QDRANT_CLOUD_API_KEY")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_CLOUD_API_KEY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


# --- Utility functions ---
# Get ingested files information
def load_ingested_files():
    if INGEST_TRACKER_FILE.exists():
        with open(INGEST_TRACKER_FILE, "r") as f:
            return set(json.load(f))
    return set()

# Track ingested file to avoid re-ingesting
def save_ingested_file(file_path):
    ingested = load_ingested_files()
    ingested.add(str(file_path))
    INGEST_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INGEST_TRACKER_FILE, "w") as f:
        json.dump(list(ingested), f)

# Create Qdrant collection if not exists
def create_collection_if_not_exists(collection_name: str):
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name not in existing:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        logger.info(f"Collection '{collection_name}' created.")
    else:
        logger.info(f"Collection '{collection_name}' already exists.")


# --- Ingestion functions ---
# Ingest static log files into Qdrant collection
def ingest_static_files(collection_name: str):
    create_collection_if_not_exists(collection_name)
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)

    ingested_files = load_ingested_files()
    files = glob(str(STATIC_LOG_FILES))
    if not files:
        logger.warning(f"No static log files found at {STATIC_LOG_FILES}")
        return

    for file_path in files:
        if file_path in ingested_files:
            logger.info(f"Skipping already ingested file: {file_path}")
            continue

        logger.info(f"Ingesting static file: {file_path}")
        with open(file_path, "r") as f:
            try:
                logs = json.load(f)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                continue

        docs = []
        for i in range(0, len(logs), LOG_BATCH):
            batch = logs[i:i+LOG_BATCH]
            for log in batch:
                vector = log.get("logEmbedding") or embeddings.embed_query(log.get("message", ""))
                doc = Document(
                    page_content=log.get("message", ""),
                    metadata={**{k: log.get(k) for k in ["namespace","pod","application","level","timestamp"]}, "vector": vector}
                )
                docs.append(doc)

        # Chunk and add
        if docs:
            chunks = text_splitter.split_documents(docs)
            for i in range(0, len(chunks), BATCH_SIZE):
                vector_store.add_documents(chunks[i:i+BATCH_SIZE])
                time.sleep(BATCH_SLEEP_TIME)

            save_ingested_file(file_path)
            logger.info(f"Finished ingesting {len(chunks)} chunks from {file_path}")


def ingest_stream(collection_name: str, stream_file: Path, interval: int = STREAM_LOG_INTERVAL):
    """
    Continuously ingest logs from a streaming file (JSON lines)
    """
    create_collection_if_not_exists(collection_name)
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)

    stream_file.touch(exist_ok=True)
    with stream_file.open("r") as f:
        f.seek(0, os.SEEK_END)  # start at end of file
        while True:
            line = f.readline()
            if not line:
                time.sleep(interval)
                continue
            try:
                log = json.loads(line)
            except Exception as e:
                logger.error(f"Failed to parse stream line: {e}")
                continue

            vector = log.get("logEmbedding") or embeddings.embed_query(log.get("message", ""))
            doc = Document(
                page_content=log.get("message", ""),
                metadata={**{k: log.get(k) for k in ["namespace","pod","application","level","timestamp"]}, "vector": vector}
            )
            chunks = text_splitter.split_documents([doc])
            vector_store.add_documents(chunks)
            logger.info(f"Ingested log from stream: {log.get('timestamp')}")


# --- CLI ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest logs into Qdrant (static or streaming)")
    parser.add_argument("collection", type=str, help="Qdrant collection name")
    parser.add_argument("--stream", action="store_true", help="Enable streaming ingestion")
    parser.add_argument("--stream-file", type=str, default=str(STREAM_LOG_FILE), help="Streaming log file path")
    parser.add_argument("--interval", type=int, default=10, help="Polling interval in seconds for streaming")
    args = parser.parse_args()

    if args.stream:
        logger.info(f"Starting streaming ingestion into collection '{args.collection}'...")
        ingest_stream(args.collection, Path(args.stream_file), interval=args.interval)
    else:
        logger.info(f"Starting static ingestion into collection '{args.collection}'...")
        ingest_static_files(args.collection)
