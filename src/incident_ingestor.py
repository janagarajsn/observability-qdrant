import os
import json
import time
import logging
from glob import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s")
logger = logging.getLogger(__name__)

# Configs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 1536))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 10))
BATCH_SLEEP_TIME = int(os.getenv("BATCH_SLEEP_TIME", 2))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PROJECT_ROOT = Path(__file__).parent.parent
INCIDENTS_FILES = PROJECT_ROOT / "incidents" / "*.json"
INGESTION_TRACKER_FILE = PROJECT_ROOT / "ingest-tracker" / "ingested_incidents.json"
QDRANT_CLOUD_API_KEY = os.getenv("QDRANT_CLOUD_API_KEY")

# Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_CLOUD_API_KEY)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Track ingested files
def load_ingested_files():
    if INGESTION_TRACKER_FILE.exists():
        with open(INGESTION_TRACKER_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_ingested_file(file_path):
    ingested = load_ingested_files()
    ingested.add(str(file_path))
    INGESTION_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INGESTION_TRACKER_FILE, "w") as f:
        json.dump(list(ingested), f)

# Create collection if not exists
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

# Ingest incidents
def ingest_incidents(collection_name: str):
    create_collection_if_not_exists(collection_name)
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)

    ingested_files = load_ingested_files()
    files = glob(str(INCIDENTS_FILES))
    if not files:
        logger.warning(f"No incident files found at {INCIDENTS_FILES}")
        return

    for file_path in files:
        if file_path in ingested_files:
            logger.info(f"Skipping already ingested file: {file_path}")
            continue

        logger.info(f"Ingesting incidents file: {file_path}")

        incidents = []
        with open(file_path, "r") as f:
            try:
                for line in f:
                    incident = json.loads(line)
                    text_content = f"{incident['title']}\n{incident['details']}\n{incident['impact']}\n{incident['resolution']}"
                    incidents.append(Document(page_content=text_content, metadata=incident))
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                continue

        # Ingest in batches
        for i in range(0, len(incidents), BATCH_SIZE):
            batch = incidents[i:i + BATCH_SIZE]
            vector_store.add_documents(batch)
            logger.info(f"Ingested batch {i // BATCH_SIZE + 1} ({len(batch)} incidents)")
            time.sleep(BATCH_SLEEP_TIME)

        save_ingested_file(file_path)
        logger.info(f"Finished ingestion of incidents into '{collection_name}'.")

# CLI support
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest generated incidents into Qdrant")
    parser.add_argument("--collection", type=str, default="incidents", help="Qdrant collection name")
    args = parser.parse_args()

    ingest_incidents(args.collection)
