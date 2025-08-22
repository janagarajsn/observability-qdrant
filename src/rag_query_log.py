import os
import sys
import logging
from typing import List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv
from pydantic import PrivateAttr
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever, Document

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s")
logger = logging.getLogger(__name__)

# Core configs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://host.docker.internal:6333")
QDRANT_CLOUD_API_KEY = os.getenv("QDRANT_CLOUD_API_KEY")
COLLECTION_LOGS = os.getenv("COLLECTION_NAME", "aks_logs")                # main logs collection (text)
COLLECTION_SCREENSHOTS = os.getenv("COLLECTION_SCREENSHOTS", "log_shots") # optional, images/diagrams
COLLECTION_INCIDENTS = os.getenv("COLLECTION_INCIDENTS", "incidents")     # optional, incident writeups
EMBEDDING_MODEL_TEXT = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# For hackathon simplicity, we reuse a text model for images; swap to a vision model in your infra
EMBEDDING_MODEL_IMAGE = os.getenv("IMAGE_EMBEDDING_MODEL", EMBEDDING_MODEL_TEXT)
RETRIEVAL_MODEL = os.getenv("RETRIEVAL_MODEL", "gpt-4.1-nano")
DEFAULT_K = int(os.getenv("DEFAULT_K", 5))
THRESHOLD_LIMIT = float(os.getenv("THRESHOLD_LIMIT", 0.5))
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", 10))

# Qdrant Client and Embedding stores
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_CLOUD_API_KEY)
text_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_TEXT)
image_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_IMAGE)  # replace with vision model if available

def _collection_exists(name: str) -> bool:
    try:
        collections = qdrant_client.get_collections().collections
        return any(c.name == name for c in collections)
    except ResponseHandlingException:
        return False

def _points_count(name: str) -> int:
    try:
        info = qdrant_client.get_collection(name)
        return getattr(info, "points_count", 0) or getattr(info.status, "points_count", 0) or 0
    except Exception:
        return 0

if not _collection_exists(COLLECTION_LOGS):
    logger.error(f"Collection '{COLLECTION_LOGS}' does not exist. Run ingestion first.")
    sys.exit(1)

if _points_count(COLLECTION_LOGS) == 0:
    logger.error(f"Collection '{COLLECTION_LOGS}' is empty. Run ingestion first.")
    sys.exit(1)
else:
    logger.info(f"Collection '{COLLECTION_LOGS}' is ready with points: {_points_count(COLLECTION_LOGS)}")

# Primary vector store for log text
log_vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_LOGS,
    embedding=text_embeddings
)

# Optional stores (created only if collections exist)
screenshot_store = None
if _collection_exists(COLLECTION_SCREENSHOTS):
    screenshot_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_SCREENSHOTS,
        embedding=image_embeddings
    )
    logger.info(f"Screenshot collection detected: '{COLLECTION_SCREENSHOTS}' with points: {_points_count(COLLECTION_SCREENSHOTS)}")
else:
    logger.warning(f"Screenshot collection '{COLLECTION_SCREENSHOTS}' not found (screenshot search disabled).")

incident_store = None
if _collection_exists(COLLECTION_INCIDENTS):
    incident_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_INCIDENTS,
        embedding=text_embeddings
    )
    logger.info(f"Incidents collection detected: '{COLLECTION_INCIDENTS}' with points: {_points_count(COLLECTION_INCIDENTS)}")
else:
    logger.warning(f"Incidents collection '{COLLECTION_INCIDENTS}' not found (recommendations disabled).")


# =============== HELPERS ===============
def embed_text(text: str) -> List[float]:
    return text_embeddings.embed_query(text)

def embed_image_descriptor(descriptor: str) -> List[float]:
    """
    Hackathon-friendly placeholder: embed a textual descriptor for an image/diagram.
    Replace with a real image-embedding (CLIP, OpenAI vision) when available.
    """
    return image_embeddings.embed_query(f"[image] {descriptor}")

def parse_filters(filter_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Parse filters like: "level=ERROR,namespace=namespace-3"
    Returns a dict suitable for langchain_qdrant filter= param.
    """
    if not filter_str:
        return None
    parts = [p.strip() for p in filter_str.split(",") if p.strip()]
    result: Dict[str, Any] = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            result[k.strip()] = v.strip()
    return result or None

def attach_score(docs_and_scores: List[Tuple[Document, float]], threshold: float) -> List[Document]:
    filtered: List[Document] = []
    for doc, score in docs_and_scores:
        if score >= threshold:
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["similarity_score"] = score
            filtered.append(doc)
    return filtered

def flag_anomaly(avg_score: float, threshold: float) -> bool:
    """
    Simple anomaly flag: if average score of returned docs < threshold, mark as anomaly.
    """
    return avg_score < threshold


# =============== RETRIEVER WITH THRESHOLD + FILTERS ===============
class ThresholdRetriever(BaseRetriever):
    _vectorstore: QdrantVectorStore = PrivateAttr()
    _k: int = PrivateAttr()
    _threshold: float = PrivateAttr()
    _filters: Optional[Dict[str, Any]] = PrivateAttr()

    def __init__(self, vectorstore: QdrantVectorStore, k: int, threshold: float, filters: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._vectorstore = vectorstore
        self._k = k
        self._threshold = threshold
        self._filters = filters

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs_and_scores = self._vectorstore.similarity_search_with_score(query, k=self._k, filter=self._filters)
        filtered_docs = attach_score(docs_and_scores, self._threshold)
        if not filtered_docs:
            logger.info(f"No documents passed the similarity score threshold={self._threshold}")
        return filtered_docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


# =============== CORE QUERY FUNCTIONS ===============
# Plain Text Query
def query_logs_text(
    query_text: str,
    k: int = DEFAULT_K,
    threshold: float = THRESHOLD_LIMIT,
    filters: Optional[Dict[str, Any]] = None,
    with_incident_info: bool = True
) -> Tuple[str, List[Document], List[Document], Dict[str, Any]]:
    """
    Classic text-based RAG over logs with threshold + optional incident recommendations.
    Returns: (answer, docs, diagnostics)
    """
    logger.info(f"Text query: '{query_text}', k={k}, threshold={threshold}, filters={filters}")
    retriever = ThresholdRetriever(vectorstore=log_vector_store, k=k, threshold=threshold, filters=filters)

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model=RETRIEVAL_MODEL, api_key=OPENAI_API_KEY, temperature=0),
        retriever=retriever,
        return_source_documents=True
    )

    response = qa_chain.invoke({"query": query_text})
    log_docs: List[Document] = response.get("source_documents", []) or []
    if not log_docs:
        return "Sorry, the question is out of my scope", [], [], {"anomaly": True, "avg_score": 0.0}

    # Add incident recommendations (optional)
    incident_docs: List[Document] = []
    if with_incident_info and incident_store:
        # Use plain similarity search (no threshold on incidents, keep it exploratory)
        incident_docs = incident_store.similarity_search(query_text, k=min(3, k))
        for d in incident_docs:
            d.metadata = dict(d.metadata or {})
            d.metadata["source_type"] = "incident_info"

    # Compute simple anomaly indicator
    scores = [d.metadata.get("similarity_score", 0.0) for d in log_docs]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    diagnostics = {"avg_score": avg_score, "anomaly": flag_anomaly(avg_score, threshold)}

    return response["result"], log_docs, incident_docs, diagnostics

# Screenshot Query
def query_logs_screenshot(
    screenshot_descriptor: str,
    k: int = DEFAULT_K,
    threshold: float = THRESHOLD_LIMIT,
    filters: Optional[Dict[str, Any]] = None
) -> Tuple[str, List[Document], Dict[str, Any]]:
    """
    Screenshot / diagram search using a textual descriptor (e.g., filename or caption).
    In production, replace with real image embeddings. Requires screenshot collection.
    """
    if not screenshot_store:
        logger.warning("Screenshot collection not available; screenshot search disabled.")
        return "Screenshot search is not configured.", [], {"anomaly": True, "avg_score": 0.0}

    logger.info(f"Screenshot query: '{screenshot_descriptor}', k={k}, threshold={threshold}, filters={filters}")
    vec = embed_image_descriptor(screenshot_descriptor)

    # Direct vector search using the screenshot store
    # langchain_qdrant doesn't expose "search_with_vector", but we can use similarity_search_by_vector
    docs_and_scores = screenshot_store.similarity_search_with_score_by_vector(vec, k=k, filter=filters)
    filtered_docs = attach_score(docs_and_scores, threshold)

    if not filtered_docs:
        return "No relevant logs/diagrams for this screenshot descriptor.", [], {"anomaly": True, "avg_score": 0.0}

    scores = [d.metadata.get("similarity_score", 0.0) for d in filtered_docs]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return "Screenshot-related results retrieved.", filtered_docs, {"avg_score": avg_score, "anomaly": flag_anomaly(avg_score, threshold)}

# Hybrid Query
def query_logs_hybrid(
    query_text: Optional[str],
    screenshot_descriptor: Optional[str],
    k: int = DEFAULT_K,
    threshold: float = THRESHOLD_LIMIT,
    filters: Optional[Dict[str, Any]] = None,
    alpha_text: float = 0.5
) -> Tuple[str, List[Document], Dict[str, Any]]:
    """
    Hybrid retrieval:
      - Text mode over log collection
      - Screenshot mode over screenshot collection
      - Weighted late fusion of results (alpha_text controls blend)
    """
    if not query_text and not screenshot_descriptor:
        return "Provide text and/or screenshot descriptor.", [], {"anomaly": True, "avg_score": 0.0}

    results: List[Tuple[Document, float, str]] = []  # (doc, score, source_collection)

    # Text arm
    if query_text:
        docs_scores_text = log_vector_store.similarity_search_with_score(query_text, k=k, filter=filters)
        for d, s in docs_scores_text:
            if s >= threshold:
                d.metadata = dict(d.metadata or {})
                d.metadata["similarity_score"] = s
                results.append((d, s, "logs"))

    # Screenshot arm
    if screenshot_descriptor and screenshot_store:
        vec = embed_image_descriptor(screenshot_descriptor)
        docs_scores_img = screenshot_store.similarity_search_with_score_by_vector(vec, k=k, filter=filters)
        for d, s in docs_scores_img:
            if s >= threshold:
                d.metadata = dict(d.metadata or {})
                d.metadata["similarity_score"] = s
                results.append((d, s, "screenshots"))

    if not results:
        return "No relevant results for hybrid query.", [], {"anomaly": True, "avg_score": 0.0}

    # Late fusion: normalize scores per source and blend
    from collections import defaultdict
    scores_by_id: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    for d, s, src in results:
        # try a stable ID or fall back to hash of payload
        doc_id = d.metadata.get("id") or d.metadata.get("traceId") or d.page_content[:64]
        scores_by_id[doc_id].append((s, src))

    def blend(score_list: List[Tuple[float, str]]) -> float:
        # Normalize per source and blend via alpha
        logs_scores = [s for s, src in score_list if src == "logs"]
        shot_scores = [s for s, src in score_list if src == "screenshots"]
        s_logs = max(logs_scores) if logs_scores else 0.0
        s_shots = max(shot_scores) if shot_scores else 0.0
        return alpha_text * s_logs + (1 - alpha_text) * s_shots

    blended = []
    for doc, s, src in results:
        doc_id = doc.metadata.get("id") or doc.metadata.get("traceId") or doc.page_content[:64]
        blended_score = blend(scores_by_id[doc_id])
        doc.metadata["blended_score"] = blended_score
        blended.append((doc, blended_score))

    blended.sort(key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in blended[:k]]

    scores = [d.metadata.get("blended_score", d.metadata.get("similarity_score", 0.0)) for d in top_docs]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return "Hybrid results retrieved.", top_docs, {"avg_score": avg_score, "anomaly": flag_anomaly(avg_score, threshold)}


# =============== CLI ===============
def _print_results(answer: str, log_docs: List[Document], diag: dict, incident_docs: List[Document]=[]):
    print("\n=== Answer ===")
    print(answer)
    print("\n=== Diagnostics ===")
    print(diag)
    print("\n=== Log Sources ===")
    
    for i, d in enumerate(log_docs, 1):
        meta = dict(d.metadata or {})
        snippet = d.page_content[:160].replace("\n", " ")
        # Safely extract selected metadata
        meta_info = {key: meta[key] for key in ['level', 'namespace', 'timestamp'] if key in meta}
        score = meta.get('similarity_score', meta.get('blended_score', 'N/A'))
        print(f"{i:02d}. score={score} | meta={meta_info} | {snippet}...")

    if incident_docs:
        print("\n=== Incident References ===")
        for i, d in enumerate(incident_docs, 1):
            meta = dict(d.metadata or {})
            snippet = d.page_content[:160].replace("\n", " ")
            incident_id = meta.get("id", "INC-UNKNOWN")
            print(f"{i:02d}. id={incident_id} | {snippet}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-modal RAG over logs (Qdrant + threshold + filters)")
    parser.add_argument("--mode", choices=["text", "reasoning", "action", "screenshot", "hybrid"], default="text", help="Query mode")
    parser.add_argument("--query", type=str, default=None, help="Text query (for text/hybrid)")
    parser.add_argument("--screenshot", type=str, default=None, help="Screenshot descriptor (filename/caption) for screenshot/hybrid")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-K results")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_LIMIT, help="Cosine similarity threshold")
    parser.add_argument("--filters", type=str, default=None, help="Comma-separated filters, e.g. 'level=ERROR,namespace=namespace-3'")
    parser.add_argument("--incidenthistory", action="store_true", help="Fetch similar incidents reported in the history (Only for text mode)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid blend weight for text arm (0..1)")
    args = parser.parse_args()

    filt = parse_filters(args.filters)

    if args.mode == "text":
        if not args.query:
            print("Error: --query is required for text mode")
            sys.exit(2)
        answer, log_docs, incident_docs, diag = query_logs_text(
            query_text=args.query,
            k=args.k,
            threshold=args.threshold,
            filters=filt,
            with_incident_info=args.incidenthistory
        )
        _print_results(answer, log_docs, diag, incident_docs=incident_docs)

    elif args.mode in ("reason", "action"):
        from reasoning import query_logs_reasoned
        if not args.query:
            print("Error: --query is required for text mode")
            sys.exit(2)
        answer, log_docs, incident_docs, diag = query_logs_reasoned(
            query_text=args.query,
            k=args.k,
            threshold=args.threshold,
            filters=filt,
            with_incident_info=args.incidenthistory
        )
        _print_results(answer, log_docs, diag, incident_docs=incident_docs)

    elif args.mode == "screenshot":
        if not args.screenshot:
            print("Error: --screenshot is required for screenshot mode")
            sys.exit(2)
        answer, docs, diag = query_logs_screenshot(
            screenshot_descriptor=args.screenshot,
            k=args.k,
            threshold=args.threshold,
            filters=filt
        )
        _print_results(answer, docs, diag)

    else:  # hybrid
        if not args.query and not args.screenshot:
            print("Error: at least one of --query or --screenshot is required for hybrid mode")
            sys.exit(2)
        answer, docs, diag = query_logs_hybrid(
            query_text=args.query,
            screenshot_descriptor=args.screenshot,
            k=args.k,
            threshold=args.threshold,
            filters=filt,
            alpha_text=args.alpha
        )
        _print_results(answer, docs, diag)
