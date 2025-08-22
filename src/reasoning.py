# reasoning.py
from typing import Any, Dict, List, Tuple, Optional
import json
import re
import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
# Import your existing RAG retrieval functions
from rag_query_log import query_logs_text, Document

# Load Environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s")
logger = logging.getLogger(__name__)

# Configs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://host.docker.internal:6333")
QDRANT_CLOUD_API_KEY = os.getenv("QDRANT_CLOUD_API_KEY")
RETRIEVAL_MODEL = os.getenv("RETRIEVAL_MODEL", "gpt-4.1-nano")

# Open AI Client
client = OpenAI()

@dataclass
class ReasoningOutput:
    answer: str
    root_cause: Optional[str]
    recommended_actions: List[Dict[str, Any]]
    confidence: float
    insufficient_evidence: bool
    evidence_doc_ids: List[str]

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "root_cause": {"type": ["string", "null"]},
        "recommended_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "why": {"type": "string"},
                    "priority": {"type": "string"},
                    "owner": {"type": "string"},
                    "runbook": {"type": "string"}
                },
                "required": ["action", "why"]
            }
        },
        "confidence": {"type": "number"},
        "insufficient_evidence": {"type": "boolean"},
        "evidence_doc_ids": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["answer", "recommended_actions", "confidence", "insufficient_evidence", "evidence_doc_ids"]
}

def _summarize_docs(docs: List[Document], max_items: int = 12, max_chars: int = 240) -> str:
    """Summarize retrieved documents for LLM prompt."""
    logger.info(f"Summarizing {len(docs)} documents (max_items={max_items}, max_chars={max_chars})")
    
    lines = []
    for i, d in enumerate(docs[:max_items]):
        src = d.metadata.get("source") or d.metadata.get("file") or d.metadata.get("id") or "unknown"
        ts  = d.metadata.get("timestamp") or d.metadata.get("time") or ""
        msg = (d.page_content or "").strip().replace("\n", " ")
        original_length = len(msg)
        
        if len(msg) > max_chars:
            msg = msg[:max_chars-3] + "..."
            logger.info(f"Document {i} truncated from {original_length} to {len(msg)} chars")
        
        did = d.metadata.get("id") or d.metadata.get("doc_id") or src
        lines.append(f"- [{did}] {ts} {src}: {msg}")
    
    if not lines:
        logger.warning("No documents to summarize")
        return "(no documents)"
    
    joined_lines = "\n".join(lines)
    logger.info(f"Summarized {len(lines)} documents into {len(joined_lines)} characters")
    return joined_lines

def _build_prompt(query_text: str, logs_summary: str, incidents_summary: str) -> str:
    logger.info(f"Building prompt for query: '{query_text[:100]}{'...' if len(query_text) > 100 else ''}'")
    logger.info(f"Logs summary length: {len(logs_summary)} chars")
    logger.info(f"Incidents summary length: {len(incidents_summary)} chars")
    
    prompt = f"""
You are an SRE assistant. Use only the provided context to answer. 
If evidence is weak or missing, set "insufficient_evidence" true and avoid speculation.

## User Question
{query_text}

## Logs (summarized)
{logs_summary}

## Past Incidents (summarized)
{incidents_summary}

## Output JSON schema
{json.dumps(JSON_SCHEMA)}

Respond with only valid JSON matching the schema. Do not include explanations outside JSON.
"""
    
    logger.info(f"Built prompt with total length: {len(prompt)} characters")
    return prompt

def call_llm_json(prompt: str) -> Dict[str, Any]:
    """
    Calls OpenAI Chat model with temperature=0 and returns JSON matching the schema.
    """
    logger.info(f"Calling LLM with prompt length: {len(prompt)} characters")
    try:
        resp = client.chat.completions.create(
            model=RETRIEVAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = resp.choices[0].message.content
        logger.info(f"Received response from LLM: {text[:200]}...")  # log first 200 chars
        return _extract_first_json(text)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        # Return fallback structure so reasoning.py still returns 4-tuple
        return {
            "answer": f"LLM call failed: {type(e).__name__}",
            "root_cause": None,
            "recommended_actions": [],
            "confidence": 0.0,
            "insufficient_evidence": True,
            "evidence_doc_ids": []
        }

def _extract_first_json(text: str) -> Dict[str, Any]:
    """Extract first top-level JSON object from LLM text output."""
    logger.info(f"Extracting JSON from text of length: {len(text)}")
    
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        logger.error(f"No JSON found in LLM output. Text preview: '{text[:200]}{'...' if len(text) > 200 else ''}'")
        raise ValueError("No JSON found in LLM output")
    
    block = m.group(0)
    logger.info(f"Found JSON block of length: {len(block)}")
    
    try:
        parsed = json.loads(block)
        logger.info("Successfully parsed JSON from LLM output")
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}. JSON block: '{block[:500]}{'...' if len(block) > 500 else ''}'")
        raise

def query_logs_reasoned(
    query_text: str,
    k: int = 10,
    threshold: float = 0.25,
    filters: Optional[Dict[str, Any]] = None,
    with_incident_info: bool = True
) -> Tuple[str, List[Document], List[Document], Dict[str, Any]]:
    """
    Retrieve logs/incidents -> summarize -> call reasoning LLM -> return structured answer.
    Returns: (answer, log_docs, incident_docs, diagnostics)
    """
    logger.info(f"Starting reasoned query: '{query_text[:100]}{'...' if len(query_text) > 100 else ''}'")
    logger.info(f"Parameters: k={k}, threshold={threshold}, filters={filters}, with_incident_info={with_incident_info}")
    
    # 1) Retrieve context using existing RAG function
    logger.info("Retrieving logs and incidents using RAG")
    answer, log_docs, incident_docs, diag = query_logs_text(
        query_text=query_text,
        k=k,
        threshold=threshold,
        filters=filters,
        with_incident_info=with_incident_info
    )
    
    logger.info(f"Retrieved {len(log_docs)} log documents and {len(incident_docs)} incident documents")

    logs_summary = _summarize_docs(log_docs)
    incidents_summary = _summarize_docs(incident_docs)

    prompt = _build_prompt(query_text, logs_summary, incidents_summary)

    logger.info("Calling LLM for reasoning")
    try:
        raw = call_llm_json(prompt)
        logger.info("Successfully received LLM response")
        logger.info(f"LLM response keys: {list(raw.keys())}")
    except NotImplementedError:
        logger.warning("LLM provider not implemented, returning default response")
        raw = {
            "answer": "Reasoning provider not configured.",
            "root_cause": None,
            "recommended_actions": [],
            "confidence": 0.0,
            "insufficient_evidence": True,
            "evidence_doc_ids": []
        }
    except Exception as e:
        logger.error(f"LLM call failed: {type(e).__name__}: {e}")
        raw = {
            "answer": f"Reasoning failed: {type(e).__name__}",
            "root_cause": None,
            "recommended_actions": [],
            "confidence": 0.0,
            "insufficient_evidence": True,
            "evidence_doc_ids": []
        }

    logger.info("Creating ReasoningOutput from LLM response")
    ro = ReasoningOutput(
        answer=raw.get("answer", ""),
        root_cause=raw.get("root_cause"),
        recommended_actions=raw.get("recommended_actions", []),
        confidence=float(raw.get("confidence", 0.0) or 0.0),
        insufficient_evidence=bool(raw.get("insufficient_evidence", False)),
        evidence_doc_ids=raw.get("evidence_doc_ids", [])
    )
    
    logger.info(f"Reasoning output: confidence={ro.confidence:.2f}, insufficient_evidence={ro.insufficient_evidence}")
    logger.info(f"Generated {len(ro.recommended_actions)} recommended actions")
    if ro.root_cause:
        logger.info(f"Root cause identified: {ro.root_cause[:100]}{'...' if len(ro.root_cause) > 100 else ''}")

    # Compose display answer
    logger.info("Composing display answer")
    display = []
    if ro.root_cause:
        display.append(f"Root cause (hypothesis): {ro.root_cause}")
    display.append(ro.answer)
    if ro.recommended_actions:
        display.append("\nRecommended actions:")
        for i, a in enumerate(ro.recommended_actions, 1):
            p = a.get("priority", "P2")
            owner = a.get("owner", "SRE")
            why = a.get("why", "")
            rb = f" (runbook: {a['runbook']})" if a.get("runbook") else ""
            display.append(f"  {i}. [{p}] {a['action']} — {owner}. {why}{rb}")
            logger.info(f"Action {i}: {a.get('action', 'N/A')} (priority: {p}, owner: {owner})")
    display.append(f"\nConfidence: {ro.confidence:.2f} | Insufficient evidence: {ro.insufficient_evidence}")

    # Enrich diagnostics
    diag = {
        **(diag or {}),
        "reasoning_used": True,
        "evidence_doc_ids": ro.evidence_doc_ids,
        "confidence": ro.confidence,
        "insufficient_evidence": ro.insufficient_evidence,
        "docs_counts": {
            "logs": len(log_docs),
            "incidents": len(incident_docs)
        }
    }
    
    joined_display = "\n".join(display)
    logger.info(f"Query completed successfully. Final answer length: {len(joined_display)} characters")
    logger.info(f"Diagnostics: {diag}")

    return joined_display, log_docs, incident_docs, diag
