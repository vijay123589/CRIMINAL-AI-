# ─────────────────────────────────────────────
#  rag_engine.py – ChromaDB Vector Store & Semantic Case Search
# ─────────────────────────────────────────────
from __future__ import annotations
import logging
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import sqlite3
from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
    RAG_CHUNKS,
    DB_PATH,
)

logger = logging.getLogger(__name__)

# Singletons
_client: chromadb.PersistentClient | None = None
_schema_col: chromadb.Collection | None = None
_case_col: chromadb.Collection | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _get_schema_collection() -> chromadb.Collection:
    global _schema_col
    if _schema_col is None:
        client = _get_client()
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _schema_col = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _schema_col


def _get_case_collection() -> chromadb.Collection:
    global _case_col
    if _case_col is None:
        client = _get_client()
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _case_col = client.get_or_create_collection(
            name="case_descriptions_vectors",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _case_col


def build_vector_store(force: bool = False) -> bool:
    """Populate vector stores with schema knowledge chunks and case narratives."""
    try:
        scol = _get_schema_collection()
        ccol = _get_case_collection()

        # Build schema knowledge store
        if force or scol.count() < len(RAG_CHUNKS):
            if scol.count() > 0:
                scol.delete(ids=scol.get()["ids"])
            ids       = [c["id"] for c in RAG_CHUNKS]
            documents = [c["text"] for c in RAG_CHUNKS]
            metadatas = [c["metadata"] for c in RAG_CHUNKS]
            scol.add(ids=ids, documents=documents, metadatas=metadatas)

        # Index actual database case descriptions & evidence into case collection
        if Path(DB_PATH).exists():
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            
            # Fetch cases
            cursor.execute("SELECT case_id, crime_type, district, description, severity FROM crime_incidents")
            cases = cursor.fetchall()
            
            case_ids = []
            case_docs = []
            case_meta = []
            for row in cases:
                cid, ctype, dist, desc, sev = row
                case_ids.append(f"case_{cid}")
                case_docs.append(f"Case ID: {cid} | Type: {ctype} | District: {dist} | Severity: {sev} | Modus Operandi: {desc}")
                case_meta.append({"case_id": cid, "crime_type": ctype, "district": dist, "severity": sev})

            # Fetch evidence
            cursor.execute("SELECT evidence_id, case_id, evidence_type, description FROM evidence_records")
            evidences = cursor.fetchall()
            for row in evidences:
                eid, cid, etype, edesc = row
                case_ids.append(f"evd_{eid}")
                case_docs.append(f"Evidence ID: {eid} for Case: {cid} | Type: {etype} | Details: {edesc}")
                case_meta.append({"case_id": cid, "evidence_id": eid, "evidence_type": etype})

            conn.close()

            if case_ids:
                if ccol.count() > 0:
                    ccol.delete(ids=ccol.get()["ids"])
                ccol.add(ids=case_ids, documents=case_docs, metadatas=case_meta)

        logger.info("✅ ChromaDB vector stores populated successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ Vector store build failed: {e}")
        return False


def retrieve_context(query: str, top_k: int = TOP_K_RESULTS) -> str:
    """Retrieve schema context chunks for Text-to-SQL prompt injection."""
    try:
        col = _get_schema_collection()
        if col.count() == 0:
            build_vector_store()
        results = col.query(
            query_texts=[query],
            n_results=min(top_k, col.count()),
            include=["documents"],
        )
        docs = results["documents"][0]
        return "\n\n---\n\n".join(docs)
    except Exception as e:
        logger.error(f"❌ RAG retrieval failed: {e}")
        return ""


def search_semantic_case_files(query: str, top_k: int = 5) -> list[dict]:
    """
    Perform semantic embedding similarity search over case descriptions & evidence records.
    Returns list of matching case result dicts.
    """
    try:
        ccol = _get_case_collection()
        if ccol.count() == 0:
            build_vector_store(force=True)

        results = ccol.query(
            query_texts=[query],
            n_results=min(top_k, ccol.count()),
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            for doc, meta, dist in zip(docs, metas, dists):
                similarity_score = round(max(0.0, (1.0 - float(dist))) * 100, 1)
                matches.append({
                    "case_id": meta.get("case_id", "N/A"),
                    "document": doc,
                    "metadata": meta,
                    "similarity_score": f"{similarity_score}%",
                    "raw_distance": round(float(dist), 4),
                })
        return matches
    except Exception as e:
        logger.error(f"❌ Semantic case search failed: {e}")
        return []


def get_store_stats() -> dict:
    """Return basic stats about vector stores."""
    try:
        scol = _get_schema_collection()
        ccol = _get_case_collection()
        return {
            "collection": CHROMA_COLLECTION,
            "schema_chunks": scol.count(),
            "case_vectors": ccol.count(),
            "embedding_model": EMBEDDING_MODEL,
        }
    except Exception as e:
        return {"error": str(e)}
