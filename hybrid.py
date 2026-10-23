"""Hybrid retrieval: dense (embedding) search + BM25 keyword search,
merged with Reciprocal Rank Fusion (RRF).

Works on any Chroma collection, so it serves both the terminal tools and the
per-session collections of the web app.
"""
import re

from rank_bm25 import BM25Okapi

from config import QUERY_PREFIX, TOP_K, get_embedder

RRF_K = 60     # standard RRF constant
POOL = 20      # candidates taken from each retriever before fusion

_cache = {}    # (collection name, size) -> BM25 index + chunk lookup


def _tokens(text: str):
    return re.findall(r"\w+", text.lower())


def _index_for(col):
    key = (col.name, col.count())
    if key not in _cache:
        if len(_cache) > 50:
            _cache.clear()
        data = col.get(include=["documents", "metadatas"])
        _cache[key] = {
            "ids": data["ids"],
            "docs": data["documents"],
            "metas": data["metadatas"],
            "pos": {cid: i for i, cid in enumerate(data["ids"])},
            "bm25": BM25Okapi([_tokens(d) for d in data["documents"]]),
        }
    return _cache[key]


def dense_search(col, query: str, n: int = POOL):
    """Returns [(chunk_id, cosine_similarity), ...] best first."""
    n = min(n, col.count())
    q_emb = get_embedder().encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=n)
    return [(cid, 1 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]


def bm25_search(col, query: str, n: int = POOL):
    """Returns [(chunk_id, bm25_score), ...] best first, only positive scores."""
    idx = _index_for(col)
    scores = idx["bm25"].get_scores(_tokens(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [(idx["ids"][i], float(scores[i])) for i in order if scores[i] > 0]


def rrf(*ranked_lists):
    """Reciprocal Rank Fusion over lists of (id, score). Returns {id: fused_score}."""
    fused = {}
    for ranked in ranked_lists:
        for rank, (cid, _) in enumerate(ranked, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return fused


def hybrid_retrieve(col, query: str, k: int = TOP_K):
    """Returns (hits, best_dense_similarity).

    best_dense_similarity is the confidence signal used by the guardrail:
    if even the closest chunk by meaning is weak, the question is probably
    not covered by the documents.
    """
    if col.count() == 0:
        return [], 0.0

    dense = dense_search(col, query)
    sparse = bm25_search(col, query)
    fused = rrf(dense, sparse)
    dense_sim = dict(dense)
    idx = _index_for(col)

    hits = []
    for cid, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]:
        i = idx["pos"][cid]
        meta = idx["metas"][i]
        # Support both 'source' (ingest.py) and 'filename' (app.py) metadata keys
        source = meta.get("source") or meta.get("filename", "Unknown")
        hits.append({
            "id": cid,
            "text": idx["docs"][i],
            "source": source,
            "page": meta.get("page", "?"),
            "score": dense_sim.get(cid, 0.0),
        })
    best = dense[0][1] if dense else 0.0
    return hits, best
