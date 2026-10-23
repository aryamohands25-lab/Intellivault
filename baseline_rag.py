"""Phase 2: baseline single-step RAG (dense retrieval -> LLM).

This is the system your agentic version will be compared against,
so keep it simple and don't "improve" it later.

Usage:
    python baseline_rag.py "What is Reciprocal Rank Fusion?"
    python baseline_rag.py            # interactive mode
"""
import sys

import os

from config import (GEMINI_MODEL, LLM_MODEL, LLM_PROVIDER, MIN_SIMILARITY, OLLAMA_MODEL,
                    OLLAMA_URL, QUERY_PREFIX, TOP_K, get_collection, get_embedder)

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Write a short answer in your own words (2-4 sentences) and cite passages inline like [1], [2]. "
    "Do not copy passage headings or repeat the context. If the passages do not contain the answer, "
    "reply exactly: 'I could not find this in your documents.' Do not use outside knowledge."
)


def call_llm(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """One place to swap LLM providers. Retrieval code never changes."""
    if LLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=800,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    if LLM_PROVIDER == "ollama":
        import requests
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    if LLM_PROVIDER == "groq":
        from groq import Groq
        groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
        )
        return resp.choices[0].message.content

    # Default: Gemini
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system),
    )
    return resp.text


def retrieve(query: str, k: int = TOP_K):
    """Dense retrieval. Returns list of dicts with text, source, page, score."""
    q_emb = get_embedder().encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()
    res = get_collection().query(query_embeddings=q_emb, n_results=k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({
            "text": doc,
            "source": meta.get("source") or meta.get("filename", "Unknown"),
            "page": meta.get("page", "?"),
            "score": 1 - dist,  # cosine distance -> similarity
        })
    return hits


def build_context(hits) -> str:
    return "\n\n".join(
        f"[{i}] (source: {h.get('source') or h.get('filename', 'Unknown')}, page {h.get('page', '?')})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )


def answer(query: str) -> dict:
    hits = retrieve(query)

    # Guardrail: if even the best match is weak, don't bother the LLM
    if not hits or hits[0]["score"] < MIN_SIMILARITY:
        return {"answer": "I could not find this in your documents.", "sources": hits}

    prompt = f"Context:\n{build_context(hits)}\n\nQuestion: {query}"
    return {"answer": call_llm(prompt), "sources": hits}


def show(result: dict):
    print("\n" + result["answer"])
    print("\nSources:")
    for i, h in enumerate(result["sources"], start=1):
        print(f"  [{i}] {h['source']} p.{h['page']}  (similarity {h['score']:.2f})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(answer(" ".join(sys.argv[1:])))
    else:
        print("IntelliVault baseline RAG. Type 'exit' to quit.")
        while True:
            q = input("\nAsk: ").strip()
            if q.lower() in {"exit", "quit", ""}:
                break
            show(answer(q))
