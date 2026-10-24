
"""
Phase 2: baseline single-step RAG (dense retrieval -> LLM).

Usage:
    python baseline_rag.py "What is Reciprocal Rank Fusion?"
    python baseline_rag.py            # interactive mode
"""

import os
import sys

from config import (
    GEMINI_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
    MIN_SIMILARITY,
    OLLAMA_MODEL,
    OLLAMA_URL,
    QUERY_PREFIX,
    TOP_K,
    get_collection,
    get_embedder,
)

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Write a short answer in your own words (2-4 sentences) and cite passages "
    "inline like [1], [2]. Do not copy passage headings or repeat the context. "
    "If the passages do not contain the answer, reply exactly: "
    "'I could not find this in your documents.' Do not use outside knowledge."
)


def call_llm(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Call the configured LLM provider without changing retrieval."""

    provider = str(LLM_PROVIDER or "gemini").strip().lower()

    if provider == "groq":
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Add it to your local .env "
                "or Streamlit Cloud Secrets."
            )

        groq_model = os.getenv(
            "GROQ_MODEL", "llama-3.3-70b-versatile"
        )

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0,
        )

        content = response.choices[0].message.content
        return content or "I could not generate an answer."

    elif provider == "gemini":
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing, but LLM_PROVIDER is set to gemini."
            )

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
            ),
        )

        return response.text or "I could not generate an answer."

    elif provider == "ollama":
        import requests

        response = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "options": {"temperature": 0},
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    elif provider == "anthropic":
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing.")

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=800,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider!r}. "
            "Choose groq, gemini, ollama, or anthropic."
        )


def retrieve(query: str, k: int = TOP_K):
    """Dense retrieval. Returns text, source, page, and similarity score."""

    query_embedding = get_embedder().encode(
        [QUERY_PREFIX + query],
        normalize_embeddings=True,
    ).tolist()

    results = get_collection().query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    hits = []

    if not results.get("documents") or not results["documents"][0]:
        return hits

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        metadata = metadata or {}

        hits.append(
            {
                "text": document,
                "source": metadata.get("source")
                or metadata.get("filename", "Unknown"),
                "page": metadata.get("page", "?"),
                "score": 1 - distance,
            }
        )

    return hits


def build_context(hits) -> str:
    """Format retrieved passages with numbered source references."""

    return "\n\n".join(
        f"[{index}] (source: {hit['source']}, page {hit['page']})\n"
        f"{hit['text']}"
        for index, hit in enumerate(hits, start=1)
    )


def answer(query: str) -> dict:
    """Retrieve relevant passages and answer using the configured LLM."""

    hits = retrieve(query)

    if not hits or hits[0]["score"] < MIN_SIMILARITY:
        return {
            "answer": "I could not find this in your documents.",
            "sources": hits,
        }

    prompt = (
        f"Context:\n{build_context(hits)}\n\n"
        f"Question: {query}"
    )

    return {
        "answer": call_llm(prompt),
        "sources": hits,
    }


def show(result: dict):
    """Print the answer and its retrieved sources."""

    print("\n" + result["answer"])
    print("\nSources:")

    for index, hit in enumerate(result["sources"], start=1):
        print(
            f"  [{index}] {hit['source']} p.{hit['page']} "
            f"(similarity {hit['score']:.2f})"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(answer(" ".join(sys.argv[1:])))
    else:
        print("IntelliVault baseline RAG. Type 'exit' to quit.")

        while True:
            query = input("\nAsk: ").strip()

            if query.lower() in {"exit", "quit", ""}:
                break

            try:
                show(answer(query))
            except Exception as error:
                print(f"\nError: {error}")
