"""The agent layer of IntelliVault.

Unlike the single-step baseline (retrieve once, then answer), the agent:
  1. PLANS     - decides how to search: one query, several sub-queries, or a broad summary search
  2. RETRIEVES - runs hybrid (dense + BM25) retrieval for each sub-query
  3. CHECKS    - asks the LLM whether the evidence is enough to answer
  4. RE-QUERIES- if not, issues a better search query (at most MAX_ITERS times)
  5. ANSWERS   - only from the collected evidence, or says it could not find it

Every step is logged in a trace so you can show the agent's reasoning in the UI
and in your report.
"""
import json
import re

from baseline_rag import SYSTEM_PROMPT, build_context, call_llm
from config import MIN_SIMILARITY
from hybrid import hybrid_retrieve

MAX_ITERS = 2          # how many times the agent may re-query
PER_QUERY = 4          # chunks kept per sub-query
PER_QUERY_SUMMARY = 8  # chunks kept for broad summary searches
MAX_EVIDENCE = 12      # total chunks passed to the final answer

NOT_FOUND = "I could not find this in your documents."

PLAN_SYSTEM = (
    "You plan how to search a user's document collection. Reply with JSON only, no other text.\n"
    '{"strategy": "single" | "decompose" | "summarize", "sub_queries": ["..."]}\n'
    "- single: a simple factual question. sub_queries has ONE rewritten search query.\n"
    "- decompose: a multi-part or comparison question. Split it into 2-4 focused search queries.\n"
    "- summarize: asks for an overview or summary. sub_queries has 1-2 broad search queries."
)

CHECK_SYSTEM = (
    "You judge whether evidence passages are enough to answer a question. Reply with JSON only.\n"
    '{"sufficient": true | false, "new_query": "a better search query if not sufficient, else empty"}'
)


def _json(text: str, default: dict) -> dict:
    """Pull the first JSON object out of an LLM reply; fall back safely."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return default
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return default
def is_simple_question(question: str) -> bool:
    """Identify questions that usually need one retrieval pass."""
    q = question.lower().strip()

    complex_patterns = (
        "compare", "comparison", "difference between",
        "pros and cons", "advantages and disadvantages",
        "summarize", "summary", "overview",
        "all the", "explain both", "step by step",
        "relationship between", "based on multiple"
    )

    return not any(pattern in q for pattern in complex_patterns)

def run_agent(col, question: str):
    """Returns (answer_text, evidence_hits, trace)."""
    trace = []
    # Fast path: skip planning and evidence-check LLM calls
    # for ordinary factual questions.
    if is_simple_question(question):
        hits, best = hybrid_retrieve(col, question, k=4)

        if not hits or best < MIN_SIMILARITY:
            return NOT_FOUND, hits, [
                "Fast path: one retrieval pass; no extra LLM calls."
            ]

        prompt = (
            f"Context:\n{build_context(hits)}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely using only the supplied context. "
            "If the context does not contain the answer, say so."
        )

        answer = call_llm(prompt)
        trace.append(
            f"Fast path: retrieved {len(hits)} chunks; "
            "skipped planning and evidence-check calls."
        )
        return answer, hits, trace
    # 1. PLAN
    plan = _json(call_llm(f"Question: {question}", system=PLAN_SYSTEM), {})
    strategy = plan.get("strategy", "single")
    if strategy not in {"single", "decompose", "summarize"}:
        strategy = "single"
    subs = [q.strip() for q in plan.get("sub_queries", []) if isinstance(q, str) and q.strip()][:4]
    subs = subs or [question]
    trace.append(f"Plan: strategy = {strategy}; search queries = {subs}")

    per_query = PER_QUERY_SUMMARY if strategy == "summarize" else PER_QUERY
    evidence = {}          # chunk id -> hit (insertion order = priority)
    best = 0.0
    tried = set()

    def gather(queries):
        nonlocal best
        for q in queries:
            tried.add(q.lower())
            hits, b = hybrid_retrieve(col, q, k=per_query)
            best = max(best, b)
            new = 0
            for h in hits:
                if h["id"] not in evidence:
                    evidence[h["id"]] = h
                    new += 1
            trace.append(f"Retrieved for '{q}': {len(hits)} chunks ({new} new), best similarity {b:.2f}")

    # 2. RETRIEVE
    gather(subs)

    # Guardrail: nothing in the documents is even close to the question
    if best < MIN_SIMILARITY:
        trace.append(f"Best similarity {best:.2f} is below {MIN_SIMILARITY}; stopping without answering.")
        return NOT_FOUND, list(evidence.values())[:MAX_EVIDENCE], trace

    # 3 + 4. CHECK and RE-QUERY
    for i in range(1, MAX_ITERS + 1):
        passages = list(evidence.values())[:MAX_EVIDENCE]
        verdict = _json(
            call_llm(f"Question: {question}\n\nEvidence:\n{build_context(passages)}", system=CHECK_SYSTEM),
            {"sufficient": True},
        )
        if verdict.get("sufficient", True):
            trace.append(f"Check {i}: evidence judged sufficient.")
            break
        new_query = str(verdict.get("new_query", "")).strip()
        if not new_query or new_query.lower() in tried:
            trace.append(f"Check {i}: evidence incomplete but no new query to try.")
            break
        trace.append(f"Check {i}: evidence incomplete; re-querying with '{new_query}'.")
        gather([new_query])

    # 5. ANSWER (grounded in the collected evidence only)
    passages = list(evidence.values())[:MAX_EVIDENCE]
    prompt = f"Context:\n{build_context(passages)}\n\nQuestion: {question}"
    trace.append(f"Answering from {len(passages)} chunks.")
    return call_llm(prompt), passages, trace
