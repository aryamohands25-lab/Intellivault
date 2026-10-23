"""Evaluation for IntelliVault: Baseline vs Hybrid vs Agentic on a fixed question set.

Run from the Intellivault folder, after `py -3.11 ingest.py` has indexed your PDFs:

    py -3.11 evaluate.py --limit 5            quick trial on the first 5 questions
    py -3.11 evaluate.py                      everything (slow on a local model)
    py -3.11 evaluate.py --modes baseline hybrid
    py -3.11 evaluate.py --summary-only       rebuild the tables from saved results

Progress is saved to results.jsonl after every answer. You can stop with Ctrl+C
and rerun later: finished (question, mode) pairs are skipped, failed ones are retried.

How answers are scored
  * Answerable questions: correct if the answer is not a refusal AND contains every
    expected fact listed in questions.json ("must"; each inner list = acceptable spellings).
  * Retrieval hit: every expected fact appears in the passages the system retrieved.
    "hit but wrong" = a generation/reasoning error; "no hit and wrong" = a retrieval miss.
  * Unanswerable questions: correct only if the system refuses ("I could not find ...").
Keyword matching is a rough proxy, so read a few answers by hand before quoting numbers.
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

from agent import NOT_FOUND, run_agent
from baseline_rag import build_context, call_llm
from config import (GEMINI_MODEL, LLM_MODEL, LLM_PROVIDER, MIN_SIMILARITY, OLLAMA_MODEL,
                    TOP_K, get_collection)
from hybrid import _index_for, dense_search, hybrid_retrieve

HERE = Path(__file__).parent
QUESTIONS_FILE = HERE / "questions.json"
RESULTS_FILE = HERE / "results.jsonl"
SUMMARY_FILE = HERE / "results_summary.md"
CSV_FILE = HERE / "results.csv"
MODES = ["baseline", "hybrid", "agentic"]
LABELS = {"baseline": "Baseline", "hybrid": "Hybrid", "agentic": "Agentic"}


# ----------------------------------------------------------------- text matching
def squash(s: str) -> str:
    """Lowercase and drop spaces/hyphens so 'RAG-Sequence', 'RAG Sequence' and line-broken text all match."""
    s = (s or "").lower().replace("\ufb01", "fi").replace("\ufb02", "fl")
    return re.sub(r"[\s\-\u2010-\u2015]+", "", s)


def has(text: str, phrase: str) -> bool:
    return squash(phrase) in squash(text)


def covers(text: str, groups) -> bool:
    return all(any(has(text, alt) for alt in group) for group in groups)


# ----------------------------------------------------------------- the three pipelines
def run_mode(col, mode: str, question: str):
    """Returns (answer, evidence_hits, trace)."""
    if mode == "agentic":
        return run_agent(col, question)

    if mode == "hybrid":
        hits, best = hybrid_retrieve(col, question)
    else:  # baseline: dense retrieval only, single step
        idx = _index_for(col)
        hits = []
        for cid, sim in dense_search(col, question, n=TOP_K):
            i = idx["pos"][cid]
            hits.append({"id": cid, "text": idx["docs"][i], "source": idx["metas"][i]["source"],
                         "page": idx["metas"][i]["page"], "score": sim})
        best = hits[0]["score"] if hits else 0.0

    if not hits or best < MIN_SIMILARITY:
        return NOT_FOUND, hits, []
    prompt = f"Context:\n{build_context(hits)}\n\nQuestion: {question}"
    return call_llm(prompt), hits, []


def score(q: dict, answer: str, hits: list) -> dict:
    refused = has(answer, "could not find")
    if q["type"] == "unanswerable":
        return {"correct": refused, "retrieval_hit": None, "refused": refused}
    passages = " ".join(h["text"] for h in hits)
    return {
        "correct": (not refused) and covers(answer, q["must"]),
        "retrieval_hit": covers(passages, q["must"]),
        "refused": refused,
    }


# ----------------------------------------------------------------- saved results
def load_results() -> dict:
    latest = {}
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                latest[(r["id"], r["mode"])] = r   # later lines replace earlier ones
    return latest


def model_name() -> str:
    return {"ollama": OLLAMA_MODEL, "gemini": GEMINI_MODEL}.get(LLM_PROVIDER, LLM_MODEL)


# ----------------------------------------------------------------- running
def run_evaluation(args):
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    col = get_collection()
    if col.count() == 0:
        sys.exit("The index is empty. Run:  py -3.11 ingest.py  first.")
    sources = sorted({m["source"] for m in col.get(include=["metadatas"])["metadatas"]})
    print(f"LLM: {LLM_PROVIDER} / {model_name()}   similarity cutoff: {MIN_SIMILARITY}   top-k: {TOP_K}")
    print(f"Index: {col.count()} chunks from {', '.join(sources)}")

    done = {k for k, r in load_results().items() if not r.get("error")}
    todo = [(q, m) for q in questions for m in args.modes if (q["id"], m) not in done]
    print(f"{len(todo)} runs to do, {len(questions) * len(args.modes) - len(todo)} already done.\n")

    try:
        with RESULTS_FILE.open("a", encoding="utf-8") as f:
            for n, (q, mode) in enumerate(todo, start=1):
                t0 = time.time()
                try:
                    answer, hits, trace = run_mode(col, mode, q["question"])
                    err = None
                except Exception as e:  # network, model not found, etc.
                    answer, hits, trace, err = "", [], [], f"{type(e).__name__}: {e}"
                secs = round(time.time() - t0, 1)

                res = {"correct": False, "retrieval_hit": None, "refused": False} if err else score(q, answer, hits)
                rec = {
                    "id": q["id"], "mode": mode, "type": q["type"], "question": q["question"],
                    "answer": answer, "seconds": secs, "error": err, **res,
                    "sources": [f"{h['source']} p.{h['page']}" for h in hits],
                    "trace": trace,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

                status = "ERROR" if err else ("correct" if res["correct"] else "wrong")
                print(f"[{n}/{len(todo)}] {q['id']} {mode:<8} {status:<8} {secs:>6.1f}s", flush=True)
                if err:
                    print(f"      {err[:200]}")
    except KeyboardInterrupt:
        print("\nStopped. Progress is saved; rerun the same command to continue.")

    summarize(json.loads(QUESTIONS_FILE.read_text(encoding="utf-8")))


# ----------------------------------------------------------------- summary tables
def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def summarize(questions: list):
    latest = load_results()
    qmap = {q["id"]: q for q in questions}
    order = list(qmap)
    present = [m for m in MODES if any(k[1] == m and not r.get("error") for k, r in latest.items())]
    if not present:
        print("No finished results yet.")
        return

    ok = {m: {i: r for (i, mm), r in latest.items() if mm == m and not r.get("error") and i in qmap}
          for m in present}
    common = sorted(set.intersection(*[set(ok[m]) for m in present]), key=order.index)
    if not common:
        print("No question has been finished in every mode yet.")
        return
    ans = [i for i in common if qmap[i]["type"] != "unanswerable"]
    una = [i for i in common if qmap[i]["type"] == "unanswerable"]

    def column(m):
        R = ok[m]
        return {
            "correct": pct(sum(R[i]["correct"] for i in ans), len(ans)),
            "hit": pct(sum(bool(R[i]["retrieval_hit"]) for i in ans), len(ans)),
            "gen": pct(sum(1 for i in ans if R[i]["retrieval_hit"] and not R[i]["correct"]), len(ans)),
            "miss": pct(sum(1 for i in ans if not R[i]["retrieval_hit"] and not R[i]["correct"]), len(ans)),
            "refuse": pct(sum(bool(R[i]["refused"]) for i in ans), len(ans)),
            "una": pct(sum(bool(R[i]["correct"]) for i in una), len(una)),
            "secs": f"{sum(R[i]['seconds'] for i in common) / len(common):.1f}",
        }

    cols = {m: column(m) for m in present}
    rows = [
        ("Answerable questions answered correctly", "correct"),
        ("Retrieval hit (needed facts were retrieved)", "hit"),
        ("Retrieved but answered wrong (generation errors)", "gen"),
        ("Not retrieved and wrong (retrieval misses)", "miss"),
        ("Answerable questions wrongly refused", "refuse"),
        ("Unanswerable questions correctly refused", "una"),
        ("Average seconds per question", "secs"),
    ]

    md = ["# IntelliVault evaluation results", "",
          f"LLM: {LLM_PROVIDER} / {model_name()} | similarity cutoff: {MIN_SIMILARITY} | top-k: {TOP_K}",
          f"Questions compared: {len(common)} ({len(ans)} answerable, {len(una)} unanswerable)", "",
          "| Metric | " + " | ".join(LABELS[m] for m in present) + " |",
          "|---|" + "---|" * len(present)]
    for label, key in rows:
        md.append(f"| {label} | " + " | ".join(cols[m][key] for m in present) + " |")

    md += ["", "## Per-question results (OK = correct, -- = wrong, ERR = error, blank = not run)", "",
           "| ID | Type | " + " | ".join(LABELS[m] for m in present) + " |",
           "|---|---|" + "---|" * len(present)]
    for i in order:
        cells = []
        for m in present:
            r = latest.get((i, m))
            cells.append("" if r is None else "ERR" if r.get("error") else "OK" if r["correct"] else "--")
        if any(cells):
            md.append(f"| {i} | {qmap[i]['type']} | " + " | ".join(cells) + " |")

    text = "\n".join(md)
    print("\n" + text)
    SUMMARY_FILE.write_text(text + "\n", encoding="utf-8")

    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "mode", "type", "question", "correct", "retrieval_hit", "refused", "seconds", "sources", "answer"])
        for (i, m), r in sorted(latest.items(), key=lambda kv: (order.index(kv[0][0]) if kv[0][0] in order else 999, kv[0][1])):
            w.writerow([i, m, r["type"], r["question"], r["correct"], r["retrieval_hit"], r["refused"],
                        r["seconds"], "; ".join(r["sources"]), r["answer"].replace("\n", " ")])
    print(f"\nSaved: {SUMMARY_FILE.name}, {CSV_FILE.name}, {RESULTS_FILE.name}")


# ----------------------------------------------------------------- entry point
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Evaluate Baseline vs Hybrid vs Agentic RAG.")
    p.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    p.add_argument("--limit", type=int, default=0, help="only the first N questions")
    p.add_argument("--summary-only", action="store_true", help="rebuild tables from saved results")
    args = p.parse_args()

    if args.summary_only:
        summarize(json.loads(QUESTIONS_FILE.read_text(encoding="utf-8")))
    else:
        run_evaluation(args)
