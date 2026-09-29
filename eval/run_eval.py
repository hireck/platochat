#!/usr/bin/env python
"""Run the evaluation questions against the chatbot and score the results.

Two modes:

    python eval/run_eval.py            # retrieval only: fast, no LLM involved
    python eval/run_eval.py --full     # end to end: router, search, answer

**Retrieval mode** sends each question straight to the search and asks one
thing: does a paper that contains the answer come back? It reports this at two
points -- in the candidate pool (what vector + BM25 search found) and in the
final passages (what survived the reranker) -- because those are different
failures. A paper missing from the pool is a search problem (chunking,
embedding, BM25 fields); a paper in the pool but not in the final passages is a
reranker problem (threshold, top-k).

A question with ``paper_search`` (find_papers arguments) is about the papers
themselves -- who wrote what, what came out when -- and is scored in retrieval
mode through the paper index instead; see run_paper_search().

**Full mode** runs the real pipeline and checks, per question: did the router
make the expected decision, did the model cite an expected paper, is the
general-knowledge label present exactly where it should be, and does the
answer contain the expected fact (a regular expression, e.g. ``\\b26\\b``
cameras; ``answer_must_not_match`` lists what must *not* appear). The patterns
are deliberately crude: they catch a wrong or missing number, not a badly
worded answer, so the answers are also written to a markdown report for
reading.

Results go to eval/results/<timestamp>-<label>/. Compare two runs with
``--against eval/results/<earlier>/results.json``.

The questions live in eval/questions.json; ``expected_papers`` are bibcodes as
in ingest/papers.json, and older bibcodes of the same paper match too. A page
of ESA's PLATO website counts as a source too, named ``esa:<page>`` after its
address (``esa:ao-1`` for https://www.cosmos.esa.int/web/plato/ao-1). A
question whose expected sources are all such pages is scored in retrieval mode
through the website index (search_esa_website) instead of the papers'.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "platochat"))


def load_aliases() -> dict[str, str]:
    """Any bibcode a paper has ever had -> its current one."""
    path = os.path.join(REPO, "ingest", "papers.json")
    with open(path, encoding="utf-8") as fh:
        papers = json.load(fh)["papers"]
    return {alias: bib for bib, rec in papers.items() for alias in rec["aliases"]}


def source_of(doc, aliases: dict[str, str]) -> str:
    """The current bibcode of a doc's paper, or ``esa:<page>`` for a website passage."""
    props = doc.properties
    if props.get("page"):
        return "esa:" + props["page"]
    # A passage names its paper in parent_doc, a find_papers record in bibcode.
    bib = props.get("parent_doc") or props.get("bibcode") or ""
    return aliases.get(bib, bib)


def is_site_question(item: dict) -> bool:
    expected = item.get("expected_papers") or []
    return bool(expected) and all(e.startswith("esa:") for e in expected)


def papers_of(docs: list, aliases: dict[str, str]) -> list[str]:
    """Current bibcode of each doc's paper (or esa:<page>), in order, without repeats."""
    out: list[str] = []
    for d in docs:
        bib = source_of(d, aliases)
        if bib not in out:
            out.append(bib)
    return out


def first_rank(found: list[str], expected: list[str]) -> int | None:
    """1-based position of the first expected paper in ``found``."""
    for i, bib in enumerate(found, start=1):
        if bib in expected:
            return i
    return None


def run_paper_search(core, item: dict, aliases: dict) -> dict:
    """Retrieval mode for a question about papers rather than their content.

    Scored through the paper index with the find_papers arguments the question
    carries (``paper_search``) -- the passage search ignores authors and dates,
    so it has nothing to find there. Whether the model picks those arguments is
    for full mode to show. Here "in the pool" means *every* expected paper came
    back: a list of papers missing one is wrong, not merely lower-ranked.
    """
    t0 = time.time()
    found = core.find_papers(**item["paper_search"])
    expected = item["expected_papers"]
    top = papers_of(found.papers, aliases)
    return {
        "pool_size": found.total,
        "in_pool": all(bib in top for bib in expected),
        "found_via": ["papers"],
        "final_rank": first_rank(top, expected),
        "final_papers": top,
        "seconds": round(time.time() - t0, 2),
    }


def run_retrieval(core, item: dict, aliases: dict) -> dict:
    if "paper_search" in item:
        return run_paper_search(core, item, aliases)
    t0 = time.time()
    if is_site_question(item):
        candidates, found_by = core.fetch_candidates(item["question"], core.site_collection,
                                                     core.SITE_BM25_PROPERTIES)
    else:
        candidates, found_by = core.fetch_candidates(item["question"])
    final = core.rerank(item["question"], candidates, found_by)
    expected = item["expected_papers"]

    # Which retriever surfaced the expected paper -- how we tell whether BM25
    # is finding things vector search misses, or only duplicating it.
    via: set[str] = set()
    for d in candidates:
        if source_of(d, aliases) in expected:
            via.update(found_by[str(d.uuid)])

    pool, top = papers_of(candidates, aliases), papers_of(final, aliases)
    return {
        "pool_size": len(candidates),
        "in_pool": first_rank(pool, expected) is not None,
        "found_via": sorted(via),
        "final_rank": first_rank(top, expected),
        "final_papers": top,
        "seconds": round(time.time() - t0, 2),
    }


# The patterns are written with ASCII hyphens and spaces, and a model may use
# look-alikes: GLM-5.3-Flash joins "noise-to-signal" with non-breaking hyphens
# (U+2011), which failed a correct answer. Soft hyphens (U+00AD), which it also
# puts inside words, are invisible on the page and go.
PLAIN_TEXT = str.maketrans({"\u2010": "-", "\u2011": "-", "\u00a0": " ", "\u202f": " ",
                            "\u00ad": ""})


def run_full(core, item: dict, aliases: dict) -> dict:
    t0 = time.time()
    try:
        ans, error = core.answer_question_detailed(item["question"], item.get("history") or []), None
    except Exception as exc:
        # One failed request (a timeout, a 5xx from the endpoint) must not cost
        # the rest of the run; it counts as failing every check that applies.
        print(f"  ERROR {type(exc).__name__}: {exc}")
        ans, error = core.Answer(), f"{type(exc).__name__}: {exc}"
    seconds = round(time.time() - t0, 1)
    reply = ans.reply or ""
    decision = ans.decision or core.RouterDecision(platochat_info=False)
    expected = item["expected_papers"]

    checks: dict[str, bool] = {}
    if "expect_search" in item:
        checks["router_search"] = bool(decision.search_query) == item["expect_search"]
    if "expect_info" in item:
        checks["router_info"] = decision.platochat_info == item["expect_info"]
    if expected:
        checks["retrieved_expected"] = first_rank(papers_of(ans.docs, aliases), expected) is not None
        checks["cited_expected"] = first_rank(papers_of(ans.cited, aliases), expected) is not None
    plain = reply.translate(PLAIN_TEXT)
    failed_patterns = [p for p in item.get("answer_must_match", []) if not re.search(p, plain)]
    failed_patterns += ["NOT " + p for p in item.get("answer_must_not_match", [])
                        if re.search(p, plain)]
    if item.get("answer_must_match") or item.get("answer_must_not_match"):
        checks["answer_pattern"] = not failed_patterns
    # Anything not taken from the papers must carry the fixed label -- and an
    # answer the papers fully support must not.
    flagged = core.GENERAL_KNOWLEDGE_LABEL.casefold() in reply.casefold()
    if "expect_flag" in item:
        checks["general_knowledge_flag"] = flagged == item["expect_flag"]
    if error:
        checks = {name: False for name in checks}

    # used_sources() numbers every cited passage [1], [2], ... in order of first
    # use, bogus ones too, so the unbroken run of marks from [1] counts the
    # distinct citations; those that name no passage the model was given are
    # missing from ans.cited.
    marks = {int(n) for n in re.findall(r"\[(\d+)\]", reply)}
    n_marks = 0
    while n_marks + 1 in marks:
        n_marks += 1
    return {
        "decision": ans.decision.model_dump() if ans.decision else None,
        "n_passages": len(ans.docs),
        "retrieved_papers": papers_of(ans.docs, aliases),
        "cited_papers": papers_of(ans.cited, aliases),
        "bogus_citations": n_marks - len(ans.cited) if ans.docs else 0,
        "tool_calls": ans.tool_calls,
        "checks": checks,
        "flagged_general_knowledge": flagged,
        "failed_patterns": failed_patterns,
        "reply": reply,
        "sources": ans.sources,
        "error": error,
        "seconds": seconds,
        "seconds_in": ans.seconds,
    }


def rate(flags: list[bool]) -> str:
    return f"{sum(flags)}/{len(flags)}" if flags else "–"


def summarise_retrieval(rows: list[dict]) -> None:
    print(f"\nRetrieval, {len(rows)} questions")
    print(f"  expected paper in the candidate pool : {rate([r['in_pool'] for r in rows])}")
    print(f"  expected paper in the final passages : {rate([r['final_rank'] is not None for r in rows])}")
    print(f"  ...and ranked first                  : {rate([r['final_rank'] == 1 for r in rows])}")
    only_bm25 = [r["id"] for r in rows if r["found_via"] == ["bm25"]]
    only_vec = [r["id"] for r in rows if r["found_via"] == ["vector"]]
    print(f"  found by BM25 only: {len(only_bm25)} {only_bm25}")
    print(f"  found by vector search only: {len(only_vec)} {only_vec}")
    papers = [r for r in rows if r["found_via"] == ["papers"]]
    if papers:
        print(f"  of these, scored through find_papers: {len(papers)}; every expected paper "
              f"listed in {rate([r['in_pool'] for r in papers])}")
    for r in rows:
        if r["final_rank"] is None:
            where = "lost in reranking" if r["in_pool"] else "never retrieved"
            print(f"  MISS {r['id']:24s} {where:18s} got: {', '.join(r['final_papers'][:4])}")


def summarise_full(rows: list[dict]) -> None:
    print(f"\nEnd to end, {len(rows)} questions")
    names = ["router_search", "router_info", "retrieved_expected", "cited_expected",
             "answer_pattern", "general_knowledge_flag"]
    for name in names:
        flags = [r["checks"][name] for r in rows if name in r["checks"]]
        print(f"  {name:24s}: {rate(flags)}")
    flagged = [r["id"] for r in rows if r["flagged_general_knowledge"]]
    print(f"  answers carrying the general-knowledge label: {len(flagged)} {flagged}")
    secs = sorted(r["seconds"] for r in rows)
    print(f"  seconds per answer  : median {secs[len(secs) // 2]}, max {secs[-1]}")
    for part in ("route", "search", "answer"):
        part_secs = sorted(r["seconds_in"].get(part, 0.0) for r in rows)
        print(f"    of which {part:6s}   : median {part_secs[len(part_secs) // 2]}, max {part_secs[-1]}")
    searched = [r["id"] for r in rows if r["tool_calls"]]
    print(f"  answers that called a tool: {len(searched)} {searched}")
    bogus = [r["id"] for r in rows if r["bogus_citations"]]
    if bogus:
        print(f"  answers citing a passage they were not given: {len(bogus)} {bogus}")
    for r in rows:
        bad = [k for k, ok in r["checks"].items() if not ok]
        if bad:
            extra = f"  missing: {r['failed_patterns']}" if r["failed_patterns"] else ""
            if r["error"]:
                extra = f"  error: {r['error']}"
            print(f"  FAIL {r['id']:24s} {', '.join(bad)}{extra}")


def write_report(path: str, rows: list[dict], header: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# Eval answers — {header['label']} ({header['timestamp']})\n\n")
        fh.write(f"Collection `{header['collection']}`, answer model `{header['answer_model']}`, "
                 f"router model `{header['router_model']}`.\n\n")
        for r in rows:
            marks = "  ".join(f"{'✅' if ok else '❌'} {k}" for k, ok in r["checks"].items())
            fh.write(f"## {r['id']} ({r['category']})\n\n**Q:** {r['question']}\n\n")
            fh.write(f"{marks or '(nothing checked automatically)'}  ·  {r['seconds']} s\n\n")
            if r.get("note"):
                fh.write(f"_Note: {r['note']}_\n\n")
            fh.write(f"Router: `{json.dumps(r['decision'])}`\n\n")
            for call in r["tool_calls"]:
                fh.write(f"Tool call, round {call['round']}: `{call['name']} {call['arguments']}`\n\n")
            if r["error"]:
                fh.write(f"**Error:** `{r['error']}`\n\n")
            fh.write(f"{r['reply']}\n\n")
            if r["sources"]:
                fh.write(r["sources"] + "\n\n")
            fh.write("---\n\n")


def compare(rows: list[dict], against_path: str, mode: str) -> None:
    with open(against_path, encoding="utf-8") as fh:
        before = {r["id"]: r for r in json.load(fh)["items"]}
    print(f"\nChanges against {against_path}:")
    changed = False
    for r in rows:
        b = before.get(r["id"])
        if b is None:
            continue
        if mode == "retrieval":
            old, new = b.get("final_rank"), r["final_rank"]
            if old != new:
                changed = True
                print(f"  {r['id']:24s} final rank {old} -> {new}")
        else:
            for k, ok in r["checks"].items():
                if k in b.get("checks", {}) and b["checks"][k] != ok:
                    changed = True
                    print(f"  {r['id']:24s} {k}: {'pass' if b['checks'][k] else 'fail'} -> {'pass' if ok else 'fail'}")
    if not changed:
        print("  none")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="end to end, with the LLM")
    ap.add_argument("--questions", default=os.path.join(HERE, "questions.json"))
    ap.add_argument("--only", default="", help="comma-separated question ids")
    ap.add_argument("--label", default="run", help="name for the results folder")
    ap.add_argument("--against", default="", help="an earlier results.json to compare with")
    args = ap.parse_args()

    with open(args.questions, encoding="utf-8") as fh:
        items = json.load(fh)
    if args.only:
        wanted = set(args.only.split(","))
        items = [it for it in items if it["id"] in wanted]
    mode = "full" if args.full else "retrieval"
    if mode == "retrieval":
        # Needs a standalone question and a paper to look for.
        items = [it for it in items if it["expected_papers"] and not it.get("history")]

    aliases = load_aliases()
    import plato_core as core  # slow: loads the models and connects to Weaviate

    rows = []
    try:
        for n, item in enumerate(items, start=1):
            print(f"[{n}/{len(items)}] {item['id']}", flush=True)
            run = run_full if args.full else run_retrieval
            rows.append({"id": item["id"], "category": item["category"],
                         "question": item["question"], "note": item.get("note"),
                         **run(core, item, aliases)})
    finally:
        core.langfuse_client.flush()
        core.weaviate_client.close()

    (summarise_full if args.full else summarise_retrieval)(rows)
    if args.against:
        compare(rows, args.against, mode)

    header = {
        "label": args.label, "mode": mode,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "collection": core.WEAVIATE_COLLECTION,
        "answer_model": core.ANSWER_MODEL, "router_model": core.ROUTER_MODEL,
        "embed_model": core.EMBED_MODEL, "rerank_model": core.RERANK_MODEL,
    }
    out_dir = os.path.join(HERE, "results",
                           datetime.now().strftime("%Y%m%d-%H%M") + f"-{args.label}-{mode}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({**header, "items": rows}, fh, indent=1, ensure_ascii=False)
    if args.full:
        write_report(os.path.join(out_dir, "answers.md"), rows, header)
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
