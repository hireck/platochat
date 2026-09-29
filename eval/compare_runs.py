#!/usr/bin/env python
"""Put several full eval runs side by side, grouped by the models they ran on.

    python eval/compare_runs.py eval/results/*-cmp-*-full

The runs of one configuration (router model, answer model, collection) are
pooled: a check that passed in 2 of 3 runs counts 2 of 3. Two runs of the same
configuration differ by a check or two, because the models sample, so give each
configuration several runs and read a difference of one or two checks as noise.

Reported per configuration: the checks of run_eval.py; where the time went (the
router call, the searches, the answering model); how often the answering model
called a tool; and what no check looks at -- answer length, requests that
failed, citations naming no passage the model was given, reasoning or citation
tags left in the reply, and CJK script in an English answer. Then every
question on which the configurations differ.

Runs made before run_eval.py recorded timings and tool calls (2026-09-28) show
"–" there.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter

CHECKS = ["router_search", "router_info", "retrieved_expected", "cited_expected",
          "answer_pattern", "general_knowledge_flag"]

LEFTOVER_TAG_RE = re.compile(r"</?(?:\w+:)?(?:think|cite)\b", re.IGNORECASE)
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def load(path: str) -> dict:
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    with open(path, encoding="utf-8") as fh:
        run = json.load(fh)
    if run.get("mode") != "full":
        sys.exit(f"{path} is a retrieval run; only full runs have answers to compare.")
    run["name"] = os.path.basename(os.path.dirname(path))
    return run


def config_of(run: dict) -> str:
    return f"{run['router_model']} -> {run['answer_model']} on {run['collection']}"


def median(values: list[float]) -> str:
    return f"{statistics.median(values):.1f}" if values else "–"


def p90(values: list[float]) -> str:
    if not values:
        return "–"
    ordered = sorted(values)
    return f"{ordered[round(0.9 * (len(ordered) - 1))]:.1f}"


def count(rows: list[dict], test) -> str:
    return f"{sum(1 for r in rows if test(r))}/{len(rows)}"


def summary(rows: list[dict]) -> dict[str, str]:
    """One column of the table: every row of every run of one configuration."""
    out: dict[str, str] = {}
    for name in CHECKS:
        flags = [r["checks"][name] for r in rows if name in r["checks"]]
        out[name] = f"{sum(flags)}/{len(flags)}" if flags else "–"

    timed = [r for r in rows if "seconds_in" in r]
    out["seconds, median"] = median([r["seconds"] for r in rows])
    out["seconds, 90th percentile"] = p90([r["seconds"] for r in rows])
    for part in ("route", "search", "answer"):
        out[f"  in {part}, median"] = median([r["seconds_in"].get(part, 0.0) for r in timed])
    out["  in answer, 90th percentile"] = p90([r["seconds_in"].get("answer", 0.0) for r in timed])

    if timed:
        rounds = [max((c["round"] for c in r["tool_calls"]), default=0) for r in timed]
        names = Counter(c["name"] for r in timed for c in r["tool_calls"])
        out["answers calling a tool"] = count(timed, lambda r: r["tool_calls"])
        out["tool rounds per answer"] = f"{statistics.mean(rounds):.2f}"
        out["  search_publications calls"] = str(names["search_publications"])
        out["  find_papers calls"] = str(names["find_papers"])
        out["  search_esa_website calls"] = str(names["search_esa_website"])
    else:
        for key in ("answers calling a tool", "tool rounds per answer",
                    "  search_publications calls", "  find_papers calls",
                    "  search_esa_website calls"):
            out[key] = "–"

    out["words per answer, median"] = median([len(r["reply"].split()) for r in rows])
    out["general-knowledge label"] = count(rows, lambda r: r["flagged_general_knowledge"])
    out["failed requests"] = count(rows, lambda r: r.get("error"))
    out["empty replies"] = count(rows, lambda r: not r["reply"].strip())
    out["citing a passage not given"] = (count(timed, lambda r: r["bogus_citations"])
                                         if timed else "–")
    out["think/cite tags left in reply"] = count(rows, lambda r: LEFTOVER_TAG_RE.search(r["reply"]))
    out["CJK characters in reply"] = count(rows, lambda r: CJK_RE.search(r["reply"]))
    return out


def per_question(groups: dict[str, list[dict]]) -> list[tuple[str, str, list[str]]]:
    """(question, check, "passes/runs" per configuration) wherever they differ.

    Only checks that every configuration ran on the question are compared.
    """
    diffs = []
    ids = list(dict.fromkeys(item["id"] for runs in groups.values()
                             for run in runs for item in run["items"]))
    for qid in ids:
        for name in CHECKS:
            cells, rates = [], []
            for runs in groups.values():
                flags = [item["checks"][name] for run in runs for item in run["items"]
                         if item["id"] == qid and name in item["checks"]]
                cells.append(f"{sum(flags)}/{len(flags)}" if flags else "–")
                rates.append(sum(flags) / len(flags) if flags else None)
            if None not in rates and len(set(rates)) > 1:
                diffs.append((qid, name, cells))
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="results folders (or their results.json)")
    args = ap.parse_args()

    groups: dict[str, list[dict]] = {}
    for path in args.runs:
        run = load(path)
        groups.setdefault(config_of(run), []).append(run)

    letters = [chr(ord("A") + i) for i in range(len(groups))]
    for letter, (config, runs) in zip(letters, groups.items()):
        print(f"{letter}: {config}")
        for run in runs:
            print(f"     {run['name']}  ({len(run['items'])} questions)")

    columns = [summary([item for run in runs for item in run["items"]])
               for runs in groups.values()]
    width = max(len(key) for key in columns[0]) + 2
    print("\n" + " " * width + "".join(f"{letter:>14s}" for letter in letters))
    for key in columns[0]:
        print(f"{key:{width}s}" + "".join(f"{col[key]:>14s}" for col in columns))

    if len(groups) > 1:
        diffs = per_question(groups)
        print(f"\nQuestions on which the configurations differ (passes/runs):")
        for qid, name, cells in diffs:
            print(f"  {qid:26s} {name:24s}" + "".join(f"{c:>8s}" for c in cells))
        if not diffs:
            print("  none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
