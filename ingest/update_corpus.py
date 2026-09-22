#!/usr/bin/env python
"""Bring the PLATO corpus up to date: new papers in, changed ones redone, dropped ones out.

One command for the whole ingest, safe to run every night (see cron_update.sh):

1. metadata  refresh papers.json from the PLATO-Pub list and ADS
             (fetch_metadata.py)
2. fetch     download and convert the full text of every paper that may have
             one and does not yet (fetch_fulltext.py; LaTeXML for LaTeX,
             marker for PDFs)
3. index     bring the Weaviate collection in line with the list: write the
             objects of papers that are new or changed, delete those of papers
             that left it (reindex_weaviate.py)

Only what changed is done, and the two halves keep track differently:

* What the fetch step did for each paper is recorded in the manifest,
  <data>/manifest.json: which source the full text came from, where its
  markdown is, what failed and why. So a rerun downloads and converts only
  papers that are new, or whose failure might have gone away -- ADS reporting a
  new arXiv version or open-access copy, a PDF saved by hand into
  <data>/manual/, or --retry-failed.
* The index step asks Weaviate itself. Every object carries a fingerprint of
  what it was made from (see reindex_weaviate.fingerprint), and a paper is
  rewritten only when that changes. So the collection can be rebuilt, or lost
  along with its container, and the next run puts back exactly what is
  missing without downloading anything.

Which papers may have full text is decided in fetch_fulltext.py: arXiv and open
access yes, paywalled no. Every paper without full text is indexed by its
abstract, so that all of them can be found.

    python update_corpus.py                   # everything, into PLATO
    python update_corpus.py --dry-run         # say what would be done
    python update_corpus.py --status          # the manifest, summarised
    python update_corpus.py --collection PLATO_TEST
    python update_corpus.py --only 2025ExA....59...26R --redo
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_fulltext  # noqa: E402
from fetch_fulltext import (  # noqa: E402
    FetchError, Fetched, access_of, fulltext_allowed, fulltext_sources, sha256_file,
)
from reindex_weaviate import (  # noqa: E402
    ABSTRACT_ONLY, DEFAULT_MD_DIR, DEFAULT_METADATA, FULL_TEXT, ChunkSettings, fingerprint,
    load_papers, markdown_problem, match_markdown, paper_objects, sha256_text,
)

DATA_DIR = os.environ.get("PLATO_DATA_DIR", "/Users/hilke/data/plato_data")
DEFAULT_COLLECTION = os.environ.get("PLATO_COLLECTION", "PLATO")
DEFAULT_ENV = os.path.join(HERE, os.pardir, "platochat", ".env")

# What the manifest says about a paper.
FULLTEXT = "full text"        # markdown converted and checked
ABSTRACT = "abstract only"    # paywalled: no full text, by policy
PENDING = "pending"           # full text allowed, a source still to try
RETRY = "retry"               # a source failed for now (timeout, arXiv refusing); next round or run
FAILED = "failed"             # every allowed source failed; abstract only until something changes
REMOVED = "removed"           # no longer on the list
DOWNLOADED = "downloaded"     # only during a run: fetched, not yet converted

# Temporary failures of one source (timeouts, 5xx, bot checks, arXiv refusing
# us) after which it is given up on until --retry-failed or new ADS data. Up to
# --rounds attempts are made per run, so this is some three or four nights.
MAX_TRANSIENT_TRIES = 10

# Never take more papers than this out in one run without --allow-removals: a
# list that suddenly shrinks is likelier a broken feed than a mass retraction.
REMOVAL_GUARD = 10

LATEXML_TIMEOUT = 1800        # seconds per LaTeXML tool (latexml, latexmlpost, pandoc)
MARKER_TIMEOUT = 1800         # seconds per PDF
MARKER_STARTUP = 900          # seconds for marker to load its models

MANIFEST_ABOUT = (
    "Per-paper state of the PLATO ingest, written by ingest/update_corpus.py. "
    "'state' is one of: full text, abstract only (paywalled), pending, retry, "
    "failed, removed. To have a paper fetched and converted again, run "
    "update_corpus.py --only <bibcode> --redo."
)


def log(*args) -> None:
    print(*args, flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Paths:
    def __init__(self, data_dir: str, md_dir: str):
        self.data = data_dir
        self.md = md_dir
        self.manifest = os.path.join(data_dir, "manifest.json")
        self.downloads = os.path.join(data_dir, "downloads")
        self.manual = os.path.join(data_dir, "manual")
        self.replaced = os.path.join(data_dir, "plato_markdown_replaced")
        self.logs = os.path.join(data_dir, "logs")
        self.lock = os.path.join(data_dir, ".update.lock")

    def rel(self, path: str) -> str:
        return os.path.relpath(path, self.data)


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> dict:
    if not os.path.exists(path):
        return {"about": MANIFEST_ABOUT, "papers": {}}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_manifest(path: str, manifest: dict) -> None:
    manifest["about"] = MANIFEST_ABOUT
    manifest["updated_at"] = now_iso()
    manifest["papers"] = dict(sorted(manifest["papers"].items()))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def inputs_of(rec: dict) -> str:
    """What ADS says about where a paper's full text is to be had.

    When this changes -- a new arXiv id, an open-access flag, a new link -- a
    paper whose sources had all failed is tried again.
    """
    basis = [rec.get("arxiv_id") or "", sorted(rec.get("esources") or []),
             sorted(p for p in rec.get("property") or [] if p.endswith("OPENACCESS")),
             sorted(rec.get("doi") or [])]
    return hashlib.sha256(json.dumps(basis).encode("utf-8")).hexdigest()[:12]


def manual_pdf(paths: Paths, rec: dict) -> str | None:
    """A PDF saved by hand for this paper, under any of its bibcodes."""
    for name in dict.fromkeys([rec["bibcode"], *rec.get("aliases", [])]):
        path = os.path.join(paths.manual, name + ".pdf")
        if os.path.exists(path):
            return path
    return None


def note_try(entry: dict, source: str, result: str, transient: bool, **extra) -> None:
    """Record the latest outcome of one source (one record per source, not a log)."""
    tried = entry.setdefault("tried", {})
    count = tried.get(source, {}).get("count", 0) + 1
    tried[source] = {"result": result, "transient": transient, "at": now_iso(),
                     "count": count, **extra}


def remaining_sources(entry: dict) -> list[str]:
    """The allowed sources not yet ruled out.

    A temporary failure rules a source out only once it has happened
    MAX_TRANSIENT_TRIES times in a row: a server that times out every night,
    or a publisher whose bot check never lets us through, should end up on the
    list for downloading by hand rather than be asked for ever.
    """
    tried = entry.get("tried") or {}
    return [s for s in entry.get("sources") or []
            if s not in tried
            or (tried[s].get("transient") and tried[s].get("count", 0) < MAX_TRANSIENT_TRIES)]


def set_aside(paths: Paths, name: str) -> None:
    """Move a paper's markdown and images out of the corpus, keeping them for reference."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for suffix in (".md", "_images"):
        src = os.path.join(paths.md, name + suffix)
        if os.path.exists(src):
            os.makedirs(paths.replaced, exist_ok=True)
            shutil.move(src, os.path.join(paths.replaced, f"{stamp}-{name}{suffix}"))


def derive_state(entry: dict) -> str:
    if not entry.get("sources"):
        return ABSTRACT
    if entry.get("fulltext"):
        return FULLTEXT
    remaining = remaining_sources(entry)
    if not remaining:
        return FAILED
    tried = entry.get("tried") or {}
    return RETRY if any(s in tried for s in remaining) else PENDING


def reconcile(manifest: dict, papers: dict, aliases: dict, paths: Paths, *,
              retry_failed: bool = False, redo: set[str] = frozenset()) -> dict[str, list]:
    """Bring the manifest in line with the list; returns what changed, by kind.

    Also adopts markdown already in the corpus folder for papers the manifest
    has no full text for -- how the first run takes over the 27 papers
    converted before the manifest existed, and how a hand-made conversion
    gets in.
    """
    entries = manifest["papers"]
    notes: dict[str, list] = {k: [] for k in ("new", "renamed", "removed", "adopted", "reopened")}
    today = now_iso()[:10]

    # An entry filed under a bibcode the paper no longer has: a preprint that
    # has since been published.
    for key in list(entries):
        current = aliases.get(key)
        if current and current != key:
            entry = entries.pop(key)
            if current in entries and (entries[current].get("fulltext") or not entry.get("fulltext")):
                continue
            entry.setdefault("former_bibcodes", []).append(key)
            entries[current] = entry
            notes["renamed"].append(f"{key} -> {current}")

    md_by_paper = None
    for bib, rec in papers.items():
        entry = entries.get(bib)
        if entry is None:
            entry = entries[bib] = {"first_seen": today}
            notes["new"].append(bib)
        entry.pop("removed", None)
        entry["title"] = rec.get("title") or ""
        entry["access"] = access_of(rec)
        manual = manual_pdf(paths, rec)
        entry["sources"] = fulltext_sources(rec, manual)

        inputs = inputs_of(rec)
        if entry.get("inputs") != inputs:
            if entry.get("inputs") and not entry.get("fulltext") and entry.get("tried"):
                entry["tried"] = {}
                notes["reopened"].append(bib)
            entry["inputs"] = inputs
        tried_manual = (entry.get("tried") or {}).get("manual")
        if manual and tried_manual and tried_manual.get("sha256") != sha256_file(manual):
            del entry["tried"]["manual"]            # a different file than last time
            notes["reopened"].append(bib)
        if retry_failed and not entry.get("fulltext") and entry.get("tried"):
            entry["tried"] = {}
            notes["reopened"].append(bib)
        if bib in redo:
            ft = entry.pop("fulltext", None)
            if ft and ft.get("markdown"):
                set_aside(paths, ft["markdown"][:-3])
            entry["tried"] = {}
            notes["reopened"].append(bib)

        if entry["sources"] and not entry.get("fulltext") and bib not in redo:
            if md_by_paper is None:
                md_by_paper, _ = match_markdown(paths.md, aliases)
            path = md_by_paper.get(bib)
            if path:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
                if not markdown_problem(text, rec):
                    entry["fulltext"] = {
                        "source": "adopted", "markdown": os.path.basename(path),
                        "markdown_sha256": sha256_text(text), "at": now_iso(),
                        "note": "already in the markdown folder when first seen",
                    }
                    notes["adopted"].append(bib)
        entry["state"] = derive_state(entry)

    for key, entry in entries.items():
        if key not in papers and entry.get("state") != REMOVED:
            entry["state"] = REMOVED
            entry["removed"] = today
            notes["removed"].append(key)
    return notes


# ---------------------------------------------------------------------------
# Step 1: metadata
# ---------------------------------------------------------------------------

def refresh_metadata(metadata_path: str, env_path: str, allow_removals: bool) -> bool:
    """Rewrite papers.json from the live list; False if it had to be kept as it was."""
    from fetch_metadata import fetch_papers, load_key, write_papers

    old = {}
    if os.path.exists(metadata_path):
        old, _ = load_papers(metadata_path)
    try:
        new = fetch_papers(load_key(env_path))
    except Exception as exc:  # network, ADS down, a changed feed
        log(f"  ! could not refresh the list ({type(exc).__name__}: {exc}); "
            f"carrying on with the papers.json from before")
        return False
    known = {a for rec in new.values() for a in rec.get("aliases", [rec["bibcode"]])}
    gone = [b for b in old if b not in known]
    added = [b for b in new if not any(a in old for a in new[b].get("aliases", [b]))]
    if len(gone) > REMOVAL_GUARD and not allow_removals:
        log(f"  ! the list lost {len(gone)} of {len(old)} papers at once -- more likely a "
            f"broken feed than real removals, so papers.json is left as it was. "
            f"Re-run with --allow-removals if the list really did shrink.")
        return False
    write_papers(new, metadata_path)
    log(f"  papers.json: {len(new)} papers ({len(added)} new, {len(gone)} gone)")
    return True


# ---------------------------------------------------------------------------
# Step 2: fetch and convert
# ---------------------------------------------------------------------------

def find_marker_python(explicit: str | None = None) -> str | None:
    """A Python that can import marker: given, in PLATO_MARKER_PYTHON, this one,
    or the one that runs marker's own ``marker_single`` command."""
    for cand in (explicit, os.environ.get("PLATO_MARKER_PYTHON")):
        if cand:
            return cand
    if importlib.util.find_spec("marker") is not None:
        return sys.executable
    exe = shutil.which("marker_single")
    if exe:
        with open(exe, "rb") as fh:
            first = fh.readline().decode("utf-8", "replace").strip()
        if first.startswith("#!"):
            parts = first[2:].split()
            if parts and os.path.basename(parts[0]) == "env" and len(parts) > 1:
                return shutil.which(parts[1])
            return parts[0] if parts else None
    return None


def download_one(bib: str, rec: dict, entry: dict, paths: Paths, ads_key: str) -> Fetched | None:
    """The first of the paper's remaining sources that downloads, or None."""
    manual = manual_pdf(paths, rec)
    for source in remaining_sources(entry):
        if source == "manual" and not manual:
            continue                      # removed from the folder since the run began
        try:
            got = fetch_fulltext.fetch(rec, source, os.path.join(paths.downloads, bib),
                                       ads_key=ads_key, manual_pdf=manual)
        except FetchError as exc:
            log(f"  {bib}  {source}: {'will retry -- ' if exc.transient else ''}{exc}")
            extra = {"sha256": sha256_file(manual)} if source == "manual" else {}
            note_try(entry, source, str(exc), exc.transient, **extra)
            if exc.transient:
                entry["state"] = RETRY
                return None
            continue
        tried = entry.get("tried") or {}
        for name in {source, got.source}:     # an earlier temporary failure is history now
            if tried.get(name, {}).get("transient"):
                del tried[name]
        if got.source != source:          # arXiv had a PDF where the LaTeX should be
            note_try(entry, source, got.note, False)
        entry["state"] = DOWNLOADED
        log(f"  {bib}  {got.source}: downloaded")
        return got
    entry["state"] = derive_state(entry)
    return None


def convert_tex(bib: str, got: Fetched, paths: Paths) -> dict:
    from latexml_to_markdown import latex_to_markdown

    out = os.path.join(paths.downloads, bib, f"md-{got.source}")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    job = {"markdown": os.path.join(out, "paper.md"), "image_dir": os.path.join(out, "images"),
           "converter": "latexml", "error": None, "transient": False}
    t0 = time.time()
    try:
        markdown = latex_to_markdown(got.path, image_dir=job["image_dir"],
                                     timeout=LATEXML_TIMEOUT)
        with open(job["markdown"], "w", encoding="utf-8") as fh:
            fh.write(markdown)
    except Exception as exc:
        job["error"] = f"{type(exc).__name__}: {str(exc).strip()[:300]}"
    job["seconds"] = round(time.time() - t0, 1)
    return job


def convert_pdfs(items: list[tuple[str, Fetched]], paths: Paths, marker_py: str | None,
                 problems: list[str]) -> dict:
    """Run marker over a batch of PDFs in one subprocess; ``{bibcode: job}``.

    A marker that is missing or dies is an infrastructure problem, not a
    property of the PDFs: it goes into ``problems``, which makes the run exit
    non-zero, so that a cron monitor notices.
    """
    jobs = {}
    for bib, got in items:
        out = os.path.join(paths.downloads, bib, f"md-{got.source}")
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        jobs[bib] = {"id": bib, "pdf": got.path, "markdown": os.path.join(out, "paper.md"),
                     "image_dir": os.path.join(out, "images"), "converter": "marker",
                     "error": None, "transient": False, "seconds": 0}
    if not marker_py:
        for job in jobs.values():
            job.update(error="marker not found: install marker-pdf, or point "
                             "PLATO_MARKER_PYTHON at a Python that has it", transient=True)
        problems.append("marker not found")
        return jobs

    os.makedirs(paths.logs, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jobs_file = os.path.join(paths.logs, f"marker-{stamp}.jobs.json")
    log_file = os.path.join(paths.logs, f"marker-{stamp}.log")
    with open(jobs_file, "w", encoding="utf-8") as fh:
        json.dump([{k: j[k] for k in ("id", "pdf", "markdown", "image_dir")}
                   for j in jobs.values()], fh, indent=1)
    cmd = [marker_py, os.path.join(HERE, "pdf_to_markdown.py"), "--jobs", jobs_file,
           "--timeout", str(MARKER_TIMEOUT)]
    log(f"  marker: {len(jobs)} PDF(s), log in {log_file}")
    done, current = set(), None
    with open(log_file, "w", encoding="utf-8") as errlog:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errlog, text=True, cwd=HERE)
        watchdog = threading.Timer(MARKER_STARTUP + MARKER_TIMEOUT * len(jobs), proc.kill)
        watchdog.start()
        try:
            for line in proc.stdout:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue            # marker's own chatter
                if msg.get("event") == "start":
                    current = msg["id"]
                elif msg.get("event") == "done" and msg.get("id") in jobs:
                    job = jobs[msg["id"]]
                    job["seconds"] = msg.get("seconds", 0)
                    if not msg.get("ok"):
                        job["error"] = msg.get("error") or "marker failed"
                    done.add(msg["id"])
                    current = None
                    log(f"  marker: {msg['id']} {'converted' if msg.get('ok') else 'FAILED'} "
                        f"in {job['seconds']:.0f} s")
            proc.wait()
        finally:
            watchdog.cancel()
    if proc.returncode != 0:
        problems.append(f"marker exited {proc.returncode} (see {log_file})")
    for bib, job in jobs.items():
        if bib in done:
            continue
        if bib == current:    # it died on this one: do not feed it the same PDF again
            job["error"] = f"marker died on this PDF (exit {proc.returncode}); see {log_file}"
        else:
            job.update(error=f"marker stopped before this PDF (exit {proc.returncode})",
                       transient=True)
    return jobs


def place_markdown(bib: str, job: dict, paths: Paths) -> str:
    """Move a checked conversion into the corpus as <bibcode>.md + <bibcode>_images."""
    set_aside(paths, bib)
    target = os.path.join(paths.md, bib + ".md")
    shutil.move(job["markdown"], target)
    if os.path.isdir(job["image_dir"]):
        shutil.move(job["image_dir"], os.path.join(paths.md, bib + "_images"))
    return target


def finish(bib: str, rec: dict, entry: dict, got: Fetched, job: dict, paths: Paths) -> bool:
    """Check a conversion; place it and record it, or record why not."""
    error = job.get("error")
    if not error:
        with open(job["markdown"], encoding="utf-8") as fh:
            text = fh.read()
        error = markdown_problem(text, rec)
    if error:
        note_try(entry, got.source, f"{job['converter']}: {error}", job.get("transient", False))
        entry["state"] = derive_state(entry)
        log(f"  {bib}  {got.source}: conversion failed -- {error}")
        return False
    place_markdown(bib, job, paths)
    entry["fulltext"] = {
        "source": got.source, "url": got.url, "downloaded": paths.rel(got.path),
        "sha256": got.sha256, "converter": job["converter"], "seconds": job["seconds"],
        "markdown": bib + ".md", "markdown_sha256": sha256_text(text), "at": now_iso(),
        **({"note": got.note} if got.note else {}),
    }
    entry["state"] = FULLTEXT
    log(f"  {bib}  {got.source}: converted with {job['converter']} in {job['seconds']:.0f} s")
    return True


def fetch_stage(manifest: dict, papers: dict, paths: Paths, args,
                problems: list[str]) -> list[str]:
    """Download and convert what is missing; returns the papers that got full text."""
    entries = manifest["papers"]
    todo = [b for b in papers if entries[b]["state"] in (PENDING, RETRY)
            and (not args.only or b in args.only)]
    if not todo:
        log("Nothing to fetch.")
        return []
    if args.dry_run:
        for b in todo:
            e = entries[b]
            log(f"  would fetch {b}  ({e['state']}; next: {', '.join(remaining_sources(e))})"
                f"  {e['title'][:60]}")
        return []

    from fetch_metadata import load_key
    ads_key = load_key(args.env)
    marker_py = find_marker_python(args.marker_python)
    # latexml_to_markdown warns, with LaTeXML's whole log attached, whenever
    # latexmlpost "reported problems but produced" its output -- which is most
    # papers, and harmless: the markdown is checked in finish() either way.
    warnings.filterwarnings("ignore", message=r".* reported problems but produced ")
    added: list[str] = []
    for round_no in range(1, args.rounds + 1):
        work = [b for b in todo if entries[b]["state"] in (PENDING, RETRY)]
        if not work:
            break
        if round_no > 1:
            if any(entries[b]["state"] == RETRY for b in work):
                log(f"Waiting {args.round_pause} s before trying again …")
                time.sleep(args.round_pause)
        fetch_fulltext.reset_throttle()
        log(f"Fetch round {round_no}: {len(work)} paper(s)")

        downloaded: list[tuple[str, Fetched]] = []
        for bib in work:
            got = download_one(bib, papers[bib], entries[bib], paths, ads_key)
            if got:
                downloaded.append((bib, got))
            save_manifest(paths.manifest, manifest)

        # Both converters at once: marker's batch in a thread of its own (it
        # works the GPU), several LaTeXML runs beside it (each is one Perl
        # process on one core, and a paper can take ten minutes). Results are
        # checked and recorded here, in the main thread, as they come in.
        tex = [d for d in downloaded if d[1].kind == "tex"]
        pdfs = [d for d in downloaded if d[1].kind == "pdf"]
        with ThreadPoolExecutor(max_workers=1) as marker_pool, \
                ThreadPoolExecutor(max_workers=max(1, args.latexml_jobs)) as tex_pool:
            marker_done = (marker_pool.submit(convert_pdfs, pdfs, paths, marker_py, problems)
                           if pdfs else None)
            running = {}
            for bib, got in tex:
                log(f"  {bib}  LaTeXML: {os.path.relpath(got.path, paths.downloads)} …")
                running[tex_pool.submit(convert_tex, bib, got, paths)] = (bib, got)
            for done in as_completed(running):
                bib, got = running[done]
                if finish(bib, papers[bib], entries[bib], got, done.result(), paths):
                    added.append(bib)
                save_manifest(paths.manifest, manifest)
            if marker_done:
                jobs = marker_done.result()
                for bib, got in pdfs:
                    if finish(bib, papers[bib], entries[bib], got, jobs[bib], paths):
                        added.append(bib)
                save_manifest(paths.manifest, manifest)
    return added


# ---------------------------------------------------------------------------
# Step 3: index
# ---------------------------------------------------------------------------

def index_stage(manifest: dict, papers: dict, paths: Paths, args,
                settings: ChunkSettings) -> dict:
    """Write the objects of every paper whose fingerprint changed; delete orphans."""
    import weaviate
    from reindex_weaviate import (
        delete_objects, embed, ensure_collection, index_state, load_embedder,
        load_tokenizer_counter, replace_paper,
    )

    entries = manifest["papers"]
    stats = Counter()
    client = weaviate.connect_to_local()
    try:
        if not client.is_ready():
            raise RuntimeError("Weaviate is not ready")
        if args.dry_run and not client.collections.exists(args.collection):
            state = {}
            log(f"  collection '{args.collection}' does not exist yet; it would be created")
        else:
            collection = (client.collections.get(args.collection) if args.dry_run
                          else ensure_collection(client, args.collection))
            state = index_state(collection)

        plan, claimed = [], set()
        for bib, rec in papers.items():
            keys = {bib, *rec.get("aliases", [])}
            old = {u: fp for k in keys for u, fp in state.get(k, {}).items()}
            claimed.update(old)
            if args.only and bib not in args.only:
                continue
            entry = entries[bib]
            markdown = md_sha = None
            ft = entry.get("fulltext")
            if entry["state"] == FULLTEXT and ft and fulltext_allowed(rec):
                path = os.path.join(paths.md, ft["markdown"])
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as fh:
                        text = fh.read()
                    md_sha = sha256_text(text)
                    if md_sha != ft.get("markdown_sha256"):
                        log(f"  {bib}: its markdown was edited since it was converted")
                        ft["markdown_sha256"], ft["edited"] = md_sha, now_iso()
                    problem = markdown_problem(text, rec)
                    if problem:
                        log(f"  ! {bib}: its markdown no longer passes the checks ({problem}); "
                            f"indexing the abstract")
                    else:
                        markdown = text
                else:
                    # The markdown is gone (the server need not keep it). The
                    # objects stay as long as nothing else about the paper
                    # changed; otherwise it is fetched again next run.
                    kept = fingerprint(rec, bib, ft.get("markdown_sha256"), settings)
                    if old and set(old.values()) == {kept}:
                        continue
                    log(f"  ! {bib}: its markdown is gone and its objects are out of date; "
                        f"it will be fetched again next run")
                    entry.pop("fulltext")
                    entry["state"] = derive_state(entry)
                    if old:
                        continue        # keep the old full text rather than fall back
            fp = fingerprint(rec, bib, md_sha if markdown else None, settings)
            if old and set(old.values()) == {fp}:
                continue
            plan.append((bib, rec, markdown, fp, old))

        orphans: dict[str, set] = {}
        if not args.only:
            for parent, objs in state.items():
                stray = {u for u in objs if u not in claimed}
                if stray:
                    orphans[parent] = stray

        n_new = sum(1 for p in plan if not p[4])
        log(f"  {len(plan)} paper(s) to write ({n_new} new, {len(plan) - n_new} changed), "
            f"{len(orphans)} to remove")
        if args.dry_run:
            for bib, rec, markdown, _, old in plan[:40]:
                log(f"    {bib}  {'full text' if markdown else 'abstract'}"
                    f"{'' if old else '  (new)'}")
            if len(plan) > 40:
                log(f"    … and {len(plan) - 40} more")
            for parent in orphans:
                log(f"    remove {parent} ({len(orphans[parent])} objects)")
            return stats

        if plan:
            count_tokens = load_tokenizer_counter(settings.model)
            embedder = load_embedder(settings.model, args.device)
        for i, (bib, rec, markdown, fp, old) in enumerate(plan, start=1):
            objects = paper_objects(rec, bib, markdown, settings, count_tokens)
            for obj in objects:
                obj["properties"]["fingerprint"] = fp
            written, deleted = replace_paper(collection, objects, embed(embedder, objects), old)
            coverage = FULL_TEXT if markdown else ABSTRACT_ONLY
            entries[bib].setdefault("indexed", {})[args.collection] = {
                "coverage": coverage, "objects": written, "fingerprint": fp, "at": now_iso()}
            stats["papers written"] += 1
            stats["objects written"] += written
            stats["objects deleted"] += deleted
            log(f"  [{i}/{len(plan)}] {bib}  {coverage}, {written} object{'s' * (written != 1)}"
                + (f" ({deleted} old removed)" if deleted else "")
                + ("" if old else "  (new)"))
            if i % 10 == 0:
                save_manifest(paths.manifest, manifest)

        if len(orphans) > REMOVAL_GUARD and not args.allow_removals:
            log(f"  ! {len(orphans)} papers in '{args.collection}' are not on the list; "
                f"refusing to remove that many at once (--allow-removals)")
        elif orphans:
            for parent, uuids in orphans.items():
                stats["objects deleted"] += delete_objects(collection, uuids)
                stats["papers removed"] += 1
                log(f"  removed {parent} ({len(uuids)} object{'s' * (len(uuids) != 1)}): "
                    f"no longer on the list")
                entry = entries.get(parent)
                if entry and "indexed" in entry:
                    entry["indexed"].pop(args.collection, None)
        if not args.dry_run:
            log(f"  '{args.collection}' now holds {len(collection)} objects")
    finally:
        client.close()
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summary(manifest: dict, papers: dict) -> None:
    entries = {b: e for b, e in manifest["papers"].items() if b in papers}
    states = Counter(e["state"] for e in entries.values())
    access = Counter(e["access"] for e in entries.values())
    sources = Counter(e["fulltext"]["source"] for e in entries.values()
                      if e["state"] == FULLTEXT)
    log(f"\n{len(entries)} papers on the list: {access['arxiv']} on arXiv, "
        f"{access['open access']} open access elsewhere, {access['paywalled']} paywalled")
    log(f"  full text:      {states[FULLTEXT]}  "
        f"({', '.join(f'{k} {v}' for k, v in sources.most_common())})")
    log(f"  abstract only:  {len(entries) - states[FULLTEXT]}  ({states[ABSTRACT]} paywalled, "
        f"{states[FAILED]} failed, {states[RETRY]} to retry, {states[PENDING]} not tried yet"
        + (f", {states[DOWNLOADED]} being converted" if states[DOWNLOADED] else "") + ")")
    removed = sum(1 for e in manifest["papers"].values() if e["state"] == REMOVED)
    if removed:
        log(f"  no longer listed: {removed}")

    stuck = [(b, e) for b, e in entries.items() if e["state"] in (FAILED, RETRY)]
    if stuck:
        log("\nOpen-access papers without full text. For those a script cannot fetch, save the "
            "PDF by hand as <data>/manual/<bibcode>.pdf and the next run takes it from there:")
        for b, e in sorted(stuck):
            why = "; ".join(f"{s}: {t['result']}" for s, t in (e.get("tried") or {}).items())
            log(f"  {b}  [{e['state']}]  {e['title'][:60]}\n      {why[:300]}\n"
                f"      {papers[b].get('ads_url', '')}")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help=f"Weaviate collection to update (default: {DEFAULT_COLLECTION})")
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help=f"downloads, manifest, manual PDFs (default: {DATA_DIR})")
    ap.add_argument("--md-dir", default=DEFAULT_MD_DIR, help="the markdown corpus")
    ap.add_argument("--metadata", default=DEFAULT_METADATA)
    ap.add_argument("--env", default=DEFAULT_ENV, help="file holding ADS_API_KEY")
    ap.add_argument("--stages", default="metadata,fetch,index",
                    help="which steps to run (default: metadata,fetch,index)")
    ap.add_argument("--dry-run", action="store_true",
                    help="say what would be done; download, convert and write nothing")
    ap.add_argument("--status", action="store_true", help="summarise the manifest and stop")
    ap.add_argument("--only", nargs="+", metavar="BIBCODE", help="restrict to these papers")
    ap.add_argument("--redo", action="store_true",
                    help="with --only: fetch and convert those papers again")
    ap.add_argument("--retry-failed", action="store_true",
                    help="try again every source that failed before")
    ap.add_argument("--allow-removals", action="store_true",
                    help=f"allow more than {REMOVAL_GUARD} papers to leave in one run")
    ap.add_argument("--rounds", type=int, default=3,
                    help="fetch rounds per run, for sources that fail for now (default: 3)")
    ap.add_argument("--round-pause", type=int, default=120,
                    help="seconds between fetch rounds (default: 120)")
    ap.add_argument("--latexml-jobs", type=int, default=4,
                    help="LaTeX conversions to run side by side (default: 4)")
    ap.add_argument("--marker-python", help="Python with marker installed "
                    "(default: $PLATO_MARKER_PYTHON, else found via marker_single)")
    ap.add_argument("--device", default=os.environ.get("PLATO_EMBED_DEVICE"),
                    help="torch device for embedding (default: mps/cuda/cpu)")
    args = ap.parse_args()
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    if args.redo and not args.only:
        ap.error("--redo needs --only: say which papers to redo")

    paths = Paths(args.data_dir, args.md_dir)
    os.makedirs(paths.data, exist_ok=True)
    manifest = load_manifest(paths.manifest)

    if args.status:
        papers, _ = load_papers(args.metadata)
        summary(manifest, papers)
        last = manifest.get("last_run")
        if last:
            log(f"\nLast run: {last['at']}, {'ok' if last['ok'] else 'with problems'}; "
                f"{last.get('summary', '')}")
        return 0

    if args.dry_run:
        return run(args, paths, manifest, stages, ap)
    # One run at a time: cron may start the next while a slow one is still busy.
    with open(paths.lock, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("Another update is running; leaving it to finish.")
            return 0
        return run(args, paths, manifest, stages, ap)


def run(args, paths: Paths, manifest: dict, stages: set[str], ap) -> int:
    t0 = time.time()
    problems: list[str] = []
    started = now_iso()
    log(f"PLATO corpus update, {started} -> collection '{args.collection}'"
        + ("  (dry run)" if args.dry_run else ""))

    if "metadata" in stages and not args.dry_run and not args.only:
        log("\n1. Metadata")
        if not refresh_metadata(args.metadata, args.env, args.allow_removals):
            problems.append("the list could not be refreshed")

    papers, aliases = load_papers(args.metadata)
    if args.only:
        missing = [b for b in args.only if b not in aliases]
        if missing:
            ap.error(f"not on the list: {', '.join(missing)}")
        args.only = {aliases[b] for b in args.only}
    notes = reconcile(manifest, papers, aliases, paths, retry_failed=args.retry_failed,
                      redo=args.only if args.redo else set())
    for kind, items in notes.items():
        if items:
            log(f"  {kind}: {len(items)}" + (f" ({', '.join(items[:6])}"
                                             + (", …" if len(items) > 6 else "") + ")"))
    if not args.dry_run:
        save_manifest(paths.manifest, manifest)

    added: list[str] = []
    if "fetch" in stages:
        log("\n2. Fetch and convert")
        added = fetch_stage(manifest, papers, paths, args, problems)

    stats: Counter = Counter()
    if "index" in stages:
        log(f"\n3. Index ('{args.collection}')")
        try:
            stats = index_stage(manifest, papers, paths, args, ChunkSettings())
        except Exception as exc:
            log(f"  !! indexing failed: {type(exc).__name__}: {exc}")
            problems.append(f"indexing failed ({type(exc).__name__})")

    summary(manifest, papers)
    took = time.time() - t0
    line = (f"{len(added)} full text(s) added, {stats['papers written']} paper(s) written, "
            f"{stats['papers removed']} removed, in {took / 60:.0f} min")
    log(f"\nThis run: {line}" + (f"; problems: {'; '.join(problems)}" if problems else ""))
    if not args.dry_run:
        manifest["last_run"] = {"at": started, "ok": not problems, "summary": line,
                                "collection": args.collection, "problems": problems}
        save_manifest(paths.manifest, manifest)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
