#!/usr/bin/env python
"""Look up the licence of every paper, and of the copy of it that we hold.

Whether we download a full text at all is decided by *access*
(fetch_fulltext.py): on arXiv, or flagged open access by ADS. That makes a
paper free to read. Whether a public chatbot may pass its text on is a matter
of its *licence*, which this script records in the manifest,
<data>/manifest.json, under "licence" for each paper, and which
:func:`indexable` turns into the rule for what goes into the index. It asks:

* arXiv, for the licence the authors chose when posting the preprint (its
  OAI-PMH record; arXiv's search API does not carry it). Most pick arXiv's
  default, a non-exclusive licence *to arXiv* to distribute the paper, which
  gives nobody else any rights.
* Crossref, for the licences the publisher deposited with the DOI, and
  DataCite for the DOIs Crossref does not know (Zenodo's).
* OpenAlex, for its reading of the publisher's landing page: a licence where
  one is stated, and the kind of open access -- "bronze" means free to read on
  the publisher's site, with no licence at all.

Not the papers themselves: a licence statement printed in the text would be
the only evidence for proceedings without a DOI, but none of our 32 downloaded
PDFs carries one in the text of its first or last two pages, and none of the
166 markdown files mentions Creative Commons (checked 2026-09-22). marker
drops page footers, where publishers print them.

From that it works out the licence of the copy we hold -- the preprint when
the text came from arXiv, the published version when it came from the
publisher or from an ADS scan -- and puts it in one of three classes:

open licence      CC BY, CC BY-SA, CC0: may be reused, with attribution
with conditions   CC BY-NC(-SA), CC BY-ND, CC BY-NC-ND: may be reused by a
                  free, non-commercial service; ND only verbatim
no licence        arXiv's default licence, a publisher's own terms, free to
                  read only, or nothing found: no permission beyond reading.
                  Quoting from these rests on copyright exceptions (quotation,
                  text and data mining), not on a licence.

This records what the sources say; it is not legal advice. What the index does
with it is :func:`indexable`: a copy the authors put in the open goes in
whatever its licence, a publisher's copy only under an open one.

update_corpus.py runs the lookup as part of every update, for new papers and
for papers ADS reports something new about: a new arXiv id, DOI, link or
open-access flag (the manifest's ``inputs``, the same signal that makes the
fetch step try a paper again). There is no periodic recheck (the user's
choice, 2026-09-22), so a licence that changes with none of those -- an
author picking CC BY for a new arXiv version -- goes unnoticed until
``update_corpus.py --recheck-licences``.

    python fetch_licences.py '2014A&A...566A..92M'   # ask the sources, print, write nothing
    python fetch_licences.py --report                # what the manifest says, summarised
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fetch_fulltext import FetchError, http_get  # noqa: E402

ARXIV_OAI = "https://oaipmh.arxiv.org/oai"      # export.arxiv.org/oai2 redirects here
CROSSREF = "https://api.crossref.org/works"
DATACITE = "https://api.datacite.org/dois"
OPENALEX = "https://api.openalex.org/works"

# arXiv's own DOIs (10.48550/arXiv.<id>) name the preprint, which arXiv's
# record covers; a paper's other DOIs name the published version.
ARXIV_DOI_PREFIX = "10.48550/"

CROSSREF_BATCH = 20          # DOIs per request: one filter each, all in the URL
OPENALEX_BATCH = 50          # OpenAlex's limit for an OR filter

# The classes, most permissive first.
OPEN = "open licence"
CONDITIONS = "with conditions"
NO_LICENCE = "no licence"
CLASSES = (OPEN, CONDITIONS, NO_LICENCE)

# Which copy of a paper the index holds.
PREPRINT = "arXiv preprint"
PUBLISHED = "published version"
AUTHOR_COPY = "author's copy"

ARXIV_DEFAULT = "arXiv default licence"
NOT_AT_ARXIV = "none recorded at arXiv"
FREE_TO_READ = "free to read, no licence"
NONE_FOUND = "none found"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Naming and classifying licences
# ---------------------------------------------------------------------------

_CC_URL_RE = re.compile(r"creativecommons\.org/licen[cs]es/([a-z-]+)/(\d(?:\.\d)?)", re.I)
# CC0, the Public Domain Mark, and the public-domain dedication arXiv offered
# before CC0 (creativecommons.org/licenses/publicdomain/).
_CC_PUBLIC_RE = re.compile(
    r"creativecommons\.org/(?:publicdomain/(zero|mark)|licen[cs]es/publicdomain)", re.I)


def label_from_url(url: str) -> str:
    """A short name for a licence URL: 'CC BY 4.0', 'arXiv default licence',
    or 'publisher's terms (cambridge.org)'."""
    m = _CC_URL_RE.search(url)
    if m:
        return f"CC {m.group(1).upper()} {m.group(2)}"
    m = _CC_PUBLIC_RE.search(url)
    if m:
        return "CC0" if (m.group(1) or "").lower() == "zero" else "public domain"
    if "arxiv.org/licenses/nonexclusive-distrib" in url:
        return ARXIV_DEFAULT
    if "arxiv.org/licenses/assumed-1991-2003" in url:
        return "arXiv, before licences (1991-2003)"
    host = urllib.parse.urlsplit(url).hostname or url
    return f"publisher's terms ({host.removeprefix('www.')})"


def label_from_openalex(licence: str) -> str:
    """OpenAlex's licence ids ('cc-by-nc', 'publisher-specific-oa') as labels."""
    lic = licence.lower().rsplit("/", 1)[-1]
    if lic.startswith("cc-"):
        return "CC " + lic[3:].upper()
    if lic in ("cc0", "public-domain"):
        return "CC0" if lic == "cc0" else "public domain"
    if lic == "publisher-specific-oa":
        return "publisher's terms (OpenAlex)"
    return f"no open licence (OpenAlex: {lic})"


def class_of(label: str | None) -> str:
    if label in ("CC0", "public domain"):
        return OPEN
    if label and label.startswith("CC BY"):
        kind = label.split()[1]                     # 'BY', 'BY-SA', 'BY-NC-ND'
        return CONDITIONS if ("NC" in kind or "ND" in kind) else OPEN
    return NO_LICENCE


# ---------------------------------------------------------------------------
# Asking the sources
# ---------------------------------------------------------------------------

def _get_json(url: str) -> dict:
    _, _, body = http_get(url)
    return json.loads(body)


def publisher_dois(rec: dict) -> list[str]:
    return [d.lower() for d in rec.get("doi") or [] if not d.lower().startswith(ARXIV_DOI_PREFIX)]


_OAI ="{http://www.openarchives.org/OAI/2.0/}"
_OAI_ARXIV = "{http://arxiv.org/OAI/arXiv/}"


def arxiv_licence(arxiv_id: str) -> str | None:
    """The licence URL arXiv records for a preprint; None if it records none."""
    url = ARXIV_OAI + "?" + urllib.parse.urlencode(
        {"verb": "GetRecord", "identifier": f"oai:arXiv.org:{arxiv_id}",
         "metadataPrefix": "arXiv"})
    _, _, body = http_get(url)
    root = ET.fromstring(body)
    error = root.find(f"{_OAI}error")
    if error is not None:
        raise FetchError(f"arXiv OAI-PMH: {error.get('code')}: {(error.text or '').strip()}")
    lic = root.find(f".//{_OAI_ARXIV}license")
    return lic.text.strip() if lic is not None and lic.text else None


def crossref_licences(dois: list[str]) -> tuple[dict[str, list], dict[str, str]]:
    """``({doi: [licence, ...]}, {doi: error})`` for the DOIs Crossref knows."""
    found, failed = {}, {}
    for i in range(0, len(dois), CROSSREF_BATCH):
        batch = dois[i:i + CROSSREF_BATCH]
        url = CROSSREF + "?" + urllib.parse.urlencode(
            {"filter": ",".join("doi:" + d for d in batch), "rows": len(batch),
             "select": "DOI,license"})
        try:
            items = _get_json(url)["message"]["items"]
        except (FetchError, ValueError, KeyError) as exc:
            failed.update({d: f"Crossref: {exc}" for d in batch})
            continue
        for item in items:
            found[item["DOI"].lower()] = [
                {"url": lic.get("URL", ""),
                 "applies_to": lic.get("content-version", "unspecified"),
                 "from": ((lic.get("start") or {}).get("date-time") or "")[:10],
                 "delay_days": lic.get("delay-in-days", 0)}
                for lic in item.get("license") or []]
    return found, failed


def datacite_rights(doi: str) -> list[dict] | None:
    """DataCite's rights statements for a DOI; None if DataCite does not know it."""
    try:
        data = _get_json(f"{DATACITE}/{urllib.parse.quote(doi, safe='/')}")
    except FetchError as exc:
        if str(exc).startswith("HTTP 404"):
            return None
        raise
    return [{"url": r.get("rightsUri", ""), "name": r.get("rights", "")}
            for r in data["data"]["attributes"].get("rightsList") or []]


def openalex_works(dois: list[str]) -> tuple[dict[str, dict], dict[str, str]]:
    """``({doi: {...}}, {doi: error})``: OpenAlex's open-access status and licences."""
    found, failed = {}, {}
    for i in range(0, len(dois), OPENALEX_BATCH):
        batch = dois[i:i + OPENALEX_BATCH]
        url = OPENALEX + "?" + urllib.parse.urlencode(
            {"filter": "doi:" + "|".join(batch), "per-page": len(batch),
             "select": "doi,open_access,primary_location,locations"})
        try:
            works = _get_json(url)["results"]
        except (FetchError, ValueError, KeyError) as exc:
            failed.update({d: f"OpenAlex: {exc}" for d in batch})
            continue
        for w in works:
            doi = (w.get("doi") or "").lower().removeprefix("https://doi.org/")
            primary = w.get("primary_location") or {}
            found[doi] = {
                "oa_status": (w.get("open_access") or {}).get("oa_status"),
                "licence": primary.get("license"),
                # Other copies with a stated licence: repositories, arXiv.
                "elsewhere": sorted({
                    f"{(loc.get('source') or {}).get('display_name') or '?'}: {loc['license']}"
                    for loc in w.get("locations") or []
                    if loc.get("license") and loc.get("id") != primary.get("id")}),
            }
    return found, failed


def look_up(recs: dict[str, dict], log=print) -> tuple[dict[str, dict], dict[str, dict]]:
    """Ask every source about these papers: ``({bibcode: found}, {bibcode: {source: error}})``."""
    found = {b: {} for b in recs}
    errors = {b: {} for b in recs}
    dois = {b: publisher_dois(r) for b, r in recs.items()}
    every_doi = sorted({d for ds in dois.values() for d in ds})

    if every_doi:
        log(f"  Crossref and OpenAlex: {len(every_doi)} DOI(s)")
    crossref, cr_failed = crossref_licences(every_doi)
    openalex, oa_failed = openalex_works(every_doi)
    unknown = [d for d in every_doi if d not in crossref and d not in cr_failed]
    if unknown:
        log(f"  DataCite: {len(unknown)} DOI(s) Crossref does not know")
    datacite, dc_failed = {}, {}
    for doi in unknown:
        try:
            rights = datacite_rights(doi)
        except (FetchError, ValueError, KeyError) as exc:
            dc_failed[doi] = f"DataCite: {exc}"
            continue
        if rights is not None:
            datacite[doi] = rights

    for bib, ds in dois.items():
        for source, got, failed in (("crossref", crossref, cr_failed),
                                    ("openalex", openalex, oa_failed),
                                    ("datacite", datacite, dc_failed)):
            hits = {d: got[d] for d in ds if d in got}
            if hits:
                found[bib][source] = hits
            misses = [failed[d] for d in ds if d in failed]
            if misses:
                errors[bib][source] = misses[0]

    with_arxiv = [b for b, r in recs.items() if r.get("arxiv_id")]
    if with_arxiv:
        log(f"  arXiv: {len(with_arxiv)} preprint(s), five seconds apart")
    for bib in with_arxiv:
        arxiv_id = recs[bib]["arxiv_id"]
        try:
            found[bib]["arxiv"] = {"id": arxiv_id, "licence": arxiv_licence(arxiv_id)}
        except (FetchError, ET.ParseError) as exc:
            errors[bib]["arxiv"] = str(exc)
    return found, errors


# ---------------------------------------------------------------------------
# From what the sources say to one licence per copy
# ---------------------------------------------------------------------------

def copy_of(entry: dict) -> str | None:
    """Which version of the paper the index holds, from where its full text came."""
    source = (entry.get("fulltext") or {}).get("source")
    if not source:
        return None
    if source in ("arxiv-source", "arxiv-pdf"):
        return PREPRINT
    if source == "adopted":         # converted from arXiv LaTeX before the manifest existed
        return PREPRINT if entry.get("access") == "arxiv" else PUBLISHED
    if source == "author-pdf":
        return AUTHOR_COPY
    return PUBLISHED                # publisher-pdf, ads-scan, manual


# Sources that are the authors' own doing: they are the ones who put the paper
# there, and they are the only ones who could object to our quoting that copy.
AUTHOR_SOURCES = ("arxiv-source", "arxiv-pdf", "author-pdf")


def indexable(entry: dict) -> bool:
    """May the full text we have of this paper be indexed? (decided 2026-09-24)

    Yes for a copy the authors put in the open themselves -- an arXiv preprint,
    a thesis in a university repository -- whoever holds the copyright: a paper
    a chatbot cites gains readers rather than losing any, and no publisher is
    involved. A publisher's own copy (its PDF, or ADS's scan of the printed
    pages) only when the publisher published it under an open licence, because
    a publisher is who would complain. The paper is indexed by its abstract
    either way, as paywalled papers are.
    """
    copy = copy_of(entry)
    if copy is None:
        return False
    if copy in (PREPRINT, AUTHOR_COPY):
        return True
    return (entry.get("licence") or {}).get("published_class") in (OPEN, CONDITIONS)


def allowed_sources(entry: dict, sources: list[str]) -> list[str]:
    """Of the sources the access rule allows, those worth downloading as well.

    :func:`indexable` one step earlier: no point fetching and converting a
    publisher's copy we may not index. A licence looked up later, or one that
    changes, opens the other sources again on the next run.
    """
    if (entry.get("licence") or {}).get("published_class") in (OPEN, CONDITIONS):
        return list(sources)
    return [s for s in sources if s in AUTHOR_SOURCES]


def _most_open(options: list[tuple[str, str]]) -> tuple[str, str]:
    return min(options, key=lambda o: CLASSES.index(class_of(o[0])))


def published_licence(found: dict) -> tuple[str, str]:
    """The licence of the published version, and where that comes from.

    Where sources disagree the most permissive wins: publishers deposit their
    standard terms for every article and the Creative Commons licence only for
    the open ones, so a CC licence from any of them is the informative answer.
    """
    today = now_iso()[:10]
    options = []
    for lics in (found.get("crossref") or {}).values():
        for lic in lics:
            if lic["applies_to"] in ("vor", "unspecified") and lic["from"] <= today:
                options.append((label_from_url(lic["url"]), "Crossref"))
    for rights in (found.get("datacite") or {}).values():
        for r in rights:
            if r["url"] and not r["url"].startswith("info:eu-repo"):   # access flags, not licences
                options.append((label_from_url(r["url"]), "DataCite"))
    for work in (found.get("openalex") or {}).values():
        if work.get("licence"):
            options.append((label_from_openalex(work["licence"]), "OpenAlex"))
    if options:
        return _most_open(options)
    status = next((w["oa_status"] for w in (found.get("openalex") or {}).values()
                   if w.get("oa_status")), None)
    if status == "bronze":
        return FREE_TO_READ, "OpenAlex: bronze"
    if status:
        return NONE_FOUND, f"OpenAlex: {status}"
    return NONE_FOUND, ""


def summarise(entry: dict, rec: dict) -> None:
    """Work out the licence of the copy we indexed, from the entry's ``found``."""
    lic = entry["licence"]
    found = lic.setdefault("found", {})
    copy = copy_of(entry)      # whatever full text we have, indexed or not

    pub_label, pub_basis = published_licence(found)
    lic["published"] = pub_label
    lic["published_class"] = class_of(pub_label)
    lic["copyright"] = rec.get("copyright") or ""

    if copy == PREPRINT:
        url = (found.get("arxiv") or {}).get("licence")
        label, basis = (label_from_url(url) if url else NOT_AT_ARXIV), "arXiv"
    elif copy == PUBLISHED:
        label, basis = pub_label, pub_basis
    elif copy == AUTHOR_COPY:       # a thesis in a repository: what OpenAlex says of that copy
        options = [(label_from_openalex(e.split(": ")[-1]), "OpenAlex: " + e)
                   for work in (found.get("openalex") or {}).values()
                   for e in work.get("elsewhere") or []]
        label, basis = _most_open(options) if options else (NONE_FOUND, "")
    else:
        for key in ("copy", "copy_licence", "class", "basis"):
            lic.pop(key, None)
        return
    lic.update(copy=copy, copy_licence=label, basis=basis, **{"class": class_of(label)})


def update_licences(entries: dict, papers: dict, *, only=None,
                    force: bool = False, dry_run: bool = False, log=print) -> Counter:
    """The licence step of update_corpus.py: look up what is due, then summarise all.

    Papers are due when never looked up, when a source failed last time, and
    when ADS reports something new about them -- the entry's ``inputs``, which
    update_corpus.reconcile() keeps current. Sources that fail keep what they
    said before and are asked again on the next run.
    """
    stats: Counter = Counter()
    due = {}
    for bib, rec in papers.items():
        if only and bib not in only:
            continue
        lic = entries[bib].get("licence") or {}
        if (force or not lic.get("checked") or lic.get("errors")
                or lic.get("inputs") != entries[bib].get("inputs")):
            due[bib] = rec
    if dry_run:
        log(f"  would look up {len(due)} paper(s)")
        return stats

    if due:
        log(f"  looking up {len(due)} paper(s)")
        found, errors = look_up(due, log)
        for bib, rec in due.items():
            lic = entries[bib].setdefault("licence", {})
            old = lic.get("found") or {}
            new = found[bib]
            for source in errors[bib]:
                if source in old:
                    new[source] = old[source]
            lic["found"] = new
            lic["errors"] = errors[bib]
            if errors[bib]:
                stats["looked up with errors"] += 1
                log(f"  {bib}: {'; '.join(errors[bib].values())[:200]}")
            else:
                lic["checked"] = now_iso()
                lic["inputs"] = entries[bib].get("inputs")
                stats["looked up"] += 1

    for bib, rec in papers.items():
        if "licence" in entries[bib]:
            summarise(entries[bib], rec)
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def status_line(entries: dict, papers: dict) -> str | None:
    """One line for update_corpus.py --status; None before any lookup."""
    lics = [e["licence"] for b, e in entries.items()
            if b in papers and "class" in (e.get("licence") or {})]
    if not lics:
        return None
    n = Counter(lic["class"] for lic in lics)
    left_out = sum(1 for b, e in entries.items()
                   if b in papers and e.get("fulltext") and not indexable(e))
    return (f"  licences of the {len(lics)} full texts we hold: {n[OPEN]} open, "
            f"{n[CONDITIONS]} with conditions (NC/ND), {n[NO_LICENCE]} without a licence; "
            f"{left_out} left out of the index (a publisher's copy with no open licence); "
            f"see `make licences`")


def _words(md_dir: str, entry: dict) -> int:
    name = (entry.get("fulltext") or {}).get("markdown")
    path = os.path.join(md_dir, name) if name else None
    if not path or not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        return len(fh.read().split())


def report(entries: dict, papers: dict, md_dir: str, collection: str, log=print) -> None:
    full = {b: e for b, e in entries.items()
            if b in papers and "class" in (e.get("licence") or {})}
    if not full:
        log("No licences recorded yet: run `make licences` (or `make update`).")
        return
    if not any((e.get("indexed") or {}).get(collection) for e in full.values()):
        # Not written by update_corpus.py (yet): count in one that was.
        known = Counter(c for e in full.values() for c in e.get("indexed") or {})
        if known:
            collection = known.most_common(1)[0][0]
    words = {b: _words(md_dir, e) for b, e in full.items()}
    objects = {b: ((e.get("indexed") or {}).get(collection) or {}).get("objects", 0)
               for b, e in full.items()}
    total_w, total_o = sum(words.values()) or 1, sum(objects.values()) or 1

    def row(name: str, bibs: list[str], indent: str = "  ") -> None:
        w, o = sum(words[b] for b in bibs), sum(objects[b] for b in bibs)
        log(f"{indent}{name:44s} {len(bibs):3d} papers  {o:5d} objects ({o / total_o:4.0%})"
            f"  {w:7d} words ({w / total_w:4.0%})")

    log(f"Licences of the {len(full)} full texts we hold, for the copy we hold "
        f"(objects in '{collection}'):")
    for cls in CLASSES:
        row(cls, [b for b, e in full.items() if e["licence"]["class"] == cls])
    none = {b: e for b, e in full.items() if e["licence"]["class"] == NO_LICENCE}
    for name, test in (
            ("preprint, published version openly licensed",
             lambda lic: lic["copy"] == PREPRINT and lic["published_class"] != NO_LICENCE),
            ("preprint, published version not either",
             lambda lic: lic["copy"] == PREPRINT and lic["published_class"] == NO_LICENCE),
            ("published version or author's copy",
             lambda lic: lic["copy"] != PREPRINT)):
        row(name, [b for b, e in none.items() if test(e["licence"])], indent="      ")
    row("left out of the index (a publisher's copy)",
        [b for b, e in full.items() if not indexable(e)])

    log(f"\nThe {len(none)} without a licence:")
    for bib, e in sorted(none.items(), key=lambda kv: (kv[1]["licence"]["copy"], kv[0])):
        lic = e["licence"]
        log(f"  {bib:20s} {papers[bib].get('short_ref', '')[:24]:24s} {lic['copy']:17s} "
            f"{lic['copy_licence'][:34]:34s} published: {lic['published'][:34]:34s}"
            f" {lic.get('copyright', '')[:40]}")

    upgrades = [(b, e) for b, e in entries.items()
                if b in papers and e.get("state") != "full text"
                and (e.get("licence") or {}).get("published_class") in (OPEN, CONDITIONS)]
    if upgrades:
        log(f"\nWithout full text in the index, but openly licensed at the publisher "
            f"({len(upgrades)}):")
        for bib, e in sorted(upgrades):
            log(f"  {bib:20s} [{e['state']}] {e['licence']['published']:12s} {e['title'][:60]}")
    errors = [b for b, e in entries.items() if b in papers and (e.get("licence") or {}).get("errors")]
    if errors:
        log(f"\nLookups that failed last time, to be tried again: {', '.join(errors)}")


# ---------------------------------------------------------------------------

def main() -> int:
    from reindex_weaviate import DEFAULT_MD_DIR, DEFAULT_METADATA, load_papers
    from update_corpus import DATA_DIR, DEFAULT_COLLECTION, load_manifest

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bibcodes", nargs="*",
                    help="ask the sources about these papers and print it (writes nothing)")
    ap.add_argument("--report", action="store_true",
                    help="summarise the licences recorded in the manifest")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help=f"whose object counts the report shows (default: {DEFAULT_COLLECTION})")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--md-dir", default=DEFAULT_MD_DIR)
    ap.add_argument("--metadata", default=DEFAULT_METADATA)
    args = ap.parse_args()
    if not args.bibcodes and not args.report:
        ap.error("name some bibcodes, or --report")

    papers, aliases = load_papers(args.metadata)
    manifest = load_manifest(os.path.join(args.data_dir, "manifest.json"))
    if args.bibcodes:
        missing = [b for b in args.bibcodes if b not in aliases]
        if missing:
            ap.error(f"not on the list: {', '.join(missing)}")
        recs = {aliases[b]: papers[aliases[b]] for b in args.bibcodes}
        found, errors = look_up(recs)
        for bib in recs:
            entry = json.loads(json.dumps(manifest["papers"].get(bib) or {}))
            entry["licence"] = {"found": found[bib], "errors": errors[bib]}
            summarise(entry, recs[bib])
            print(f"\n{bib}\n" + json.dumps(entry["licence"], indent=1, ensure_ascii=False))
    if args.report:
        report(manifest["papers"], papers, args.md_dir, args.collection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
