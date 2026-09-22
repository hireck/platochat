#!/usr/bin/env python
"""Fetch the PLATO-Pub paper list and its full ADS metadata into papers.json.

Two sources, in this order:

1. The PLATO-Pub list itself. The list page (/about/ads_feed.php) is an empty
   shell that JavaScript fills from a JSON endpoint, ``ads_middleware.php``,
   so we read that endpoint rather than scraping HTML. It decides *which*
   papers belong to the corpus, but carries little about each one (title,
   authors, year, abstract, bibcode -- no date, DOI or arXiv id).

2. NASA ADS, asked once for every bibcode on the list ("bigquery"), which
   supplies everything else: publication date, identifiers, DOI, document
   type, keywords, and which full texts exist (``esources``).

The result replaces PLATOChat_papers.json (March 2024: no dates, lists stored
as strings, and by now out of step with the live list).

    python fetch_metadata.py              # writes ingest/papers.json
    python fetch_metadata.py --dry-run    # report only

Standard library only, apart from python-dotenv for the API key.

Bibcodes are not permanent: a preprint listed as 2024arXiv240104503D gets a
new bibcode when the journal version appears. ADS keeps the old ones as
aliases of the new record, and we store them all under ``aliases`` -- that is
how reindex_weaviate.py matches a markdown file converted under an old
bibcode to the paper's current record.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "papers.json")
DEFAULT_ENV = os.path.join(HERE, os.pardir, "platochat", ".env")

FEED_URL = "https://platopub.phys.au.dk/about/ads_middleware.php"
ADS_API = "https://api.adsabs.harvard.edu/v1/search"
ADS_ABS = "https://ui.adsabs.harvard.edu/abs/"

# Everything ADS holds about a paper that is of any conceivable use to us.
# Left out on purpose: `reference` and `citation` (hundreds of bibcodes per
# paper, and we have no use for a citation graph yet) and the full text
# (`body`), which ADS does not hand out through this API anyway.
ADS_FIELDS = [
    "bibcode", "alternate_bibcode", "identifier", "doi",
    "title", "alternate_title", "abstract", "keyword", "lang",
    "author", "first_author", "author_count", "aff", "orcid_pub",
    "year", "pubdate", "date", "entry_date",
    "pub", "pub_raw", "bibstem", "volume", "issue", "page", "page_range",
    "page_count", "doctype", "arxiv_class", "bibgroup", "database",
    "property", "esources", "data", "copyright", "comment", "pubnote",
    "citation_count", "read_count",
]

# A 19-character ADS bibcode: year, journal stem, volume/page, author initial.
_BIBCODE_RE = re.compile(r"^\d{4}[A-Za-z&.\d]{14}[A-Z.]$")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """ADS titles/abstracts as plain text.

    They arrive with HTML entities (``A&amp;A``) and publisher markup
    (``<SUB>2</SUB>``, ``<inline-formula><tex-math>$\\sim$</tex-math>...``).
    Dropping the tags keeps what is inside them, so the TeX survives.
    """
    if not text:
        return ""
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub("", text))).strip()


def _get_json(req: urllib.request.Request, timeout: int = 60) -> dict:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def fetch_feed() -> list[dict]:
    """Every paper on the live PLATO-Pub list, as the site serves it."""
    # rows=1000 is the site's own "Show all" request, and its hard cap. The
    # total is checked below so that outgrowing it cannot pass unnoticed.
    query = urllib.parse.urlencode(
        {"page": 1, "rows": 1000, "year": "", "author": "", "keywords": ""}
    )
    data = _get_json(urllib.request.Request(f"{FEED_URL}?{query}"))
    results = data.get("results") or []
    total = int(data.get("total") or len(results))
    if total > len(results):
        raise RuntimeError(
            f"The list reports {total} papers but returned {len(results)}: it "
            "has outgrown one page, and fetch_feed() needs to paginate."
        )
    return results


def ads_bigquery(bibcodes: list[str], key: str) -> list[dict]:
    """One ADS request for many bibcodes (the API's 'bigquery' endpoint)."""
    query = urllib.parse.urlencode(
        {"q": "*:*", "rows": 2000, "fl": ",".join(ADS_FIELDS)}
    )
    req = urllib.request.Request(
        f"{ADS_API}/bigquery?{query}",
        data=("bibcode\n" + "\n".join(bibcodes)).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "big-query/csv"},
        method="POST",
    )
    return _get_json(req)["response"]["docs"]


def ads_by_identifier(bibcode: str, key: str) -> dict | None:
    """Look one paper up by any identifier it has ever had.

    The fallback for bibcodes the bigquery did not return: ``identifier:``
    also matches superseded bibcodes, which an exact bibcode match does not.
    """
    query = urllib.parse.urlencode(
        {"q": f'identifier:"{bibcode}"', "rows": 1, "fl": ",".join(ADS_FIELDS)}
    )
    req = urllib.request.Request(
        f"{ADS_API}/query?{query}", headers={"Authorization": f"Bearer {key}"}
    )
    docs = _get_json(req)["response"]["docs"]
    return docs[0] if docs else None


def arxiv_id(identifiers: list[str]) -> str:
    for ident in identifiers:
        if ident.startswith("arXiv:"):
            return ident[len("arXiv:"):]
    return ""


def display_date(pubdate: str, year: str) -> str:
    """'2025-06' from ADS's '2025-06-00'; just the year if the month is 00.

    ADS pads unknown parts with 00 and rarely knows the day, so month is the
    finest precision worth showing.
    """
    m = re.match(r"^(\d{4})-(\d{2})", pubdate or "")
    if not m:
        return year or ""
    return m.group(1) if m.group(2) == "00" else f"{m.group(1)}-{m.group(2)}"


def short_ref(authors: list[str], year: str) -> str:
    """'Rauer et al. (2025)' -- how astronomers name a paper in running text."""
    surnames = [a.split(",")[0].strip() for a in authors if a.strip()]
    if not surnames:
        who = "Unknown"
    elif len(surnames) == 1:
        who = surnames[0]
    elif len(surnames) == 2:
        who = f"{surnames[0]} & {surnames[1]}"
    else:
        who = f"{surnames[0]} et al."
    return f"{who} ({year})" if year else who


def build_record(doc: dict, feed_bibcode: str) -> dict:
    """One papers.json entry: the ADS fields as given, plus a few derived ones."""
    rec = {k: v for k, v in doc.items() if v not in (None, "", [])}
    # ADS wraps the title in a one-element list.
    rec["title"] = clean_text(" ".join(doc.get("title") or []))
    rec["abstract"] = clean_text(doc.get("abstract"))

    identifiers = doc.get("identifier") or []
    year = str(doc.get("year") or "")
    aliases = {doc["bibcode"], feed_bibcode, *(doc.get("alternate_bibcode") or [])}
    aliases.update(i for i in identifiers if _BIBCODE_RE.match(i))

    rec["arxiv_id"] = arxiv_id(identifiers)
    rec["pubdate_display"] = display_date(doc.get("pubdate", ""), year)
    rec["short_ref"] = short_ref(doc.get("author") or [], year)
    rec["ads_url"] = ADS_ABS + urllib.parse.quote(doc["bibcode"], safe="")
    rec["aliases"] = sorted(aliases)
    rec["feed_bibcode"] = feed_bibcode
    return rec


def load_key(env_path: str) -> str:
    key = os.environ.get("ADS_API_KEY")
    if not key and os.path.exists(env_path):
        from dotenv import dotenv_values
        key = dotenv_values(env_path).get("ADS_API_KEY")
    if not key:
        raise SystemExit(f"No ADS_API_KEY in the environment or in {env_path}")
    return key


def fetch_papers(key: str) -> dict[str, dict]:
    """Every paper on the live list, with its ADS record, keyed by current bibcode."""
    print(f"Reading the PLATO-Pub list ({FEED_URL}) …")
    feed = fetch_feed()
    feed_bibcodes = [p["bibcode"] for p in feed if p.get("bibcode")]
    print(f"  {len(feed_bibcodes)} papers on the list")

    print("Asking ADS for their metadata …")
    by_bibcode = {d["bibcode"]: d for d in ads_bigquery(feed_bibcodes, key)}

    papers: dict[str, dict] = {}
    unresolved: list[str] = []
    for fb in feed_bibcodes:
        doc = by_bibcode.get(fb) or ads_by_identifier(fb, key)
        if doc is None:
            unresolved.append(fb)
            continue
        if doc["bibcode"] != fb:
            print(f"  {fb} is now {doc['bibcode']}")
        papers[doc["bibcode"]] = build_record(doc, fb)

    # A listed paper ADS cannot find still belongs to the corpus: keep what
    # the list itself told us, so it is at least findable by title.
    for p in feed:
        if p.get("bibcode") in unresolved:
            year = str(p.get("year") or "")
            papers[p["bibcode"]] = {
                "bibcode": p["bibcode"], "title": clean_text(p.get("title")),
                "abstract": clean_text(p.get("abstract")), "year": year,
                "pubdate_display": year, "short_ref": year and f"({year})",
                "ads_url": p.get("link") or ADS_ABS + p["bibcode"],
                "aliases": [p["bibcode"]], "feed_bibcode": p["bibcode"],
                "arxiv_id": "", "metadata_source": "plato-pub feed only",
            }
    if unresolved:
        print(f"  {len(unresolved)} unknown to ADS: {', '.join(unresolved)}")
    return papers


def summarise(papers: dict[str, dict]) -> None:
    recs = list(papers.values())
    print(f"\n{len(recs)} papers")
    print("  by type:        ", dict(Counter(r.get("doctype", "?") for r in recs).most_common()))
    print("  with a pubdate: ", sum(1 for r in recs if r.get("pubdate")))
    print("  with an arXiv id:", sum(1 for r in recs if r.get("arxiv_id")))
    print("  open access:    ", sum(1 for r in recs if "OPENACCESS" in r.get("property", [])))
    print("  by year:        ", dict(sorted(Counter(r.get("year", "?") for r in recs).items())))


def write_papers(papers: dict[str, dict], path: str) -> None:
    out = {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "list_url": FEED_URL,
        "n_papers": len(papers),
        "papers": dict(sorted(papers.items())),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)   # never leave a half-written papers.json behind


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--env", default=DEFAULT_ENV, help="file holding ADS_API_KEY")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    papers = fetch_papers(load_key(args.env))
    summarise(papers)
    if args.dry_run:
        print("\nDry run — nothing written.")
        return 0
    write_papers(papers, args.out)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
