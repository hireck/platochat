#!/usr/bin/env python
"""The ESA PLATO website, indexed: practical information the papers do not give.

The papers say what PLATO is and what it will measure; they say little about
how to *use* it -- how to apply for observing time as a Guest Observer, what
the current call asks for and when it closes, which proposal tools and
templates there are, where the data will be and in what form. That lives on
ESA's mission pages at https://www.cosmos.esa.int/web/plato/ , and this script
puts it into a Weaviate collection of its own, ``PLATO_ESA_SITE``
(``PLATO_SITE_COLLECTION`` overrides), for the chatbot's
``search_esa_website`` tool.

    python esa_site.py              # crawl and bring the collection up to date
    python esa_site.py --dry-run    # crawl and say what would change
    python esa_site.py --save DIR   # also keep each page's markdown in DIR

What it does, and why:

* Crawls every page under /web/plato/, starting from the home page, one
  request a second with an honest User-Agent. The site has no robots.txt
  rules. About 55 pages, a minute.
* Takes the page text from Liferay's article blocks (``journal-content-
  article``) and the news cards from the news page, and converts them to
  markdown with pandoc (already needed for LaTeXML). Menus, cookie banners
  and the site chrome are not in those blocks.
* Drops boilerplate: a block found on more than half of the pages (the
  "contact our PLATO Helpdesk" footer) is left out -- but first the
  site-wide "This website was last updated on ..." date is read from it.
* Keeps one copy of pages served under several addresses (plato-arxiv,
  plato-archive and plato-arxive are the same page): the first one found,
  which is the one the home page links to.
* Chunks and embeds like the papers (textsplitter, bge-m3), the page title
  standing in for the paper title. Links stay in the stored text, made
  absolute, because they are often the answer ("the templates are here");
  the embedder sees only their text.
* Rewrites a page only when its fingerprint changes, and deletes pages that
  vanished -- unless the crawl lost more than half of them at once, which is
  likelier a broken site than a redesign.

The linked documents (the AO-1 call, the Policies & Procedures, the Mission
Handbook, ...) are not indexed: the pages link to them, and the model can
pass those links on.

Every object carries ``checked``, the date this run confirmed its text, and
``site_updated``, the site's own last-updated date. Both are refreshed on
every run without re-embedding, so they stay out of the fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import date, datetime
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reindex_weaviate import (  # noqa: E402
    DEFAULT_MODEL, HEADERS, ChunkSettings, delete_objects, embed, embedding_text,
    load_embedder, load_tokenizer_counter,
)
from textsplitter import split_markdown_packed  # noqa: E402

SITE = "https://www.cosmos.esa.int"
ROOT = SITE + "/web/plato/"
HOME = ROOT + "home"
DEFAULT_COLLECTION = os.environ.get("PLATO_SITE_COLLECTION", "PLATO_ESA_SITE")
USER_AGENT = "PLATOChat indexer (Aarhus University; https://platopub.phys.au.dk)"
PAUSE = 1.0                  # seconds between requests
MAX_PAGES = 300              # a runaway crawl stops here
# Addresses under /web/plato/ that are not pages worth indexing.
SKIP = {"logout", "old-pages"}

# Bump when the objects change in a way the fingerprint does not see.
SITE_INDEX_VERSION = 1

# A block on more than this share of the pages is site furniture.
FURNITURE_SHARE = 0.5
# Refuse to delete more than this share of the indexed pages in one run.
REMOVAL_SHARE = 0.5

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
         "source", "track", "wbr"}
_TITLE_TAIL_RE = re.compile(r"\s*-\s*PLATO\s*-\s*Cosmos\s*$", re.I)
_UPDATED_RE = re.compile(r"last updated on\s+(\d{1,2}\s+\w+\s+\d{4})", re.I)
_DATE_FORMATS = ("%d %B %Y", "%d %b %Y")


# ---------------------------------------------------------------------------
# Fetching and extracting
# ---------------------------------------------------------------------------

def normalise(url: str) -> str:
    """One spelling per page: https, no query or fragment, no trailing slash."""
    url = urldefrag(url)[0].split("?")[0].rstrip("/")
    url = url.replace("http://", "https://", 1)
    return HOME if url == ROOT.rstrip("/") else url


def slug(url: str) -> str:
    return url[len(ROOT):] if url.startswith(ROOT) else url


def is_page(url: str) -> bool:
    path = urlparse(url).path
    return (urlparse(url).netloc == urlparse(SITE).netloc and path.startswith("/web/plato/")
            and "/-/" not in path and slug(url).split("/")[0] not in SKIP)


class _PageParser(HTMLParser):
    """The article blocks of a Liferay page, as raw HTML, plus its title and news cards."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.blocks: list[str] = []
        self.links: list[str] = []
        self.title = ""
        self.news: list[dict] = []
        self._buf: list[str] | None = None
        self._depth = 0
        self._in_title = False
        self._card: dict | None = None
        self._card_field: str | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])
        if tag == "meta" and a.get("property") == "og:title":
            self.title = a.get("content") or self.title
        if tag == "title" and not self.title:
            self._in_title = True
        # News cards (the asset publisher on the news page): a link wrapping a
        # title and a date.
        cls = a.get("class") or ""
        if tag == "a" and self._buf is None and a.get("title") and a.get("href"):
            self._card = {"url": a["href"], "title": unescape(a["title"]), "date": ""}
        elif self._card is not None and tag == "span" and "inner-date" in cls:
            self._card_field = "date"

        if self._buf is not None:
            self._buf.append(self.get_starttag_text())
            if tag not in _VOID:
                self._depth += 1
        elif tag == "div" and "journal-content-article" in cls.split():
            self._buf, self._depth = [], 1

    def handle_startendtag(self, tag, attrs):
        if self._buf is not None:
            self._buf.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._card is not None:
            if self._card["date"]:
                self.news.append(self._card)
            self._card = None
        if tag == "span":
            self._card_field = None
        if self._buf is None or tag in _VOID:
            return
        self._depth -= 1
        if self._depth == 0:
            self.blocks.append("".join(self._buf))
            self._buf = None
        else:
            self._buf.append(f"</{tag}>")

    def handle_data(self, data):
        if self._in_title and not self.title:
            self.title = data
        if self._card_field and self._card is not None:
            self._card[self._card_field] += data.strip()
        if self._buf is not None:
            self._buf.append(data)

    def handle_entityref(self, name):
        if self._buf is not None:
            self._buf.append(f"&{name};")

    def handle_charref(self, name):
        if self._buf is not None:
            self._buf.append(f"&#{name};")


def parse_page(html: str, url: str) -> dict:
    p = _PageParser()
    p.feed(html)
    title = _TITLE_TAIL_RE.sub("", unescape(p.title or "")).strip()
    title = re.sub(r"^COSMOS\s+", "", title) or slug(url)
    # Liferay stamps document links with a cache-busting ?t=...; without it
    # the same block reads the same on every page and every day.
    blocks = [re.sub(r"\?t=\d+", "", b) for b in p.blocks]
    links = [normalise(urljoin(url, unescape(h))) for h in p.links]
    return {"url": url, "title": title, "blocks": blocks, "links": links, "news": p.news}


def crawl(session=None, log=print, pause: float = PAUSE) -> tuple[list[dict], list[tuple]]:
    """Every page under /web/plato/, in the order found from the home page.

    Returns ``(pages, failures)``, a failure being ``(slug, reason)``. The home
    page failing raises: without it there is nothing to go on.
    """
    import requests

    session = session or requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    queue, seen, pages, failures = [HOME], set(), [], []
    while queue and len(seen) < MAX_PAGES:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        if pages or failures:
            time.sleep(pause)
        try:
            r = session.get(url, timeout=30)
        except Exception as exc:
            if url == HOME:
                raise
            failures.append((slug(url), type(exc).__name__))
            continue
        if r.status_code != 200 or "html" not in r.headers.get("content-type", "html"):
            if url == HOME:
                raise RuntimeError(f"the home page answered {r.status_code}")
            failures.append((slug(url), f"HTTP {r.status_code}"))
            continue
        page = parse_page(r.text, url)
        pages.append(page)
        for link in page["links"]:
            if is_page(link) and link not in seen and link not in queue:
                queue.append(link)
    log(f"  crawled {len(pages)} pages" + (f", {len(failures)} failed" if failures else ""))
    return pages, failures


# ---------------------------------------------------------------------------
# From pages to markdown
# ---------------------------------------------------------------------------

def site_date(text: str) -> str:
    """'This website was last updated on 14 September 2026' -> '2026-09-14', else ''."""
    m = _UPDATED_RE.search(text)
    if not m:
        return ""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(m.group(1), fmt).date().isoformat()
        except ValueError:
            pass
    return ""


def to_markdown(html: str) -> str:
    out = subprocess.run(
        ["pandoc", "-f", "html", "-t", "gfm-raw_html", "--wrap=none"],
        input=html, capture_output=True, text=True, check=True, timeout=120,
    ).stdout
    return tidy_markdown(out)


_EMPTY_LINK_RE = re.compile(r"(?<!!)\[\]\([^)]*\)")
_IMAGE_RE = re.compile(r"!\[(?:\\.|[^\]\\])*\]\([^)]*\)")
_REL_LINK_RE = re.compile(r"\]\((/[^)\s]*)")
_LINK_RE = re.compile(r"\[((?:\\.|[^\]\\])*)\]\([^)]*\)")


def tidy_markdown(md: str) -> str:
    md = _IMAGE_RE.sub("", md)               # figures are not indexed
    md = _EMPTY_LINK_RE.sub("", md)          # Liferay leaves empty anchors beside links
    md = _REL_LINK_RE.sub(lambda m: "](" + SITE + m.group(1), md)
    md = md.replace(" ", " ").replace(" ", " ")
    md = re.sub(r"^\s*-{3,}\s*$", "", md, flags=re.M)     # rules between blocks
    md = re.sub(r"[ \t]+$", "", md, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def link_text(md: str) -> str:
    """Links reduced to their text, for the embedder: URLs are noise to it."""
    return _LINK_RE.sub(r"\1", md)


def news_markdown(news: list[dict]) -> str:
    lines = [f"- {n['date']}: [{n['title']}]({n['url']})" for n in news]
    return "## News items\n\n" + "\n".join(lines) if lines else ""


def build_pages(crawled: list[dict], log=print) -> tuple[list[dict], str]:
    """Pages as markdown, boilerplate out, duplicates merged. Returns ``(pages, site_updated)``."""
    seen_on = Counter(b for p in crawled for b in set(p["blocks"]))
    limit = FURNITURE_SHARE * len(crawled)
    furniture = {b for b, n in seen_on.items() if n > limit}
    updated = ""
    for b in furniture:
        updated = updated or site_date(to_markdown(b))

    out, by_text = [], {}
    for p in crawled:
        parts = [to_markdown(b) for b in p["blocks"] if b not in furniture]
        # The news page's cards are not article blocks; the internal pages
        # they point to are crawled anyway, so list them all.
        if p["news"]:
            parts.append(news_markdown(p["news"]))
        md = "\n\n".join(x for x in parts if x.strip())
        if len(md.split()) < 5:
            continue
        key = hashlib.sha256(md.encode("utf-8")).hexdigest()
        if key in by_text:
            by_text[key]["aliases"].append(p["url"])
            continue
        page = {"url": p["url"], "slug": slug(p["url"]), "title": p["title"],
                "markdown": md, "aliases": []}
        by_text[key] = page
        out.append(page)
    merged = sum(len(p["aliases"]) for p in out)
    log(f"  {len(out)} pages with text"
        + (f" ({merged} duplicate address(es) merged)" if merged else "")
        + (f"; {len(furniture)} boilerplate block(s) dropped" if furniture else "")
        + (f"; site last updated {updated}" if updated else ""))
    return out, updated


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------

def chunk_id_for(page_slug: str, n: int) -> str:
    return f"esa:{page_slug}#{n:04d}"


def _without_title(headers: list[str], title: str) -> list[str]:
    """Most pages open with their title as a heading; the title is shown anyway."""
    if headers and headers[0].strip().casefold() == title.strip().casefold():
        return headers[1:]
    return headers


def page_fingerprint(page: dict, settings: ChunkSettings) -> str:
    basis = {"markdown": page["markdown"], "title": page["title"], "url": page["url"],
             "chunks": settings.signature(), "version": SITE_INDEX_VERSION}
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def page_objects(page: dict, settings: ChunkSettings, count_tokens) -> list[dict]:
    pieces = split_markdown_packed(
        page["markdown"],
        headers_to_split_on=HEADERS,
        max_tokens=settings.max_tokens,
        overlap_tokens=settings.overlap,
        min_tokens=settings.min_tokens,
        atomic_max_tokens=settings.atomic_max,
        drop_sections=None,
        count_tokens=count_tokens,
    )
    out = []
    for n, piece in enumerate(pieces, start=1):
        headers = _without_title(piece.headers, page["title"])
        out.append({
            "properties": {
                "chunk_id":        chunk_id_for(page["slug"], n),
                "chunk_index":     n,
                "page":            page["slug"],
                "title":           page["title"],
                "url":             page["url"],
                "section_headers": headers,
                "page_content":    piece.content,
            },
            "embed": embedding_text("ESA PLATO website: " + page["title"],
                                    " > ".join(headers), link_text(piece.content)),
        })
    return out


def object_uuid(chunk_id: str) -> str:
    from weaviate.util import generate_uuid5
    return str(generate_uuid5(chunk_id))


# ---------------------------------------------------------------------------
# Weaviate
# ---------------------------------------------------------------------------

def _schema():
    from weaviate.classes.config import DataType, Property, Tokenization

    def exact(name, data_type=DataType.TEXT, filterable=True):
        return Property(name=name, data_type=data_type, tokenization=Tokenization.FIELD,
                        index_searchable=False, index_filterable=filterable)

    return [
        exact("chunk_id"),
        Property(name="chunk_index", data_type=DataType.INT),
        exact("page"),
        # What BM25 searches (SITE_BM25_PROPERTIES in plato_core.py).
        Property(name="title", data_type=DataType.TEXT),
        Property(name="section_headers", data_type=DataType.TEXT_ARRAY),
        Property(name="page_content", data_type=DataType.TEXT),
        exact("url", filterable=False),
        exact("checked", filterable=False),
        exact("site_updated", filterable=False),
        exact("fingerprint", filterable=False),
    ]


def ensure_site_collection(client, name: str):
    from weaviate.classes.config import Configure, VectorDistances

    if not client.collections.exists(name):
        print(f"  creating collection '{name}'")
        client.collections.create(
            name,
            properties=_schema(),
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE),
        )
    return client.collections.get(name)


def site_state(collection) -> dict[str, dict[str, dict]]:
    """``{page slug: {uuid: {fingerprint, checked, site_updated}}}`` as the collection has it."""
    out: dict[str, dict[str, dict]] = {}
    for obj in collection.iterator(return_properties=["page", "fingerprint", "checked",
                                                      "site_updated"]):
        p = obj.properties
        out.setdefault(p.get("page") or "", {})[str(obj.uuid)] = {
            "fingerprint": p.get("fingerprint"), "checked": p.get("checked"),
            "site_updated": p.get("site_updated")}
    return out


def fetch_site(save_dir: str | None = None, model: str = DEFAULT_MODEL, session=None,
               log=print) -> dict:
    """Crawl the site and turn it into markdown pages and their objects.

    Separate from :func:`sync_site`, and to be run before connecting to
    Weaviate: pandoc runs in a subprocess, and forking while the client's gRPC
    channel is open fills the log with gRPC's fork warnings.
    """
    crawled, failures = crawl(session, log=log)
    for page_slug, reason in failures:
        log(f"    ! {page_slug}: {reason}")
    pages, updated = build_pages(crawled, log=log)
    settings = ChunkSettings(model=model)
    count_tokens = load_tokenizer_counter(model)
    for p in pages:
        p["fingerprint"] = page_fingerprint(p, settings)
        p["objects"] = page_objects(p, settings, count_tokens)
    # A page too short to make a passage of (the registration page, a line
    # long) would otherwise count as new on every run.
    pages = [p for p in pages if p["objects"]]
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for p in pages:
            with open(os.path.join(save_dir, p["slug"].replace("/", "_") + ".md"), "w",
                      encoding="utf-8") as fh:
                fh.write(f"<!-- {p['url']} -->\n\n{p['markdown']}\n")
    return {"pages": pages, "site_updated": updated, "failures": failures}


def sync_site(client, site: dict, name: str = DEFAULT_COLLECTION, *, model: str = DEFAULT_MODEL,
              device: str | None = None, embedder=None, dry_run: bool = False,
              allow_removals: bool = False, log=print) -> dict:
    """Make the collection hold exactly the pages :func:`fetch_site` found, up to date."""
    pages, updated, failures = site["pages"], site["site_updated"], site["failures"]
    today = date.today().isoformat()

    if dry_run and not client.collections.exists(name):
        state = {}
        log(f"  collection '{name}' does not exist yet; it would be created")
    else:
        collection = client.collections.get(name) if dry_run else ensure_site_collection(client, name)
        state = site_state(collection)

    todo = []
    for p in pages:
        old = state.get(p["slug"], {})
        if not old or {o["fingerprint"] for o in old.values()} != {p["fingerprint"]}:
            todo.append((p, p["fingerprint"], old))
    # A page that failed to load this time keeps what it had.
    live = {p["slug"] for p in pages} | {page_slug for page_slug, _ in failures}
    stale = {s: objs for s, objs in state.items() if s not in live}
    stats = {"pages": len(pages), "written": len(todo), "removed": 0, "failed": len(failures)}
    log(f"  '{name}': {len(pages)} pages, {len(todo)} to write, {len(stale)} to remove")
    for p, _, old in todo:
        log(f"    {p['slug']}{'' if old else '  (new)'}")
    for s in stale:
        log(f"    remove {s}")
    if dry_run:
        return stats

    if todo:
        embedder = embedder or load_embedder(model, device)
    for p, fp, old in todo:
        objects = p["objects"]
        vectors = embed(embedder, objects)
        uuids = []
        with collection.batch.fixed_size(batch_size=64) as b:
            for obj, vector in zip(objects, vectors):
                props = {**obj["properties"], "fingerprint": fp, "checked": today,
                         "site_updated": updated}
                uuid = object_uuid(props["chunk_id"])
                uuids.append(uuid)
                b.add_object(properties=props, vector=vector.tolist(), uuid=uuid)
        failed = collection.batch.failed_objects
        if failed:
            raise RuntimeError(f"{len(failed)} object(s) of {p['slug']} failed: {failed[0].message}")
        # New objects first, leftovers after: the page is never missing.
        leftover = set(old) - set(uuids)
        if leftover:
            delete_objects(collection, leftover)

    if stale:
        if not allow_removals and len(stale) > REMOVAL_SHARE * max(len(state), 1):
            log(f"  ! {len(stale)} of {len(state)} indexed pages were not found; refusing to "
                f"remove that many at once (--allow-removals)")
        else:
            for s, objs in stale.items():
                delete_objects(collection, set(objs))
                stats["removed"] += 1

    # The dates change far more often than the text: patch them in place,
    # without re-embedding.
    written = {p["slug"] for p, _, _ in todo}
    for s, objs in state.items():
        if s in written or s in stale:
            continue
        for uuid, o in objs.items():
            if o["checked"] != today or o["site_updated"] != updated:
                collection.data.update(uuid=uuid, properties={"checked": today,
                                                              "site_updated": updated})
    log(f"  '{name}' now holds {len(collection)} objects")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help="default: $PLATO_SITE_COLLECTION, else PLATO_ESA_SITE")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=os.environ.get("PLATO_EMBED_DEVICE"))
    ap.add_argument("--dry-run", action="store_true", help="crawl, say what would change, write nothing")
    ap.add_argument("--save", metavar="DIR", help="also write each page's markdown to DIR")
    ap.add_argument("--allow-removals", action="store_true",
                    help="remove vanished pages even when more than half of them went at once")
    args = ap.parse_args()

    import weaviate

    site = fetch_site(args.save, args.model)
    client = weaviate.connect_to_local()
    try:
        if not client.is_ready():
            print("Weaviate is not ready.", file=sys.stderr)
            return 1
        sync_site(client, site, args.collection, model=args.model, device=args.device,
                  dry_run=args.dry_run, allow_removals=args.allow_removals)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
