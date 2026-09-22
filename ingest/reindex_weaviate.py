#!/usr/bin/env python
"""Rebuild the Weaviate PLATO collection from the markdown corpus.

This is the from-scratch rebuild. The everyday path is update_corpus.py, which
changes one paper at a time with the functions defined here -- so a rebuild and
a series of updates produce the same objects, and after a rebuild the next
update finds nothing to do.

What goes in, per paper on the PLATO-Pub list (papers.json):

* **Full text**, header-split *and* paragraph-packed (see ``textsplitter.py``),
  when there is a markdown file for the paper and its licence allows it (see
  ``fetch_fulltext.fulltext_allowed``). Header splitting alone made one paper
  section the retrieval unit -- 890 tokens at the median, up to 70k -- which no
  embedding model can see whole.
* **Its abstract alone**, from ADS, for every other paper -- paywalled ones,
  and open ones whose full text we could not get yet -- so that every paper on
  the list can be found. Such objects carry ``coverage = "abstract only"``, and
  the chatbot tells the model so. A full text whose markdown lost the abstract
  (it happens in PDF conversion) gets the ADS abstract as well.

Vectors come from bge-m3 (1024 dims, 8192-token window). The rebuild drops
the collection first, which is why ``--yes`` is required to touch an existing
one. Run ``--dry-run`` first to see the chunk statistics without connecting
to Weaviate at all.

    python reindex_weaviate.py --dry-run
    python reindex_weaviate.py --collection PLATO_TEST      # new collection
    python reindex_weaviate.py --yes                        # rebuild PLATO

Keep the model here in step with ``PLATO_EMBED_MODEL`` in plato_core.py: a
query embedded by one model cannot be compared against a corpus embedded by
another.

Paper metadata comes from papers.json (written by fetch_metadata.py), which
mirrors the live PLATO-Pub list. Consequences:

* Every object carries its paper's publication date, so the chatbot can show
  it and prefer recent sources.
* A markdown file whose paper is not on the list (any more) is left out of the
  index. A file named after an *old* bibcode of a listed paper -- a preprint
  that has since been published -- is matched through the record's aliases.
* A markdown file that does not contain its own paper's title is left out and
  reported: it means the conversion picked up the wrong file (this happened --
  an arXiv bundle's copy of the MNRAS author guide got converted in place of
  the paper).

Every object also carries a ``fingerprint``: a hash of everything that went
into it (the markdown, the paper's metadata, the chunking settings and
INDEX_VERSION). update_corpus.py compares fingerprints to find the papers
whose objects are out of date.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textsplitter import DROP_SECTIONS, split_markdown_packed  # noqa: E402
from latexml_to_markdown import strip_false_dates  # noqa: E402
from fetch_fulltext import fulltext_allowed, title_share  # noqa: E402

DEFAULT_MD_DIR   = os.environ.get(
    "PLATO_MD_DIR", "/Users/hilke/data/plato_data/plato_markdown"
)
DEFAULT_METADATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "papers.json"
)
DEFAULT_MODEL    = os.environ.get("PLATO_EMBED_MODEL", "BAAI/bge-m3")

# Bump this when the chunking or the objects change in a way ChunkSettings does
# not capture -- a fix in textsplitter.py, a new property, a different
# embedding prefix. Every fingerprint changes with it, so the next update
# re-indexes every paper (about ten minutes on the Mac's GPU for 180 papers;
# far longer on a CPU-only server).
INDEX_VERSION = 1

# Header levels to split on: all six. Papers converted from LaTeX/PDF nest to
# #####, and LaTeXML writes the unnumbered front- and back-matter blocks as
# ###### -- "Abstract", "Key Words.:", "Acknowledgements." (the only level-6
# headings in the corpus, checked 2026-09-21). Left unsplit, the abstract had no
# section name for the sources line to show, and the acknowledgements rode
# along inside the last section's chunks instead of being dropped.
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"), ("#####", "h5"),
           ("######", "h6")]

# A markdown file must contain at least this share of its paper's title words
# near the top, or it is taken for the wrong document. Measured on the corpus:
# real papers score 0.92-1.0, the wrongly converted author guide scored 0.0.
TITLE_CHECK_MIN = 0.5
TITLE_CHECK_CHARS = 8000   # A&A papers open with a long affiliation block
# ...and the first few headings wherever they are: the PLATO mission paper
# (Rauer et al. 2025) opens with 26,000 characters of consortium affiliations
# before its "# The PLATO Mission". A wrong file's first heading is its own
# title, so this does not let one through.
TITLE_CHECK_HEADINGS = 3
_HEADING_LINE_RE = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+)$", re.M)
# Below this many words a "paper" is a failed conversion (a scan with no text
# layer, a poster that came out as its title) rather than a short abstract.
MIN_WORDS = 100

# A full text counts as containing its abstract if this share of the ADS
# abstract's five-word sequences occurs near its top. Otherwise the ADS
# abstract is indexed alongside, so that no paper loses the most quotable
# summary of it. Sequences rather than single words because the introduction
# repeats the abstract's vocabulary: measured on the 27 papers of 2026-09-21,
# word overlap gave 0.93-1.00 with the abstract and still 0.67-0.84 with it cut
# out; five-word sequences gave 0.64-1.00 against 0.01-0.21.
ABSTRACT_CHECK_MIN = 0.4
ABSTRACT_CHECK_CHARS = 30000
_SHINGLE = 5

# A date standing alone in a paper's front matter that is this much later than
# the paper's publication is not the paper's: it is LaTeXML's \today, i.e. the
# day we converted it. The converter strips these now; this catches markdown
# converted before it did. The slack is there because ADS dates are only good
# to the month -- and errs harmlessly either way: a false date that survives is
# within a month of the true one, a true date that goes was redundant with
# `pubdate`.
FALSE_DATE_SLACK = timedelta(days=31)

FULL_TEXT = "full text"
ABSTRACT_ONLY = "abstract only"


@dataclass(frozen=True)
class ChunkSettings:
    """Everything about chunking and embedding that changes the objects."""

    max_tokens: int = 512
    overlap: int = 64
    min_tokens: int = 48
    # Tables are kept whole rather than word-split, so their ceiling is the
    # embedder's window, not the chunk budget -- past it the model truncates
    # anyway, so that is the point at which splitting starts to be the lesser
    # evil. 8192 is bge-m3's; lower it if you switch to a shorter-context model.
    atomic_max: int = 8192
    keep_references: bool = False
    model: str = DEFAULT_MODEL

    @property
    def drop_sections(self):
        return None if self.keep_references else DROP_SECTIONS

    def signature(self) -> dict:
        return {
            "version": INDEX_VERSION, "headers": len(HEADERS),
            "max_tokens": self.max_tokens, "overlap": self.overlap,
            "min_tokens": self.min_tokens, "atomic_max": self.atomic_max,
            "drop": None if self.keep_references else DROP_SECTIONS.pattern,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# Papers and their markdown
# ---------------------------------------------------------------------------

def chunk_id_for(bibcode: str, n: int) -> str:
    """'2021A&A...653A..98M#0017' -- the 17th chunk of that paper.

    Built from the paper and the position within it, never from a counter
    running over the whole corpus: adding or removing one paper must not
    renumber the others. The id of a chunk changes only if its own paper's
    markdown or the chunking parameters change.
    """
    return f"{bibcode}#{n:04d}"


def abstract_id_for(bibcode: str) -> str:
    return f"{bibcode}#abstract"


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def _shingles(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)}


def abstract_share(abstract: str, markdown: str) -> float:
    """Share of the ADS abstract's five-word sequences found near the top of the markdown."""
    wanted = _shingles(_words(abstract))
    if not wanted:
        return 1.0
    return len(wanted & _shingles(_words(markdown[:ABSTRACT_CHECK_CHARS]))) / len(wanted)


def markdown_problem(markdown: str, meta: dict) -> str | None:
    """Why this markdown cannot be indexed as the paper's full text, or None."""
    headings = [m.group(1) for _, m in zip(range(TITLE_CHECK_HEADINGS),
                                            _HEADING_LINE_RE.finditer(markdown))]
    share = title_share(meta.get("title") or "",
                        markdown[:TITLE_CHECK_CHARS] + "\n" + "\n".join(headings))
    if share < TITLE_CHECK_MIN:
        return (f"its paper's title is not in it ({share:.0%} of the title words) -- "
                "probably the wrong file was converted")
    n_words = len(markdown.split())
    if n_words < MIN_WORDS:
        return f"only {n_words} words -- the conversion found no text"
    return None


def load_papers(metadata_path: str) -> tuple[dict, dict]:
    """papers.json as ``(records by bibcode, any known bibcode -> current one)``."""
    with open(metadata_path, encoding="utf-8") as fh:
        papers = json.load(fh)["papers"]
    aliases = {a: bib for bib, rec in papers.items() for a in rec.get("aliases", [bib])}
    return papers, aliases


def match_markdown(md_dir: str, aliases: dict) -> tuple[dict, list[str]]:
    """``({bibcode: markdown path}, [files whose paper is not on the list])``.

    A file is named after the bibcode the paper had when it was converted,
    which for a preprint since published is an alias of the current one. If
    two files belong to the same paper, the one named after the current
    bibcode wins, else the first by name.
    """
    found: dict[str, str] = {}
    unlisted: list[str] = []
    for fn in sorted(f for f in os.listdir(md_dir) if f.endswith(".md")):
        doc_id = fn[:-3]
        bibcode = aliases.get(doc_id)
        if bibcode is None:
            unlisted.append(doc_id)
            continue
        if bibcode in found:
            kept = os.path.basename(found[bibcode])[:-3]
            if doc_id != bibcode:
                print(f"  ! {doc_id} and {kept} are the same paper ({bibcode}); keeping {kept}")
                continue
            print(f"  ! {doc_id} and {kept} are the same paper; keeping {doc_id}")
        found[bibcode] = os.path.join(md_dir, fn)
    return found, unlisted


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Objects for one paper
# ---------------------------------------------------------------------------

def paper_props(meta: dict, bibcode: str) -> dict:
    """The properties every object of a paper carries."""
    return {
        "title":      meta.get("title") or "",
        "authors":    "; ".join(meta.get("author") or []),
        "short_ref":  meta.get("short_ref") or "",
        "journal":    meta.get("pub") or "",
        "year":       int(meta["year"]) if str(meta.get("year") or "").isdigit() else None,
        # Two forms of the same date: `pubdate` is what people read
        # ("2024-06", or just "2024" when ADS knows no month), `pub_date`
        # is a real date that Weaviate can filter and sort on.
        "pubdate":    meta.get("pubdate_display") or "",
        "pub_date":   (datetime.fromisoformat(meta["date"].replace("Z", "+00:00"))
                       if meta.get("date") else None),
        "parent_doc": bibcode,
        "link":       meta.get("ads_url") or "",
        "doi":        (meta.get("doi") or [""])[0],
        "arxiv_id":   meta.get("arxiv_id") or "",
    }


def fingerprint(meta: dict, bibcode: str, markdown_sha: str | None,
                settings: ChunkSettings) -> str:
    """A hash of everything the paper's objects are made from.

    ``markdown_sha`` is None for an abstract-only paper. Computed without
    chunking or embedding anything, so checking a whole corpus is cheap.
    """
    basis = {
        "props": {k: str(v) for k, v in paper_props(meta, bibcode).items()},
        "abstract": meta.get("abstract") or "",
        "markdown": markdown_sha,
        "settings": settings.signature(),
    }
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def embedding_text(title: str, header_path: str, content: str) -> str:
    """The text handed to the embedder — not the same as what we store.

    A chunk five paragraphs into "3.2 Noise budget" has no idea, on its own,
    which paper or section it belongs to; two such chunks from different papers
    can be near-identical in isolation. Prefixing the title and header path
    anchors the vector to its context. ``page_content`` keeps the clean text,
    so the prefix never reaches the answering model or the citation display.
    """
    return "\n".join(p for p in (title, header_path, "", content) if p is not None)


def abstract_object(meta: dict, bibcode: str, coverage: str) -> dict:
    """The paper's ADS abstract as one object (its title, if ADS has no abstract)."""
    title = meta.get("title") or ""
    content = meta.get("abstract") or title
    return {
        "properties": {
            **paper_props(meta, bibcode),
            "page_content":    content,
            "section_headers": ["Abstract"],
            "chunk_id":        abstract_id_for(bibcode),
            "chunk_index":     0,
            "coverage":        coverage,
        },
        "embed": embedding_text(title, "Abstract", content),
    }


def paper_objects(meta: dict, bibcode: str, markdown: str | None,
                  settings: ChunkSettings, count_tokens) -> list[dict]:
    """Every object of one paper: its full-text chunks, or its abstract alone.

    ``markdown`` must already have passed :func:`markdown_problem` and the
    licence check; None means the paper is indexed by its abstract.
    ``fingerprint`` is left for the caller to add.
    """
    if markdown is None:
        return [abstract_object(meta, bibcode, ABSTRACT_ONLY)]

    if meta.get("date"):
        latest = datetime.fromisoformat(meta["date"][:10]).date() + FALSE_DATE_SLACK
        markdown, _ = strip_false_dates(markdown, lambda day: day > latest)

    pieces = split_markdown_packed(
        markdown,
        headers_to_split_on=HEADERS,
        max_tokens=settings.max_tokens,
        overlap_tokens=settings.overlap,
        min_tokens=settings.min_tokens,
        atomic_max_tokens=settings.atomic_max,
        drop_sections=settings.drop_sections,
        count_tokens=count_tokens,
    )
    title = meta.get("title") or ""
    props = paper_props(meta, bibcode)
    out = []
    if meta.get("abstract") and abstract_share(meta["abstract"], markdown) < ABSTRACT_CHECK_MIN:
        out.append(abstract_object(meta, bibcode, FULL_TEXT))
    for n, piece in enumerate(pieces, start=1):
        out.append({
            "properties": {
                **props,
                "page_content":    piece.content,
                "section_headers": piece.headers,
                "chunk_id":        chunk_id_for(bibcode, n),
                "chunk_index":     n,
                "coverage":        FULL_TEXT,
            },
            "embed": embedding_text(title, piece.header_path(), piece.content),
        })
    return out


def build_chunks(md_dir: str, metadata_path: str, settings: ChunkSettings, count_tokens,
                 title_check: bool = True) -> list[dict]:
    """Every object of every paper on the list, for a rebuild."""
    papers, aliases = load_papers(metadata_path)
    md_by_paper, unlisted = match_markdown(md_dir, aliases)

    out: list[dict] = []
    wrong_content: list[str] = []
    not_allowed: list[str] = []
    n_full = 0
    for bibcode, meta in papers.items():
        markdown = markdown_sha = None
        path = md_by_paper.get(bibcode)
        if path:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            name = os.path.basename(path)[:-3]
            problem = markdown_problem(text, meta) if title_check else None
            if not fulltext_allowed(meta):
                not_allowed.append(name)
            elif problem:
                wrong_content.append(f"{name} ({problem})")
            else:
                markdown, markdown_sha = text, sha256_text(text)
                n_full += 1
                if name != bibcode:
                    print(f"  {name} is indexed under its current bibcode, {bibcode}")
        fp = fingerprint(meta, bibcode, markdown_sha, settings)
        for obj in paper_objects(meta, bibcode, markdown, settings, count_tokens):
            obj["properties"]["fingerprint"] = fp
            out.append(obj)

    print(f"  {len(papers)} papers: {n_full} full text, {len(papers) - n_full} abstract only")
    if unlisted:
        print(f"  ! {len(unlisted)} markdown file(s) left out, their paper is not on "
              f"the PLATO-Pub list: {', '.join(unlisted)}")
    if not_allowed:
        print(f"  ! {len(not_allowed)} markdown file(s) left out because the paper is "
              f"paywalled (abstract only): {', '.join(not_allowed)}")
    if wrong_content:
        print(f"  !! {len(wrong_content)} markdown file(s) LEFT OUT -- check the "
              f"conversion: {'; '.join(wrong_content)}")
    return out


def report(chunks: list[dict], count_tokens, max_tokens: int) -> None:
    lens = sorted(count_tokens(c["properties"]["page_content"]) for c in chunks)
    docs = {c["properties"]["parent_doc"] for c in chunks}
    over = sum(1 for l in lens if l > max_tokens)
    print(f"  {len(chunks)} objects from {len(docs)} papers")
    if over:
        print(f"  {over} over the {max_tokens}-token budget — tables kept whole")
    print(f"  tokens/object: median {statistics.median(lens):.0f}  "
          f"mean {statistics.mean(lens):.0f}  min {min(lens)}  max {max(lens)}")
    print(f"  total tokens indexed: {sum(lens):,}")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_tokenizer_counter(model: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    return lambda s: len(tok.encode(s, add_special_tokens=False))


def load_embedder(model: str, device: str | None = None):
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Loading {model} on {device} …")
    return SentenceTransformer(model, device=device)


def embed(embedder, objects: list[dict], batch_size: int = 16, progress: bool = False):
    return embedder.encode(
        [o["embed"] for o in objects],
        batch_size=batch_size,
        normalize_embeddings=True,   # must match vectorize() in plato_core.py
        show_progress_bar=progress,
    )


# ---------------------------------------------------------------------------
# Weaviate
# ---------------------------------------------------------------------------

def _schema():
    from weaviate.classes.config import DataType, Property, Tokenization
    return [
        Property(name="title",           data_type=DataType.TEXT),
        Property(name="authors",         data_type=DataType.TEXT),
        Property(name="short_ref",       data_type=DataType.TEXT),
        Property(name="journal",         data_type=DataType.TEXT),
        Property(name="year",            data_type=DataType.INT),
        Property(name="pubdate",         data_type=DataType.TEXT),
        Property(name="pub_date",        data_type=DataType.DATE),
        Property(name="page_content",    data_type=DataType.TEXT),
        Property(name="section_headers", data_type=DataType.TEXT_ARRAY),
        Property(name="parent_doc",      data_type=DataType.TEXT),
        Property(name="chunk_id",        data_type=DataType.TEXT),
        Property(name="chunk_index",     data_type=DataType.INT),
        Property(name="link",            data_type=DataType.TEXT),
        Property(name="doi",             data_type=DataType.TEXT),
        Property(name="arxiv_id",        data_type=DataType.TEXT),
        # "full text" or "abstract only": what the index holds of the paper.
        Property(name="coverage",        data_type=DataType.TEXT,
                 tokenization=Tokenization.FIELD, index_searchable=False),
        # See fingerprint(). Only ever read back, never searched or filtered.
        Property(name="fingerprint",     data_type=DataType.TEXT,
                 tokenization=Tokenization.FIELD, index_searchable=False,
                 index_filterable=False),
    ]


def _create_collection(client, name: str):
    from weaviate.classes.config import Configure, VectorDistances

    # vectorizer=none: vectors are computed here and passed in, so Weaviate
    # never needs the embedding model. Cosine matches how bge-m3 is trained
    # and how vectorize() normalises in plato_core.py.
    client.collections.create(
        name,
        properties=_schema(),
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE
        ),
    )
    return client.collections.get(name)


def recreate_collection(client, name: str, dim: int):
    """Drop and recreate the collection. Destructive, by design."""
    if client.collections.exists(name):
        n = len(client.collections.get(name))
        print(f"  dropping existing '{name}' ({n} objects) …")
        client.collections.delete(name)
    collection = _create_collection(client, name)
    print(f"  created '{name}' for {dim}-dim vectors")
    return collection


def ensure_collection(client, name: str):
    """The collection, created if missing and given any property it lacks.

    Adding a property leaves existing objects without it (None), which is how
    the objects of a collection built before ``fingerprint`` existed show up as
    out of date.
    """
    if not client.collections.exists(name):
        print(f"  creating collection '{name}'")
        return _create_collection(client, name)
    collection = client.collections.get(name)
    have = {p.name for p in collection.config.get().properties}
    for prop in _schema():
        if prop.name not in have:
            print(f"  adding property '{prop.name}' to '{name}'")
            collection.config.add_property(prop)
    return collection


def index_state(collection) -> dict[str, dict[str, str | None]]:
    """``{parent_doc: {object uuid: fingerprint}}`` for the whole collection."""
    state: dict[str, dict[str, str | None]] = {}
    for obj in collection.iterator(return_properties=["parent_doc", "fingerprint"]):
        props = obj.properties
        state.setdefault(props.get("parent_doc") or "", {})[str(obj.uuid)] = props.get("fingerprint")
    return state


def object_uuid(obj: dict) -> str:
    from weaviate.util import generate_uuid5
    # Derived from chunk_id, so writing a chunk that is already there replaces
    # it rather than adding a twin.
    return str(generate_uuid5(obj["properties"]["chunk_id"]))


def upsert(collection, objects: list[dict], vectors) -> None:
    """Insert or replace objects (a batch write with an existing UUID replaces it)."""
    with collection.batch.fixed_size(batch_size=64) as batch:
        for obj, vector in zip(objects, vectors):
            props = {k: v for k, v in obj["properties"].items() if v is not None}
            batch.add_object(properties=props, vector=vector.tolist(),
                             uuid=object_uuid(obj))
    failed = collection.batch.failed_objects
    if failed:
        raise RuntimeError(f"{len(failed)} object(s) failed: {failed[0].message}")


def delete_objects(collection, uuids) -> int:
    """Delete objects by UUID; returns how many went."""
    from weaviate.classes.query import Filter

    uuids, deleted = sorted(uuids), 0
    for i in range(0, len(uuids), 500):
        res = collection.data.delete_many(where=Filter.by_id().contains_any(uuids[i:i + 500]))
        deleted += res.successful
    return deleted


def replace_paper(collection, objects: list[dict], vectors, old_uuids) -> tuple[int, int]:
    """Make the paper's objects in the collection exactly ``objects``.

    New objects are written first and the leftovers deleted after, so the paper
    is never missing from the index, only briefly doubled. If this stops half
    way the leftovers keep their old fingerprint, and the next update sees the
    paper as out of date and finishes the job. Returns ``(written, deleted)``.
    """
    upsert(collection, objects, vectors)
    stale = set(old_uuids) - {object_uuid(o) for o in objects}
    return len(objects), (delete_objects(collection, stale) if stale else 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", default="PLATO", help="target collection (default: PLATO)")
    ap.add_argument("--md-dir", default=DEFAULT_MD_DIR)
    ap.add_argument("--metadata", default=DEFAULT_METADATA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=os.environ.get("PLATO_EMBED_DEVICE"),
                    help="torch device (default: mps/cuda/cpu, whichever is available)")
    ap.add_argument("--max-tokens", type=int, default=512, help="chunk ceiling (default: 512)")
    ap.add_argument("--overlap", type=int, default=64, help="chunk overlap (default: 64)")
    ap.add_argument("--min-tokens", type=int, default=48, help="merge below this (default: 48)")
    ap.add_argument("--atomic-max-tokens", type=int, default=8192,
                    help="ceiling for tables, which are never word-split (default: 8192)")
    ap.add_argument("--no-title-check", action="store_true",
                    help="index a markdown file even if its paper's title is not in it")
    ap.add_argument("--keep-references", action="store_true",
                    help="index bibliographies, acknowledgements and keyword blocks too "
                         "(default: skip them)")
    ap.add_argument("--batch-size", type=int, default=16, help="embedding batch (default: 16)")
    ap.add_argument("--dry-run", action="store_true",
                    help="chunk and report, touch neither the model nor Weaviate")
    ap.add_argument("--yes", action="store_true",
                    help="required to drop a collection that already exists")
    args = ap.parse_args()

    if not os.path.isdir(args.md_dir):
        print(f"No markdown directory at {args.md_dir}", file=sys.stderr)
        return 1
    settings = ChunkSettings(max_tokens=args.max_tokens, overlap=args.overlap,
                             min_tokens=args.min_tokens, atomic_max=args.atomic_max_tokens,
                             keep_references=args.keep_references, model=args.model)

    # --dry-run keeps the fast, dependency-light path: the stdlib estimator in
    # textsplitter is close enough to preview chunk counts without a 2 GB load.
    count_tokens = None
    if not args.dry_run:
        print(f"Loading tokenizer ({args.model}) …")
        count_tokens = load_tokenizer_counter(args.model)

    print(f"Chunking {args.md_dir} …")
    if settings.drop_sections:
        print("  skipping References / Bibliography / Acknowledgements / Keywords sections")
    chunks = build_chunks(args.md_dir, args.metadata, settings, count_tokens,
                          title_check=not args.no_title_check)
    if not chunks:
        print("No chunks produced — nothing to index.", file=sys.stderr)
        return 1
    report(chunks, count_tokens or (lambda s: len(s) // 3), args.max_tokens)

    if args.dry_run:
        print("\nDry run — nothing written. Token counts are estimates; "
              "drop --dry-run for exact ones.")
        return 0

    embedder = load_embedder(args.model, args.device)
    t0 = time.time()
    vectors = embed(embedder, chunks, args.batch_size, progress=True)
    dim = int(vectors.shape[1])
    print(f"  embedded {len(chunks)} objects in {time.time() - t0:.0f}s ({dim} dims)")

    import weaviate

    client = weaviate.connect_to_local()
    try:
        if not client.is_ready():
            print("Weaviate is not ready.", file=sys.stderr)
            return 1
        if client.collections.exists(args.collection) and not args.yes:
            print(f"\nCollection '{args.collection}' already exists. Rebuilding it "
                  f"deletes every object in it.\nRe-run with --yes to confirm, or "
                  f"--collection NAME to build a new one alongside.", file=sys.stderr)
            return 1

        collection = recreate_collection(client, args.collection, dim)
        print(f"  inserting {len(chunks)} objects …")
        upsert(collection, chunks, vectors)
        print(f"\nDone. '{args.collection}' holds {len(collection)} objects "
              f"at {dim} dims.")
        print(f"Point the chatbot at it with PLATO_EMBED_MODEL={args.model} "
              f"(already the default in plato_core.py).")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
