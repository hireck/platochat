#!/usr/bin/env python
"""Rebuild the Weaviate PLATO collection from the markdown corpus.

Replaces the chunk-and-load half of ``get_present_papers.py``, with two
changes that go together:

* Chunks are header-split *and* paragraph-packed (see ``textsplitter.py``).
  Header splitting alone made one paper section the retrieval unit — 890
  tokens at the median, up to 70k — which no embedding model can see whole.
* Vectors come from bge-m3 (1024 dims, 8192-token window) rather than
  all-MiniLM-L12-v2 (384 dims, 128 tokens).

The dimension change alone means the collection cannot be updated in place:
it has to be dropped and rebuilt, which is why ``--yes`` is required to touch
an existing one. Run ``--dry-run`` first to see the chunk statistics without
connecting to Weaviate at all.

    python reindex_weaviate.py --dry-run
    python reindex_weaviate.py --collection PLATO_TEST      # new collection
    python reindex_weaviate.py --yes                        # rebuild PLATO

Keep the model here in step with ``PLATO_EMBED_MODEL`` in plato_core.py: a
query embedded by one model cannot be compared against a corpus embedded by
another.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textsplitter import DROP_SECTIONS, split_markdown_packed  # noqa: E402

DEFAULT_MD_DIR   = os.environ.get(
    "PLATO_MD_DIR", "/Users/hilke/data/plato_data/plato_markdown"
)
DEFAULT_METADATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "PLATOChat_papers.json"
)
DEFAULT_MODEL    = os.environ.get("PLATO_EMBED_MODEL", "BAAI/bge-m3")

# Header levels to split on. Papers converted from LaTeX/PDF nest to #####,
# and every level that starts a section should start a chunk.
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"), ("#####", "h5")]

# Metadata fields copied from PLATOChat_papers.json onto every chunk.
META_FIELDS = ["title", "authors", "journal", "arXiv ID"]


def embedding_text(title: str, header_path: str, content: str) -> str:
    """The text handed to the embedder — not the same as what we store.

    A chunk five paragraphs into "3.2 Noise budget" has no idea, on its own,
    which paper or section it belongs to; two such chunks from different papers
    can be near-identical in isolation. Prefixing the title and header path
    anchors the vector to its context. ``page_content`` keeps the clean text,
    so the prefix never reaches the answering model or the citation display.
    """
    return "\n".join(p for p in (title, header_path, "", content) if p is not None)


def build_chunks(md_dir: str, metadata_path: str, max_tokens: int, overlap: int,
                 min_tokens: int, atomic_max: int, drop_sections, count_tokens) -> list[dict]:
    with open(metadata_path) as fh:
        papers = json.load(fh)

    md_files = sorted(f for f in os.listdir(md_dir) if f.endswith(".md"))
    out: list[dict] = []
    skipped: list[str] = []
    # chunk_number is the citation handle: the answering model cites it as
    # <cite>N</cite> and build_sources_text() looks it up. It must therefore be
    # unique across the whole collection, not per paper.
    chunk_number = 1

    for fn in md_files:
        doc_id = fn[:-3]
        meta = papers.get(doc_id)
        if meta is None:
            skipped.append(doc_id)
            continue

        with open(os.path.join(md_dir, fn), encoding="utf-8") as fh:
            pieces = split_markdown_packed(
                fh.read(),
                headers_to_split_on=HEADERS,
                max_tokens=max_tokens,
                overlap_tokens=overlap,
                min_tokens=min_tokens,
                atomic_max_tokens=atomic_max,
                drop_sections=drop_sections,
                count_tokens=count_tokens,
            )

        title = meta.get("title") or ""
        for piece in pieces:
            out.append({
                "properties": {
                    "title":           title,
                    "authors":         meta.get("authors") or "",
                    "journal":         meta.get("journal") or "",
                    "page_content":    piece.content,
                    "section_headers": piece.headers,
                    "parent_doc":      doc_id,
                    "chunk_number":    chunk_number,
                    "link":            meta.get("arXiv ID") or "",
                },
                "embed": embedding_text(title, piece.header_path(), piece.content),
            })
            chunk_number += 1

    if skipped:
        print(f"  ! {len(skipped)} markdown file(s) had no metadata entry, skipped: "
              f"{', '.join(skipped[:5])}{' …' if len(skipped) > 5 else ''}")
    return out


def report(chunks: list[dict], count_tokens, max_tokens: int) -> None:
    lens = sorted(count_tokens(c["properties"]["page_content"]) for c in chunks)
    docs = {c["properties"]["parent_doc"] for c in chunks}
    over = sum(1 for l in lens if l > max_tokens)
    print(f"  {len(chunks)} chunks from {len(docs)} papers")
    if over:
        print(f"  {over} over the {max_tokens}-token budget — tables kept whole")
    print(f"  tokens/chunk: median {statistics.median(lens):.0f}  "
          f"mean {statistics.mean(lens):.0f}  min {min(lens)}  max {max(lens)}")
    print(f"  total tokens indexed: {sum(lens):,}")


def recreate_collection(client, name: str, dim: int):
    """Drop and recreate the collection. Destructive, by design."""
    from weaviate.classes.config import (
        Configure, DataType, Property, VectorDistances,
    )

    if client.collections.exists(name):
        n = len(client.collections.get(name))
        print(f"  dropping existing '{name}' ({n} objects) …")
        client.collections.delete(name)

    # vectorizer=none: vectors are computed here and passed in, so Weaviate
    # never needs the embedding model. Cosine matches how bge-m3 is trained
    # and how vectorize() normalises in plato_core.py.
    client.collections.create(
        name,
        properties=[
            Property(name="title",           data_type=DataType.TEXT),
            Property(name="authors",         data_type=DataType.TEXT),
            Property(name="journal",         data_type=DataType.TEXT),
            Property(name="page_content",    data_type=DataType.TEXT),
            Property(name="section_headers", data_type=DataType.TEXT_ARRAY),
            Property(name="parent_doc",      data_type=DataType.TEXT),
            Property(name="chunk_number",    data_type=DataType.INT),
            Property(name="link",            data_type=DataType.TEXT),
        ],
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE
        ),
    )
    print(f"  created '{name}' for {dim}-dim vectors")
    return client.collections.get(name)


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
    # Tables are kept whole rather than word-split, so their ceiling is the
    # embedder's window, not the chunk budget -- past it the model truncates
    # anyway, so that is the point at which splitting starts to be the lesser
    # evil. 8192 is bge-m3's; lower it if you switch to a shorter-context model.
    ap.add_argument("--atomic-max-tokens", type=int, default=8192,
                    help="ceiling for tables, which are never word-split (default: 8192)")
    ap.add_argument("--keep-references", action="store_true",
                    help="index bibliographies and acknowledgements too (default: skip them)")
    ap.add_argument("--batch-size", type=int, default=16, help="embedding batch (default: 16)")
    ap.add_argument("--dry-run", action="store_true",
                    help="chunk and report, touch neither the model nor Weaviate")
    ap.add_argument("--yes", action="store_true",
                    help="required to drop a collection that already exists")
    args = ap.parse_args()

    if not os.path.isdir(args.md_dir):
        print(f"No markdown directory at {args.md_dir}", file=sys.stderr)
        return 1

    # --dry-run keeps the fast, dependency-light path: the stdlib estimator in
    # textsplitter is close enough to preview chunk counts without a 2 GB load.
    count_tokens = None
    if not args.dry_run:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer ({args.model}) …")
        tok = AutoTokenizer.from_pretrained(args.model)
        count_tokens = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731

    drop_sections = None if args.keep_references else DROP_SECTIONS
    print(f"Chunking {args.md_dir} …")
    if drop_sections:
        print("  skipping References / Bibliography / Acknowledgements sections")
    chunks = build_chunks(args.md_dir, args.metadata, args.max_tokens, args.overlap,
                          args.min_tokens, args.atomic_max_tokens, drop_sections,
                          count_tokens)
    if not chunks:
        print("No chunks produced — nothing to index.", file=sys.stderr)
        return 1
    report(chunks, count_tokens or (lambda s: len(s) // 3), args.max_tokens)

    if args.dry_run:
        print("\nDry run — nothing written. Token counts are estimates; "
              "drop --dry-run for exact ones.")
        return 0

    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device or (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"\nLoading {args.model} on {device} …")
    model = SentenceTransformer(args.model, device=device)

    t0 = time.time()
    vectors = model.encode(
        [c["embed"] for c in chunks],
        batch_size=args.batch_size,
        normalize_embeddings=True,   # must match vectorize() in plato_core.py
        show_progress_bar=True,
    )
    dim = int(vectors.shape[1])
    print(f"  embedded {len(chunks)} chunks in {time.time() - t0:.0f}s ({dim} dims)")

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
        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                batch.add_object(properties=chunk["properties"], vector=vector.tolist())

        failed = collection.batch.failed_objects
        if failed:
            print(f"  ! {len(failed)} object(s) failed: {failed[0].message}", file=sys.stderr)
            return 1
        print(f"\nDone. '{args.collection}' holds {len(collection)} objects "
              f"at {dim} dims.")
        print(f"Point the chatbot at it with PLATO_EMBED_MODEL={args.model} "
              f"(already the default in plato_core.py).")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
