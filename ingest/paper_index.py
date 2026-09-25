#!/usr/bin/env python
"""The whole-paper index: one Weaviate object per paper, for finding papers.

The chunk collection (PLATO) answers questions about what the papers *say*.
This one answers questions about the papers themselves -- which papers there
are on a topic, who wrote what, what came out when -- which the chunks cannot:
their search looks at the text, never at authors or dates, and a paper's
hundred chunks would crowd a list of ten papers out of the results.

One object per paper on the PLATO-Pub list, holding its title, abstract and ADS
keywords (searched, by vector and BM25) and its authors and date (filtered on).
The vector is bge-m3 over title and abstract, as for the chunks, so the chatbot
embeds a query once for both.

Named after the chunk collection: ``PLATO`` -> ``PLATO_PAPERS``, ``PLATO_NEXT``
-> ``PLATO_NEXT_PAPERS``. update_corpus.py brings it up to date after the
chunks, every run; run on its own, this script does the same:

    python paper_index.py                          # PLATO_PAPERS
    python paper_index.py --collection PLATO_NEXT  # PLATO_NEXT_PAPERS
    python paper_index.py --dry-run

Each paper also records what the chunk collection holds of it (``coverage``:
full text or abstract only), read from that collection rather than from the
manifest -- so the chatbot can tell the model whether a full-text search will
find more of the paper. Like the chunks, every object carries a fingerprint and
is rewritten only when it changes; papers that left the list are deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reindex_weaviate import (  # noqa: E402
    DEFAULT_METADATA, DEFAULT_MODEL, FULL_TEXT, delete_objects, embed,
    load_embedder, load_papers,
)

# Bump when the objects change in a way the fingerprint does not see (a new
# property, a different embedding text). Every paper is rewritten on the next
# run -- 178 abstracts, about half a minute.
PAPER_INDEX_VERSION = 1

PAPERS_SUFFIX = "_PAPERS"

# Name particles that are never a surname on their own: "De Ridder" is found
# as "De Ridder" or "Ridder", never as "De".
_PARTICLES = {"al", "da", "das", "de", "del", "della", "den", "der", "di", "do", "dos",
              "du", "e", "el", "la", "le", "les", "st", "ten", "ter", "van", "von", "y"}
# Letters NFKD does not take apart into a base letter and an accent.
_FOLD = str.maketrans({"ø": "o", "æ": "ae", "œ": "oe", "ß": "ss", "ł": "l", "đ": "d",
                       "ð": "d", "þ": "th", "ı": "i"})


def paper_collection_name(chunk_collection: str) -> str:
    return chunk_collection + PAPERS_SUFFIX


def fold_name(name: str) -> list[str]:
    """A name as plain lower-case ASCII words: 'Aguirre Børsen-Koch' -> [aguirre, borsen, koch].

    The chatbot's ``author_keys()`` in plato_core.py does the same to the
    names it is asked for; the two must stay in step.
    """
    name = unicodedata.normalize("NFKD", name.casefold().translate(_FOLD))
    name = "".join(c for c in name if not unicodedata.combining(c))
    return "".join(c if c.isalnum() else " " for c in name).split()


def surname_keys(author: str) -> list[str]:
    """Every form of an ADS author's surname someone might search for.

    ADS writes "Surname, Given names". A compound surname is found whole or by
    any run of its parts -- "Aguirre Børsen-Koch" by "Børsen-Koch", "Koch" or
    "Aguirre" -- because people cite such names inconsistently, and ADS itself
    has "Collier Cameron" and "Collier-Cameron" for the same person. Exact
    keys, compared exactly: a fuzzier match would find "Lund" in "Lundkvist".
    """
    words = fold_name(author.split(",")[0])
    keys = []
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            key = " ".join(words[i:j])
            if j - i == 1 and key in _PARTICLES:
                continue
            if key not in keys:
                keys.append(key)
    return keys


def _keywords(meta: dict) -> list[str]:
    kw = meta.get("keyword") or []
    return [str(k) for k in kw] if isinstance(kw, list) else [str(kw)]


def paper_object(meta: dict, bibcode: str, coverage: str) -> dict:
    """One paper's object: what is searched, what is filtered on, what is shown."""
    authors = list(meta.get("author") or [])
    title = meta.get("title") or ""
    abstract = meta.get("abstract") or ""
    return {
        "properties": {
            "bibcode":           bibcode,
            "title":             title,
            "abstract":          abstract,
            "keywords":          _keywords(meta),
            "authors":           authors,
            "author_count":      len(authors),
            "author_keys":       sorted({k for a in authors for k in surname_keys(a)}),
            "first_author_keys": surname_keys(authors[0]) if authors else [],
            "short_ref":         meta.get("short_ref") or "",
            "journal":           meta.get("pub") or "",
            "doctype":           meta.get("doctype") or "",
            "year":              int(meta["year"]) if str(meta.get("year") or "").isdigit() else None,
            "pubdate":           meta.get("pubdate_display") or "",
            "pub_date":          (datetime.fromisoformat(meta["date"].replace("Z", "+00:00"))
                                  if meta.get("date") else None),
            "link":              meta.get("ads_url") or "",
            "doi":               (meta.get("doi") or [""])[0],
            "arxiv_id":          meta.get("arxiv_id") or "",
            "coverage":          coverage,
        },
        # Title and abstract, as asked of the index; without an abstract (three
        # papers) the title alone.
        "embed": "\n\n".join(p for p in (title, abstract) if p),
    }


def paper_fingerprint(obj: dict, model: str) -> str:
    basis = {
        "props": {k: str(v) for k, v in obj["properties"].items()},
        "embed": obj["embed"],
        "model": model,
        "version": PAPER_INDEX_VERSION,
    }
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode("utf-8")).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Weaviate
# ---------------------------------------------------------------------------

def _schema():
    from weaviate.classes.config import DataType, Property, Tokenization

    def exact(name, data_type=DataType.TEXT, filterable=True):
        # Identifiers and names: compared whole, never scored by BM25.
        return Property(name=name, data_type=data_type, tokenization=Tokenization.FIELD,
                        index_searchable=False, index_filterable=filterable)

    return [
        exact("bibcode"),
        # What BM25 searches (see PAPER_BM25_PROPERTIES in plato_core.py).
        Property(name="title",    data_type=DataType.TEXT),
        Property(name="abstract", data_type=DataType.TEXT),
        Property(name="keywords", data_type=DataType.TEXT_ARRAY),
        # The names as ADS writes them, for showing; the folded surname keys,
        # for filtering (see surname_keys).
        exact("authors", DataType.TEXT_ARRAY, filterable=False),
        Property(name="author_count", data_type=DataType.INT),
        exact("author_keys", DataType.TEXT_ARRAY),
        exact("first_author_keys", DataType.TEXT_ARRAY),
        exact("short_ref", filterable=False),
        exact("journal", filterable=False),
        exact("doctype"),
        Property(name="year",     data_type=DataType.INT),
        exact("pubdate", filterable=False),
        Property(name="pub_date", data_type=DataType.DATE),
        exact("link", filterable=False),
        exact("doi"),
        exact("arxiv_id"),
        exact("coverage"),
        exact("fingerprint", filterable=False),
    ]


def ensure_paper_collection(client, name: str):
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
    collection = client.collections.get(name)
    have = {p.name for p in collection.config.get().properties}
    for prop in _schema():
        if prop.name not in have:
            print(f"  adding property '{prop.name}' to '{name}'")
            collection.config.add_property(prop)
    return collection


def chunk_coverage(chunk_collection) -> dict[str, str]:
    """``{bibcode: coverage}`` as the chunk collection has it."""
    out: dict[str, str] = {}
    have = {p.name for p in chunk_collection.config.get().properties}
    wanted = ["parent_doc"] + (["coverage"] if "coverage" in have else [])
    for obj in chunk_collection.iterator(return_properties=wanted):
        bib = obj.properties.get("parent_doc") or ""
        # Objects written before `coverage` existed were all full text.
        if (obj.properties.get("coverage") or FULL_TEXT) == FULL_TEXT or bib not in out:
            out[bib] = obj.properties.get("coverage") or FULL_TEXT
    return out


def paper_uuid(bibcode: str) -> str:
    from weaviate.util import generate_uuid5
    return str(generate_uuid5("paper:" + bibcode))


def sync_papers(client, chunk_collection_name: str, papers: dict, *, model: str = DEFAULT_MODEL,
                device: str | None = None, embedder=None, dry_run: bool = False,
                max_removals: int | None = None, log=print) -> dict:
    """Make ``<chunks>_PAPERS`` hold exactly the papers on the list, up to date.

    A paper missing from the chunk collection is left out here too (it will be
    added once its chunks are): the paper index must not offer a paper that a
    full-text search cannot then find. Returns counts of what was done.
    """
    name = paper_collection_name(chunk_collection_name)
    if not client.collections.exists(chunk_collection_name):
        raise RuntimeError(f"no chunk collection '{chunk_collection_name}' to take coverage from")
    coverage = chunk_coverage(client.collections.get(chunk_collection_name))

    objects = {}
    for bib, meta in papers.items():
        cov = next((coverage[a] for a in (bib, *meta.get("aliases", [])) if a in coverage), None)
        if cov is None:
            continue
        obj = paper_object(meta, bib, cov)
        obj["properties"]["fingerprint"] = paper_fingerprint(obj, model)
        objects[paper_uuid(bib)] = obj
    missing = [b for b in papers if paper_uuid(b) not in objects]

    if dry_run and not client.collections.exists(name):
        state = {}
        log(f"  collection '{name}' does not exist yet; it would be created")
    else:
        collection = (client.collections.get(name) if dry_run
                      else ensure_paper_collection(client, name))
        state = {str(o.uuid): o.properties.get("fingerprint")
                 for o in collection.iterator(return_properties=["fingerprint"])}

    todo = [u for u, o in objects.items() if state.get(u) != o["properties"]["fingerprint"]]
    stale = set(state) - set(objects)
    stats = {"papers": len(objects), "written": len(todo), "deleted": 0, "not in chunks": len(missing)}
    log(f"  '{name}': {len(objects)} papers, {len(todo)} to write, {len(stale)} to remove"
        + (f"; {len(missing)} not in '{chunk_collection_name}' yet, left out" if missing else ""))
    if dry_run:
        return stats

    if todo:
        embedder = embedder or load_embedder(model, device)
        batch = [objects[u] for u in todo]
        vectors = embed(embedder, batch)
        with collection.batch.fixed_size(batch_size=64) as b:
            for uuid, obj, vector in zip(todo, batch, vectors):
                props = {k: v for k, v in obj["properties"].items() if v is not None}
                b.add_object(properties=props, vector=vector.tolist(), uuid=uuid)
        failed = collection.batch.failed_objects
        if failed:
            raise RuntimeError(f"{len(failed)} paper object(s) failed: {failed[0].message}")
    if stale:
        # Same guard as for the chunks: a list that loses many papers at once
        # is likelier broken than true.
        if max_removals is not None and len(stale) > max_removals:
            log(f"  ! {len(stale)} papers in '{name}' are not on the list; refusing to remove "
                f"that many at once (--allow-removals)")
        else:
            stats["deleted"] = delete_objects(collection, stale)
    log(f"  '{name}' now holds {len(collection)} papers")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", default=os.environ.get("PLATO_COLLECTION", "PLATO"),
                    help="the chunk collection; the paper index is it + '_PAPERS' "
                         "(default: $PLATO_COLLECTION, else PLATO)")
    ap.add_argument("--metadata", default=DEFAULT_METADATA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=os.environ.get("PLATO_EMBED_DEVICE"))
    ap.add_argument("--dry-run", action="store_true", help="say what would be done, write nothing")
    args = ap.parse_args()

    import weaviate

    papers, _ = load_papers(args.metadata)
    client = weaviate.connect_to_local()
    try:
        if not client.is_ready():
            print("Weaviate is not ready.", file=sys.stderr)
            return 1
        sync_papers(client, args.collection, papers, model=args.model, device=args.device,
                    dry_run=args.dry_run)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
