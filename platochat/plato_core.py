"""
PLATO chatbot — core RAG pipeline (UI-agnostic).

This module holds everything the chatbot needs to turn a user question into a
grounded, cited answer: configuration, prompts, model loading, hybrid
retrieval (vector + BM25) and rerank, the routing agent, and the
citation/sources formatting. It contains NO
Streamlit (or any other UI) code, so it can be driven from:

    - the FastAPI service in ``plato_api.py`` (the web UI), or
    - any other front-end / batch script.

The single entry point is :func:`answer_question`, which returns plain strings
(``reply`` markdown + optional ``sources`` markdown) instead of writing to a UI.

Extracted from the original Streamlit module ``plato_chat.py``. The one
behavioural difference: we do NOT apply the old ``fix_math_formatting`` step,
which rewrote display math into ``st.latex(...)`` calls — that is specific to
Streamlit's renderer. Here we keep standard LaTeX so the browser's MathJax can
typeset it.
"""

import json
import os
import re
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()  # must run before Langfuse imports so credentials are available

import instructor
import torch
from langfuse import Langfuse
from langfuse.openai import OpenAI
from pydantic import BaseModel, Field
import weaviate
import weaviate.classes.query as wq
from sentence_transformers import CrossEncoder, SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _best_device() -> str:
    """Best torch device available here: Apple GPU, CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


OPENAI_API_KEY       = os.environ.get("OPENAI_API_KEY")
LANGFUSE_HOST        = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
OPENWEBUI_API_KEY    = os.environ.get("OPENWEBUI_API_KEY")
LLM_BASE_URL         = os.environ.get("PLATO_LLM_BASE_URL", "https://labbot.nat.au.dk/api")
# The AI Lab endpoint serves GLM-5.3 in three reasoning tiers (GLM-5.3-instant /
# GLM-5.3-high / GLM-5.3-max), plus Flash variants, Qwen3.8 and lab-* aliases.
# NOTE the model ids carry the version number: when the lab moved from GLM-5.2
# to 5.3 (seen 2026-09-20) the old ids (GLM-high, GLM-instant, MiniMax, Kimi)
# vanished and every call failed with 400 "Model not found". If that error
# comes back, re-list the roster (GET {LLM_BASE_URL}/models) and update these.
#
# Answering runs on the -high tier: measured on GLM-5.2 against MiniMax on the
# same questions it returned the headline figures the retrieved passages contain
# (rather than only the surrounding formalism) and did so faster. The -max tier
# reasons harder still but took 3-5x longer per answer, which a chat UI cannot
# hide -- it is the tier to reach for if a coding path is added later, not for
# interactive Q&A.
#
# GLM-5.3-Flash -- a separate model (RedHatAI/GLM-5.3-Flash-NVFP4) on a server
# of its own, and what the lab's lab-rag, lab-standard and lab-instant aliases
# point to -- was compared on 2026-09-28: three full eval runs each, Flash
# routing and answering. The checks came out level, but Flash was not faster:
# in its default mode it reasons about 13x as long (a median of 446 hidden
# tokens per answer against 34 here), which outweighs its faster generation --
# median 2.4 s against 1.9 s in the answering model, 8.3 s against 3.1 s at the
# 90th percentile. It also searched again in more answers (32 of 141 against
# 20), and it alone returned an empty reply (once, having spent its search
# budget) and cited passages it had not been given (twice). Hence -high stays.
# eval/compare_runs.py repeats such a comparison.
#
# Routing runs on the -instant tier: the router only picks sources and rewrites
# the question into a standalone query, so a reasoning tier buys nothing, and a
# non-reasoning model cannot preface its JSON with a <think> block for
# instructor to trip over.
ANSWER_MODEL         = os.environ.get("PLATO_ANSWER_MODEL", "GLM-5.3-high")
ROUTER_MODEL         = os.environ.get("PLATO_ROUTER_MODEL", "GLM-5.3-instant")

# Workaround for the OpenWebUI proxy in front of vLLM: it injects
# stream_options={"include_usage": True} into forwarded requests but does not
# set stream=true, and vLLM rejects that combination with a 400. Streaming from
# our side makes the injected field valid. (Sending stream=False explicitly
# avoids the 400 but returns empty content, so streaming is the only way
# through.) Set PLATO_FORCE_STREAM=0 once the endpoint is fixed.
FORCE_STREAM         = os.environ.get("PLATO_FORCE_STREAM", "1").strip().lower() \
                       not in ("0", "false", "no", "off", "")

# Retrieval models. bge-m3 replaced all-MiniLM-L12-v2, which truncated at 128
# tokens -- against a corpus whose chunks ran to 890 tokens at the median, that
# meant only ~8% of the indexed text ever reached the embedder, and vector
# search was matching little more than each section's opening paragraph. bge-m3
# takes 8192 tokens, needs no "represent this sentence for searching..." query
# instruction (unlike the bge v1.5 English models), and costs ~20 ms per query
# on MPS -- indistinguishable from MiniLM in the request path. It is a far
# larger model, so it holds ~2.2 GB of RAM and a full re-index takes minutes
# rather than seconds; both are one-off costs paid off the request path.
#
# The reranker moved with it: ms-marco-MiniLM-L-6-v2 capped at 512 tokens, so
# it too read only part of each chunk.
EMBED_MODEL          = os.environ.get("PLATO_EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL         = os.environ.get("PLATO_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
# Apple GPU where available, else CUDA, else CPU. bge-m3 is ~17x slower on CPU
# (91 ms vs 20 ms per query), so this is worth getting right.
EMBED_DEVICE         = os.environ.get("PLATO_EMBED_DEVICE") or _best_device()
# Weaviate collection to query. Overridable so a re-indexed collection built
# alongside the live one (reindex_weaviate.py --collection ...) can be A/B'd
# without touching PLATO. The vectors in it must come from EMBED_MODEL.
WEAVIATE_COLLECTION  = os.environ.get("PLATO_COLLECTION", "PLATO")
# The whole-paper index built beside it by ingest/paper_index.py: one object
# per paper (title, abstract, authors, date), for the find_papers tool.
PAPER_COLLECTION     = os.environ.get("PLATO_PAPER_COLLECTION",
                                      WEAVIATE_COLLECTION + "_PAPERS")

HISTORY_WINDOW       = 4       # previous messages folded into the prompt
# Candidates fetched per retriever. Vector and BM25 run as separate queries
# whose results are unioned, so the rerank pool is up to 2x this, less overlap.
# The searches themselves are nearly free -- a local HNSW lookup and an
# inverted-index lookup. What this number costs is the cross-encoder, which
# scores every candidate against a ~512-token chunk and dominates the latency
# of retrieve_docs. Hence 15 rather than the 20 that vector-only search used.
RETRIEVAL_LIMIT      = 15
# BM25 is pinned to the prose properties. Left unset, Weaviate scores every
# TEXT property in the schema -- including parent_doc and link, which are bare
# identifiers, and authors/journal, where one surname colliding with a query
# word drags in a researcher's whole bibliography. title is boosted: a chunk
# from a paper that is *about* the query term beats a passing mention of it.
BM25_PROPERTIES      = ["page_content", "title^2", "section_headers"]
RERANK_TOP_K         = 7       # top documents after cross-encoder reranking
RERANK_FALLBACK_K    = 2       # fallback below the relevance threshold
# Minimum cross-encoder score for a candidate to count as relevant. NOTE the
# scale is model-dependent, and sentence-transformers hides the difference:
# bge-reranker-v2-m3 declares num_labels=1, so CrossEncoder applies a sigmoid
# and scores land in (0, 1) -- whereas ms-marco-MiniLM-L-6-v2 pins itself to
# Identity via sbert_ce_default_activation_function and returns raw logits
# spanning roughly -11..+7. The old threshold of 0 was meaningful there; kept
# as-is under a sigmoid it would pass every candidate and silently retire the
# fallback path below. Override only alongside PLATO_RERANK_MODEL.
RERANK_THRESHOLD     = float(os.environ.get("PLATO_RERANK_THRESHOLD", "0.5"))
# Extra searches the answering model may request after the router's first one.
# Shared by both tools: a turn may go on find_papers, search_publications or
# (several at once) both.
FOLLOWUP_SEARCHES    = int(os.environ.get("PLATO_FOLLOWUP_SEARCHES", "2"))

# find_papers. Title, abstract and ADS keywords are what a topic query is
# matched against, the title boosted as for the chunks. The reranker reads
# title and abstract together; RERANK_THRESHOLD separates on-topic papers
# there as it does for chunks (checked on six topics, 2026-09-24: "white
# dwarfs" 0.97/0.70/0.45 then below 0.06). A query naming PLATO itself lifts
# nearly every paper over it, which is what the cap is for.
PAPER_BM25_PROPERTIES = ["title^2", "abstract", "keywords"]
PAPER_RETRIEVAL_LIMIT = 20     # candidates per retriever, as RETRIEVAL_LIMIT
PAPER_DEFAULT_LIMIT   = 10     # papers returned unless the model asks for more
PAPER_MAX_LIMIT       = 25
PAPER_FALLBACK_K      = 3      # shown, as weak matches, when none passes
# A list of more papers than this gets their abstracts shortened, so that a
# long bibliography does not flood the context.
PAPER_FULL_ABSTRACTS  = 10
PAPER_SHORT_ABSTRACT  = 400    # characters
PAPER_SHOWN_AUTHORS   = 6


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# The papers come first, but an astronomy question they do not cover may be
# answered from the model's own knowledge -- provided the user can see that it
# was. The label is fixed wording rather than left to the model, so that users
# learn to recognise it and eval/run_eval.py can check for it. It matters most
# for facts about PLATO itself, where the model's training data can be years
# behind the papers (the very problem the publication dates are there to solve).
GENERAL_KNOWLEDGE_LABEL = "General knowledge, not from the PLATO publications"

system_prompt = f"""You are a PLATO expert at the ESA helpdesk. Your task is to help researchers learn about Plato and use Plato data products effectively in their work.

Be helpful and concise. Volunteer additional information where relevant, but
keep it brief.

Your answers rest on the information supplied with the user message. Do not
fabricate facts, and never present something as coming from the PLATO
publications when it does not.

If that information does not answer an astronomy-related question --- after
you have searched again, where you can --- you may answer from your own
general knowledge, but you must flag it. Say first that the PLATO publications
available to you do not cover the point. Then give the general-knowledge part
in a paragraph of its own that begins with exactly this label:

**{GENERAL_KNOWLEDGE_LABEL}:**

Never put <cite> tags in such a paragraph, and do not use the label for
anything the supplied information does support. Take particular care with
facts about PLATO itself: your general knowledge of the mission may be out of
date, so say so when you fall back on it. Never let it contradict a passage
--- where what you remember disagrees with the retrieved material, the
passage is right and your memory is the stale one. The mission's parameters
and schedule --- launch date, number of cameras, field of view, observing
fields --- come from the passages or not at all: if none of them gives the
figure, say that the publications you have do not state it, and do not supply
one from memory. If you do not know the answer either, say so plainly. If a question has nothing to do with astronomy or
PLATO, do not answer it; explain briefly that it is outside what PLATO Chat
is for.

The user message may be accompanied by:
  - the information about the chatbot and website: a description of PLATO
    Chat and its sources, plus general information about the mission and
    useful links from the public PLATO-Pub website, and/or
  - passages retrieved from the index of published articles on ESA's PLATO
    mission, each preceded by its metadata: a `passage` number, the paper it
    comes from, and when that paper was published.

Some papers are in the index by their abstract alone, mostly because their
full text is paywalled; their passages are marked `coverage: abstract only`.
An abstract says what a paper is about, not everything in it: never conclude
from one that the paper does not discuss something, and where a question
needs more detail than the abstract gives, point the user to the paper.

If retrieved passages are provided, cite each passage you use by wrapping its
passage number in <cite>...</cite> tags, e.g. `<cite>3</cite>` --- only the
number inside the tags, and each cited passage number in its own pair of tags.
Do not cite the information about the chatbot and website. Do not list
sources at the end; citations are inline only. Do not fabricate citations.
If NO retrieved passages are provided, do not emit any <cite> tags.

PLATO's design and schedule have changed over the years (number of cameras,
field of view, launch date, observing fields, ...), so an older paper can be
out of date. Check the `published` date of every passage. Where passages
disagree, rely on the most recent one, and say that earlier papers gave a
different figure. A date or figure you merely remember is not a source at
all, and is likelier to be stale than any of them. When you state something that may have changed since it was
written, say how old your source is, e.g. "as of Nascimbeni et al. (2022)".

Format the answer as markdown. Use the LaTeX notation for Math.
"""

# Appended to ``system_prompt`` for the answering call. The router has already
# run one search by the time the answering model sees this, so the tool is a
# recovery path, not the primary retrieval step -- the prompt says so, to keep
# the model from searching reflexively on every turn.
SEARCH_TOOL_INSTRUCTIONS = """

You also have a `search_publications` tool. Any passages above were retrieved
for you already, by a search that ran before you were called.

If those passages answer the question, just answer --- do not search. But if
they do not, search instead of giving up: whenever you are about to tell the
user that the retrieved material is insufficient, missing or off-topic, call
the tool first with a better query. That is exactly what it is for.

The query must stand on its own: the index cannot see this conversation, so
resolve every pronoun and back-reference first ("and its field of view?"
becomes "PLATO field of view"). You may call your tools in at most {n} more
round(s) -- calls made together count as one; after that, answer with what
you have and state plainly what the publications did not cover.
"""

# Appended after SEARCH_TOOL_INSTRUCTIONS when the paper index exists. The
# router only ever runs a passage search, so a question about authors or dates
# arrives with passages that cannot answer it; the prompt has to say that the
# other tool is the way out, or the model reports the passages as insufficient.
FIND_PAPERS_INSTRUCTIONS = """
You also have a `find_papers` tool. It searches the list of PLATO publications
by title and abstract, and filters by author surname and publication year. Use
it for questions about the papers rather than their content: which papers
there are on a topic, who wrote what, what was published when or most
recently, how many. The passages above cannot answer such questions --- the
search behind them ignores authors and dates --- so call `find_papers` for them
even when passages were given. It returns one record per paper (title,
authors, date, abstract), numbered like passages; cite the records you use in
the same way. When you list papers, give each with its authors, year and
title. Report the number of matching papers it gives you, and say so when you
show only some of them.
"""

ROUTER_PROMPT = """\
You are a routing agent for a PLATO chatbot.
Decide which information sources the answering agent should consult before
replying to the user.

Sources available:
  - platochat_info: A description of the chatbot itself and some general infomation about PLAT and PLATO-Pub from the PLATO-pub website. Use when the
    user asks about the bot itself or its immediate context.
  - plato_publications: a searchable index over the published articles on ESA's PLATO mission. Use for all technical or scientific questions.

You may select neither, one, or both. If the user message is purely
conversational (greeting, thanks, a clarification about the previous answer)
and needs no new information, select neither.

Examples:
  User: "Hi, how are you?"
  -> platochat_info=false, search_query=""

  User: "What LLM do you use?"
  -> platochat_info=true, search_query=""

  User: "Will PLATO find earth-sized exoplanets?"
  -> platochat_info=false, search_query="earth-sized exoplanets"

  User: "When will PLATO launch? Do you have the most up-to-date infomation?"
  -> platochat_info=true, search_query="PLATO lauch date"

Preceding conversation:
{conversation}

User message: {user_input}
"""


class RouterDecision(BaseModel):
    """Which information sources to consult before answering."""

    platochat_info: bool = Field(
        description=(
            "Set to true if the information about the chatbot an its immediate context should be consulted."
        ),
    )
    search_query: str = Field(
        default="",
        description=(
            "A self-contained English search query over the articles "
            "published on ESA's PLATO Mission, or an empty string if no search is "
            "needed. Make sure to take the preceding conversation into account "
            "when formulating the search query, especially the previous human message."
        ),
    )


# ---------------------------------------------------------------------------
# Model loading (done once at import)
# ---------------------------------------------------------------------------

langfuse_client = Langfuse()

print("Loading language model …")
client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=OPENWEBUI_API_KEY,
)

# Instructor-wrapped client for structured output. Wraps the existing client
# (not a fresh one via from_provider) so Langfuse tracing still applies.
router_client = instructor.from_openai(client, mode=instructor.Mode.JSON)

print(f"Loading embedding model ({EMBED_MODEL} on {EMBED_DEVICE}) …")
embed_model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)

print(f"Loading cross-encoder ({RERANK_MODEL} on {EMBED_DEVICE}) …")
cross_encoder = CrossEncoder(RERANK_MODEL, device=EMBED_DEVICE)


# ---------------------------------------------------------------------------
# Weaviate
# ---------------------------------------------------------------------------

print("Connecting to Weaviate …")
weaviate_client = weaviate.connect_to_local()
assert weaviate_client.is_ready(), "Weaviate is not ready!"
chunks_collection = weaviate_client.collections.get(WEAVIATE_COLLECTION)
# Without the paper index the chatbot still works; it just does not offer
# find_papers. Build it with ingest/paper_index.py (or `make update`).
if weaviate_client.collections.exists(PAPER_COLLECTION):
    papers_collection = weaviate_client.collections.get(PAPER_COLLECTION)
else:
    papers_collection = None
    print(f"No paper index '{PAPER_COLLECTION}': find_papers is not offered.")


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

# How a passage is identified, in three places that must not be confused:
#   chunk_id   stored in Weaviate, e.g. "2021A&A...653A..98M#0017". Stable: it
#              depends only on the paper and the position in it. Used for
#              dedup, traces and the eval -- never shown to the model.
#   passage    1, 2, 3 ... in the order the passages of *this request* were
#              handed to the model. This is what it cites. Small numbers are
#              easy to copy correctly; a bibcode full of dots and ampersands
#              is not.
#   [1], [2]   what the user sees, numbered by first use in the answer.


def vectorize(text: str):
    """Embed one string. Must match how reindex_weaviate.py embedded the corpus."""
    # Normalised because bge-m3 is trained for cosine similarity. Weaviate's
    # default cosine metric normalises internally too, so this changes no
    # ranking today -- it keeps the vectors correct if the collection is ever
    # rebuilt with a dot-product metric, where unit length does matter.
    return embed_model.encode([text], normalize_embeddings=True)[0]


def _doc_summary(doc, extra: dict | None = None) -> dict:
    """Small JSON-friendly view of a doc for Langfuse traces."""
    props = doc.properties
    out = {
        "chunk_id":        props.get("chunk_id") or "paper:" + (props.get("bibcode") or ""),
        "title":           props.get("title"),
        "pubdate":         props.get("pubdate"),
        "section_headers": props.get("section_headers"),
    }
    if extra:
        out.update(extra)
    return out


def _union(vector_hits: list, bm25_hits: list) -> tuple[list, dict]:
    """Merge both candidate lists, deduped by UUID, recording who found what.

    Deliberately no fused score (no RRF, no alpha): the cross-encoder below
    rescores every candidate from scratch, so a blended rank would be computed
    and then discarded microseconds later. The union's only job is to decide
    which candidates reach the reranker, and taking it whole is what keeps a
    chunk that one retriever ranked first and the other never saw -- precisely
    the case hybrid search exists for, and the one a fused list truncated at
    `limit` can still drop.

    found_by is for the traces, not for ranking: it is how we tell whether BM25
    is earning the rerank budget it costs.
    """
    merged, found_by = [], {}
    for label, hits in (("vector", vector_hits), ("bm25", bm25_hits)):
        for doc in hits:
            key = str(doc.uuid)
            if key in found_by:
                found_by[key].append(label)
            else:
                found_by[key] = [label]
                merged.append(doc)
    return merged, found_by


def fetch_candidates(query: str) -> tuple[list, dict]:
    """First stage: the vector and BM25 searches, unioned.

    Returns ``(candidates, found_by)``. Separate from :func:`rerank` so that
    eval/run_eval.py can tell a paper the searches never surfaced from one the
    reranker then threw away -- two failures with different fixes.
    """
    query_vector = vectorize(query)

    with langfuse_client.start_as_current_observation(name="vector-search") as vs_obs:
        vs_obs.update(input={"query": query, "limit": RETRIEVAL_LIMIT})
        response = chunks_collection.query.near_vector(
            near_vector=query_vector,
            limit=RETRIEVAL_LIMIT,
            return_metadata=wq.MetadataQuery(distance=True),
        )
        vector_hits = response.objects
        vs_obs.update(output={
            "n_candidates": len(vector_hits),
            "candidates": [
                _doc_summary(d, {"distance": getattr(d.metadata, "distance", None)})
                for d in vector_hits
            ],
        })

    # Lexical half of the hybrid. bge-m3 embeds camera designations (N-CAM,
    # F-CAM), ESA document numbers, star identifiers and author names into
    # rough neighbourhoods rather than matching them; BM25 matches them
    # exactly, which is most of what it is here for.
    with langfuse_client.start_as_current_observation(name="bm25-search") as bm_obs:
        bm_obs.update(input={"query": query, "limit": RETRIEVAL_LIMIT,
                             "properties": BM25_PROPERTIES})
        response = chunks_collection.query.bm25(
            query=query,
            query_properties=BM25_PROPERTIES,
            limit=RETRIEVAL_LIMIT,
            return_metadata=wq.MetadataQuery(score=True),
        )
        bm25_hits = response.objects
        bm_obs.update(output={
            "n_candidates": len(bm25_hits),
            "candidates": [
                _doc_summary(d, {"score": getattr(d.metadata, "score", None)})
                for d in bm25_hits
            ],
        })

    return _union(vector_hits, bm25_hits)


def rerank(query: str, retrieved: list, found_by: dict) -> list:
    """Second stage: cross-encoder scores every candidate; keep the best few."""
    if not retrieved:
        return []

    with langfuse_client.start_as_current_observation(name="rerank") as rr_obs:
        n_by = {label: sum(1 for v in found_by.values() if label in v)
                for label in ("vector", "bm25")}
        rr_obs.update(input={
            "query":        query,
            "n_candidates": len(retrieved),
            "n_vector":     n_by["vector"],
            "n_bm25":       n_by["bm25"],
            "n_overlap":    sum(1 for v in found_by.values() if len(v) > 1),
        })
        cross_inp    = [[query, d.properties["page_content"]] for d in retrieved]
        cross_scores = cross_encoder.predict(cross_inp)

        scored   = list(zip(cross_scores, retrieved))
        positive = [(s, d) for s, d in scored if s > RERANK_THRESHOLD]

        if positive:
            reranked = sorted(positive, key=lambda t: t[0], reverse=True)[:RERANK_TOP_K]
            fallback = False
        else:
            reranked = sorted(scored, key=lambda t: t[0], reverse=True)[:RERANK_FALLBACK_K]
            fallback = True

        rr_obs.update(output={
            "n_selected":    len(reranked),
            "fallback_used": fallback,
            "selected":      [
                _doc_summary(d, {"score":    float(s),
                                 "found_by": found_by[str(d.uuid)]})
                for s, d in reranked
            ],
        })
        return [d for _, d in reranked]


def retrieve_docs(query: str) -> list:
    return rerank(query, *fetch_candidates(query))


def format_docs(docs: list, first_number: int = 1) -> str:
    """Passages as the model sees them, numbered from ``first_number``."""
    if not docs:
        return "No relevant documents were found."
    parts = []
    for number, d in enumerate(docs, start=first_number):
        props = d.properties
        meta = {
            "passage":   number,
            "paper":     props.get("short_ref"),
            "published": props.get("pubdate"),
            "title":     props.get("title"),
            "section":   props.get("section_headers"),
        }
        # Said only where it matters, so full-text passages look as before:
        # the model must not take an abstract for everything a paper says.
        if props.get("coverage") == "abstract only":
            meta["coverage"] = "abstract only"
        parts.append(json.dumps(meta, indent=4, ensure_ascii=False)
                     + "\n" + props["page_content"])
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# The paper index (find_papers)
# ---------------------------------------------------------------------------

# Must match fold_name() in ingest/paper_index.py, which made the keys.
_NAME_FOLD = str.maketrans({"ø": "o", "æ": "ae", "œ": "oe", "ß": "ss", "ł": "l", "đ": "d",
                            "ð": "d", "þ": "th", "ı": "i"})


def author_key(name: str) -> str:
    """A surname as the paper index stores it: 'Børsen-Koch' -> 'borsen koch'.

    Given "Surname, Initials" (the ADS form), only the surname is kept.
    """
    name = unicodedata.normalize("NFKD", name.split(",")[0].casefold().translate(_NAME_FOLD))
    name = "".join(c for c in name if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in name).split())


def is_paper(doc) -> bool:
    """A paper record from find_papers, as opposed to a passage (a chunk)."""
    return "chunk_id" not in doc.properties


def doc_key(doc) -> str:
    """What makes a passage or paper record the same one twice in a request."""
    props = doc.properties
    return "paper:" + props["bibcode"] if is_paper(doc) else props["chunk_id"]


def _year(value) -> int | None:
    try:
        return int(str(value).strip()[:4]) if value not in (None, "") else None
    except ValueError:
        return None


def _paper_filters(key: str, first_author_only: bool, year_from, year_to):
    parts = []
    if key:
        prop = "first_author_keys" if first_author_only else "author_keys"
        parts.append(wq.Filter.by_property(prop).contains_any([key]))
    if year_from is not None:
        parts.append(wq.Filter.by_property("year").greater_or_equal(year_from))
    if year_to is not None:
        parts.append(wq.Filter.by_property("year").less_or_equal(year_to))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else wq.Filter.all_of(parts)


@dataclass
class PaperSearch:
    """What one find_papers call found, and how."""

    papers: list
    total: int                 # papers matching the filters, before any ranking
    ranked: bool               # by relevance to a query, else newest first
    weak: bool = False         # nothing passed the reranker; these are the closest
    author_key: str = ""       # the surname actually filtered on


def find_papers(query: str = "", author: str = "", first_author_only: bool = False,
                year_from=None, year_to=None, limit=None) -> PaperSearch:
    """Search the paper index: by topic, author and year, in any combination.

    With a query, papers are found by vector and BM25 search over title and
    abstract and reranked like chunks; without one, every paper matching the
    filters is listed, newest first. ``total`` counts the papers that match the
    filters, so the model can say "37 papers, here are ten".

    An author given with first names ("Mikkel Lund") matches no surname key,
    so if the whole string finds nothing, its last word is tried.
    """
    query = (query or "").strip()
    year_from, year_to = _year(year_from), _year(year_to)
    try:
        limit = int(limit) if limit else PAPER_DEFAULT_LIMIT
    except (TypeError, ValueError):
        limit = PAPER_DEFAULT_LIMIT
    limit = max(1, min(limit, PAPER_MAX_LIMIT))

    key = author_key(author or "")
    filters = _paper_filters(key, first_author_only, year_from, year_to)
    total = papers_collection.aggregate.over_all(filters=filters, total_count=True).total_count
    if key and not total and " " in key:
        key = key.split()[-1]
        filters = _paper_filters(key, first_author_only, year_from, year_to)
        total = papers_collection.aggregate.over_all(filters=filters, total_count=True).total_count
    if not total:
        return PaperSearch([], 0, bool(query), author_key=key)

    if not query:
        response = papers_collection.query.fetch_objects(
            filters=filters, limit=limit,
            sort=wq.Sort.by_property("pub_date", ascending=False),
        )
        return PaperSearch(response.objects, total, False, author_key=key)

    vector_hits = papers_collection.query.near_vector(
        near_vector=vectorize(query), filters=filters, limit=PAPER_RETRIEVAL_LIMIT,
    ).objects
    bm25_hits = papers_collection.query.bm25(
        query=query, query_properties=PAPER_BM25_PROPERTIES, filters=filters,
        limit=PAPER_RETRIEVAL_LIMIT,
    ).objects
    candidates, _ = _union(vector_hits, bm25_hits)
    if not candidates:
        return PaperSearch([], total, True, author_key=key)
    scores = cross_encoder.predict([
        [query, d.properties["title"] + "\n\n" + (d.properties.get("abstract") or "")]
        for d in candidates
    ])
    scored = sorted(zip(scores, candidates), key=lambda t: t[0], reverse=True)
    passing = [d for s, d in scored if s > RERANK_THRESHOLD][:limit]
    if passing:
        return PaperSearch(passing, total, True, author_key=key)
    return PaperSearch([d for _, d in scored[:PAPER_FALLBACK_K]], total, True, weak=True,
                       author_key=key)


def _authors_line(authors: list[str], key: str) -> str:
    """The first few authors -- and, if the search was for someone further down, them too."""
    shown = authors[:PAPER_SHOWN_AUTHORS]
    line = "; ".join(shown)
    rest = len(authors) - len(shown)
    if rest > 0:
        line += f"; and {rest} more"
        # The index's rule: the key is a run of whole words of the surname.
        wanted = [a for a in authors[PAPER_SHOWN_AUTHORS:]
                  if key and f" {key} " in f" {author_key(a)} "]
        if wanted:
            line += " (among them " + "; ".join(wanted[:3]) + ")"
    return line


def format_papers(papers: list, first_number: int, author_key_: str = "") -> str:
    """Paper records as the model sees them, numbered on from the passages."""
    short = len(papers) > PAPER_FULL_ABSTRACTS
    parts = []
    for number, d in enumerate(papers, start=first_number):
        props = d.properties
        meta = {
            "passage":   number,
            "paper":     props.get("short_ref"),
            "published": props.get("pubdate"),
            "title":     props.get("title"),
            "authors":   _authors_line(props.get("authors") or [], author_key_),
            "journal":   props.get("journal"),
            # Whether search_publications can find more than this abstract.
            "indexed":   props.get("coverage"),
        }
        abstract = props.get("abstract") or "(no abstract)"
        if short and len(abstract) > PAPER_SHORT_ABSTRACT:
            abstract = abstract[:PAPER_SHORT_ABSTRACT].rsplit(" ", 1)[0] + " …"
        parts.append(json.dumps(meta, indent=4, ensure_ascii=False) + "\nAbstract: " + abstract)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Citation / source helpers
# ---------------------------------------------------------------------------

_CITE_RE = re.compile(r"<cite>.*?</cite>", re.DOTALL)

# The GLM tiers return no visible chain-of-thought, so this is a no-op for the
# default models. It still runs because the model is env-overridable: MiniMax
# emits its reasoning inline, wrapped in <mm:think>…</mm:think>, and would
# otherwise leak it into the answer. Models differ in how they tag reasoning:
# plain <think>, or namespaced like MiniMax's <mm:think>, so accept an optional
# "prefix:" on both open and close. Also handle a stray opening tag with no
# close (truncated reasoning) by dropping everything up to it.
_THINK_RE = re.compile(r"<(?:\w+:)?think>.*?</(?:\w+:)?think>\s*", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"^.*?<(?:\w+:)?think>", re.DOTALL | re.IGNORECASE)
_ANY_OPEN_THINK_RE = re.compile(r"<(?:\w+:)?think>", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    text = _THINK_RE.sub("", text)
    # If a lone, unclosed <think> remains, treat the rest as reasoning preamble.
    if _ANY_OPEN_THINK_RE.search(text):
        text = _OPEN_THINK_RE.sub("", text)
    return text.strip()


def _dedup(seq: list) -> list:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def used_sources(answer: str) -> tuple[str, list[str]]:
    """Replace <cite>N</cite> tags with sequential [1], [2], … references."""
    numbers = _dedup(
        s.strip() for s in re.findall(r"<cite>(.*?)</cite>", answer, re.DOTALL)
    )
    for idx, num in enumerate(numbers, start=1):
        answer = re.sub(rf"<cite> ?{num} ?</cite>", f"[{idx}]", answer)
    return answer, numbers


def cited_docs(docs: list, source_numbers: list[str]) -> list:
    """The docs behind the cited passage numbers; None where a number is bogus."""
    out = []
    for num in source_numbers:
        n = int(num) if num.isdigit() else 0
        out.append(docs[n - 1] if 1 <= n <= len(docs) else None)
    return out


def _without_title(headers: list[str], title: str) -> list[str]:
    """Section path minus the paper's own title.

    The outermost header of a converted paper is its title, which the sources
    line already shows -- and in the markdown it often drags a footnote along
    ("The all-sky PLATO input catalogue^(†)thanks: The catalogue described in
    this article is only available ..."). Matched on a shared opening, because
    the header may be the title cut short ("Transit least-squares survey") or
    the title with debris after it.
    """
    if not headers:
        return headers
    first, full = headers[0].casefold(), title.casefold()
    n = min(len(first), len(full), 25)
    return headers[1:] if n >= 10 and first[:n] == full[:n] else headers


# ![alt](path) and [text](url); the alt text may hold escaped brackets, as in
# LaTeXML's ORCID badges: [![\[Uncaptioned image\]](x1.png)](https://orcid...).
_MD_IMAGE_RE = re.compile(r"!\[(?:\\.|[^\]\\])*\]\([^)]*\)")
_MD_LINK_RE  = re.compile(r"\[((?:\\.|[^\]\\])*)\]\([^)]*\)")


def _snippet(text: str, limit: int = 60) -> str:
    """The opening of a passage as one plain line, for a passage with no section.

    The sources line is itself markdown, so raw passage text cannot go into it
    as it stands: a blank line ends the citation and starts a new paragraph, a
    '#' starts a heading in the middle of it. Images go, links keep their
    text, heading markers go, and all whitespace becomes single spaces.
    """
    text = _MD_IMAGE_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = re.sub(r"^\s*#+\s+", "", text, flags=re.M)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def build_sources_text(docs: list, source_numbers: list[str]) -> str:
    no_refs = (
        "_The information presented here does not explicitly reference the "
        "retrieved sources. Extra caution with respect to accuracy may be "
        "in order._"
    )
    if not docs or not source_numbers:
        return no_refs
    lines = []
    for idx, doc in enumerate(cited_docs(docs, source_numbers), start=1):
        if doc is None:
            continue
        props   = doc.properties
        title   = props.get("title") or "Unknown title"
        # A paper record from find_papers is its title and abstract.
        headers = (["Abstract"] if is_paper(doc)
                   else _without_title(props.get("section_headers") or [], title))
        section = (
            "Section: " + ", ".join(headers)
            if headers
            else _snippet(props["page_content"])
        )
        # The title links to the paper's ADS page, the same place the PLATO-Pub
        # list sends people; from there they reach the publisher through their
        # own institution's access. arXiv is linked directly where it exists,
        # because that copy is free to read.
        if props.get("link"):
            title = f"[{title}]({props['link']})"
        line = f"[{idx}] {props.get('short_ref') or ''} *{title}*".replace("  ", " ")
        # The year is already in the short reference; add the month if known.
        if len(props.get("pubdate") or "") > 4:
            line += f", published {props['pubdate']}"
        line += f" — {section}"
        extra = []
        if props.get("doi"):
            extra.append(f"[publisher](https://doi.org/{props['doi']})")
        if props.get("arxiv_id"):
            extra.append(f"[arXiv](https://arxiv.org/abs/{props['arxiv_id']})")
        if extra:
            line += " (" + " · ".join(extra) + ")"
        lines.append(line)
    if not lines:
        return no_refs
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Tools exposed to the model
# ---------------------------------------------------------------------------

PLATOCHAT_INFO_PATH = os.environ.get(
    "PLATOCHAT_INFO_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "platochat_info.md"),
)

print(f"Loading information from {PLATOCHAT_INFO_PATH} …")
with open(PLATOCHAT_INFO_PATH, "r", encoding="utf-8") as _fh:
    PLATOCHAT_INFO = _fh.read()


def tool_get_platochat_info() -> str:
    return PLATOCHAT_INFO


def tool_find_papers(**kwargs) -> PaperSearch:
    with langfuse_client.start_as_current_observation(name="find-papers") as obs:
        obs.update(input=kwargs)
        found = find_papers(**kwargs)
        obs.update(output={
            "total": found.total, "ranked": found.ranked, "weak": found.weak,
            "author_key": found.author_key,
            "papers": [_doc_summary(d) for d in found.papers],
        })
    return found


def tool_search_publications(query: str) -> tuple[str, list]:
    """Run the RAG retrieval. Returns (formatted_context, raw_docs)."""
    with langfuse_client.start_as_current_observation(name="retrieve-and-rerank") as obs:
        obs.update(input={"query": query})
        docs = retrieve_docs(query)
        obs.update(output={"n_docs": len(docs)})
    return format_docs(docs), docs


# Offered to the *answering* model only (the router picks the first search
# itself, via RouterDecision). Every model on the endpoint honours
# tool_choice="auto" through the OpenWebUI/vLLM proxy and correctly declines to
# call this on conversational turns, so it is offered on every turn: it doubles
# as the recovery path for a router that wrongly decided no search was needed.
# GLM-max may request several searches in one turn; the answer loop below runs
# them all, so parallel calls cost one round of the follow-up budget, not one
# per call.
SEARCH_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_publications",
            "description": (
                "Search the index of published articles on ESA's PLATO mission. "
                "Use only when the passages you were already given do not answer "
                "the question. The query must be self-contained English, with all "
                "references to the conversation resolved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A self-contained search query over the PLATO literature."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# The paper index. Offered next to search_publications whenever the paper
# collection exists; both draw on the same follow-up budget.
FIND_PAPERS_TOOL = {
    "type": "function",
    "function": {
        "name": "find_papers",
        "description": (
            "Find papers on the PLATO-Pub publication list by topic, author and "
            "publication year, in any combination. Searches titles and abstracts, "
            "not the full texts, and returns one record per paper: title, authors, "
            "date, journal, abstract. Use it for questions about the papers "
            "themselves -- which papers exist on a topic, who wrote what, what was "
            "published when, how many -- which search_publications cannot answer, "
            "as it does not look at authors or dates. Without a query, lists the "
            "papers matching the filters, newest first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Topic to search titles and abstracts for, in English. "
                        "Leave empty to list papers by author and/or year alone."
                    ),
                },
                "author": {
                    "type": "string",
                    "description": (
                        "Surname of one author, e.g. 'Rauer' or 'Christensen-Dalsgaard'. "
                        "Surname only, no first names or initials."
                    ),
                },
                "first_author_only": {
                    "type": "boolean",
                    "description": "Only papers with that author as first author.",
                },
                "year_from": {"type": "integer", "description": "Earliest publication year."},
                "year_to": {"type": "integer", "description": "Latest publication year."},
                "limit": {
                    "type": "integer",
                    "description": (
                        f"How many papers to return (default {PAPER_DEFAULT_LIMIT}, "
                        f"at most {PAPER_MAX_LIMIT})."
                    ),
                },
            },
        },
    },
}


def available_tools() -> list:
    return SEARCH_TOOL_SCHEMA + ([FIND_PAPERS_TOOL] if papers_collection is not None else [])


def run_search_tool_call(call: dict, given: list) -> tuple[str, list]:
    """Execute one model-requested tool call: search_publications or find_papers.

    ``given`` is every passage and paper record handed to the model so far in
    this request, in order; a record's passage number is its position there.
    Returns ``(tool_message_content, new_docs)``, the new ones numbered on from
    ``len(given) + 1``. Nothing is handed over twice: re-sending would waste
    context and let the same source be cited under two numbers.

    Never raises: a bad tool call is reported back to the model as text so it can
    recover on the next turn rather than taking the request down.
    """
    name = call["function"]["name"]
    if name not in {t["function"]["name"] for t in available_tools()}:
        return f"Error: no tool named {name!r} is available.", []

    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return "Error: the tool arguments were not valid JSON. Try again.", []
    if not isinstance(args, dict):
        return "Error: the tool arguments must be a JSON object. Try again.", []

    number_of = {doc_key(d): n for n, d in enumerate(given, start=1)}
    if name == "find_papers":
        try:
            return _run_find_papers(args, number_of, len(given) + 1)
        except Exception as exc:
            print(f"[find papers] failed: {type(exc).__name__}: {exc}")
            return ("Error: find_papers failed with these arguments. Check them "
                    "(query and author are strings, years are integers) and try "
                    "again, or answer without it."), []

    query = (args.get("query") or "").strip()
    if not query:
        return "Error: 'query' is required and must be a non-empty string.", []

    print(f"[follow-up search] {query!r}")
    _, docs = tool_search_publications(query)
    fresh = [d for d in docs if doc_key(d) not in number_of]

    if not fresh:
        return (
            "That search returned only passages you have already been given. "
            "Answer with what you have, and say what is missing."
        ), []

    return (
        "The following passages were retrieved from the publications. Cite each "
        "one you use with <cite>N</cite> tags, where N is the passage number shown "
        "in the metadata (e.g. <cite>9</cite>):\n\n"
        + format_docs(fresh, len(given) + 1)
    ), fresh


def _run_find_papers(args: dict, number_of: dict, first_number: int) -> tuple[str, list]:
    kwargs = {k: args.get(k) for k in ("query", "author", "first_author_only",
                                       "year_from", "year_to", "limit")
              if args.get(k) not in (None, "")}
    kwargs["first_author_only"] = bool(kwargs.get("first_author_only"))
    print(f"[find papers] {kwargs}")
    found = tool_find_papers(**kwargs)

    # Say back what was searched for, as the index understood it.
    what = []
    if kwargs.get("query"):
        what.append(f"on {kwargs['query']!r}")
    if found.author_key:
        # As asked, unless only its last word matched ("Mikkel Lund" -> Lund).
        asked = kwargs.get("author", "")
        name = asked if author_key(asked) == found.author_key else found.author_key.title()
        what.append(("with first author " if kwargs["first_author_only"] else "by ") + name)
    years = [str(y) for y in (_year(kwargs.get("year_from")), _year(kwargs.get("year_to")))]
    if years != ["None", "None"]:
        what.append("published " + "-".join(y if y != "None" else "" for y in years))
    what = " ".join(what) or "(no filters)"
    filtered = bool(found.author_key or years != ["None", "None"])
    try:
        limit = max(1, min(int(kwargs.get("limit") or PAPER_DEFAULT_LIMIT), PAPER_MAX_LIMIT))
    except (TypeError, ValueError):
        limit = PAPER_DEFAULT_LIMIT

    if not found.papers:
        if found.total == 0:
            return (f"No paper on the PLATO-Pub list matches: {what}. The list holds only "
                    "papers about the PLATO mission, so an author or topic may simply not "
                    "be on it. Say so; do not guess other papers."), []
        return (f"{found.total} papers match the filters ({what}), but none of them is "
                "about the query. Say so, or search again without the query."), []

    old = [number_of[doc_key(d)] for d in found.papers if doc_key(d) in number_of]
    fresh = [d for d in found.papers if doc_key(d) not in number_of]

    if found.weak:
        head = (f"No paper is clearly about this ({what}); these {len(found.papers)} are "
                "the closest, and may not be relevant.")
    elif found.ranked:
        head = f"{len(found.papers)} paper(s) {what}, most relevant first"
        if filtered:
            head += f" (of the {found.total} matching the author/year filters)"
        head += ":"
        # Capped rather than exhausted: more may be relevant.
        if len(found.papers) == limit:
            head += " There may be more; ask for a higher limit to see them."
    else:
        head = (f"{found.total} papers on the PLATO-Pub list match ({what}); "
                + ("all of them" if len(found.papers) == found.total
                   else f"the {len(found.papers)} most recent")
                + ", newest first:")
        if len(found.papers) < found.total:
            head += " Ask for a higher limit, or narrow the filters, to see the others."
    if old:
        head += (" Already given to you above, and part of this result: passage(s) "
                 + ", ".join(map(str, old)) + ".")
    if not fresh:
        return head, []
    return (
        head + "\n\nEach record is one paper: its title, authors, date and abstract. "
        "Cite a record you use with <cite>N</cite> like a passage. An abstract does not "
        "say everything a paper contains; for details, use search_publications.\n\n"
        + format_papers(fresh, first_number, found.author_key)
    ), fresh


def route(user_input: str, prev_conv: str) -> RouterDecision:
    """One LLM call that decides which sources to consult.

    Returns a validated RouterDecision. On failure (instructor exhausts
    retries), falls back to running a search with the raw user input so the
    bot still produces a grounded answer.
    """
    prompt = ROUTER_PROMPT.format(
        conversation=prev_conv or "(none)",
        user_input=user_input,
    )
    router_kwargs = dict(
        model=ROUTER_MODEL,
        response_model=RouterDecision,
        max_retries=2,
        messages=[
            {"role": "system", "content": "You are a routing agent."},
            {"role": "user", "content": prompt},
        ],
        name="route",
    )
    try:
        if FORCE_STREAM:
            # create_partial() streams, which keeps the proxy-injected
            # stream_options valid. Take the last partial and re-validate it as
            # a full RouterDecision so a truncated stream fails loudly here and
            # is caught by the fallback below.
            partial = None
            for partial in router_client.chat.completions.create_partial(**router_kwargs):
                pass
            if partial is None:
                raise RuntimeError("router stream yielded no object")
            decision = RouterDecision.model_validate(partial.model_dump())
        else:
            decision = router_client.chat.completions.create(**router_kwargs)
    except Exception as exc:
        print(f"[router] structured-output call failed ({exc}) — falling back to search with user input")
        return RouterDecision(platochat_info=False, search_query=user_input)

    decision.search_query = decision.search_query.strip()
    print(f"[router decision] {decision.model_dump()}")
    return decision


# ---------------------------------------------------------------------------
# Core agent pipeline (UI-agnostic)
# ---------------------------------------------------------------------------

def complete(messages: list[dict], tools: list | None = None) -> tuple[str, list[dict]]:
    """One call to the answering model. Returns ``(text, tool_calls)``.

    ``text`` has the model's reasoning block stripped; ``tool_calls`` comes back
    in wire format, ready to append to ``messages`` as an assistant turn.

    A reasoning model (e.g. MiniMax, if ANSWER_MODEL is overridden) streams its
    reasoning as ordinary content deltas *before* emitting a tool call, and sets
    ``content: null`` on the tool-call chunk itself, so strip_reasoning() still
    does the right thing here. Arguments arrive whole for short calls, but vLLM
    may split longer ones, so they are accumulated by index.
    """
    kwargs = dict(model=ANSWER_MODEL, messages=messages, name="answer")
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    if not FORCE_STREAM:
        message = client.chat.completions.create(**kwargs).choices[0].message
        calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in (message.tool_calls or [])
        ]
        return strip_reasoning(message.content or ""), calls

    # See FORCE_STREAM above: stream and reassemble the deltas.
    parts: list[str] = []
    acc: dict[int, dict] = {}
    for chunk in client.chat.completions.create(stream=True, **kwargs):
        if not chunk.choices:
            continue                       # usage-only chunk at end of stream
        delta = chunk.choices[0].delta
        if delta.content:
            parts.append(delta.content)
        for tc in delta.tool_calls or []:
            slot = acc.setdefault(
                tc.index,
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["function"]["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                slot["function"]["arguments"] += tc.function.arguments

    return strip_reasoning("".join(parts)), [acc[i] for i in sorted(acc)]


@dataclass
class Answer:
    """Everything one request produced, not just the two strings the UI shows.

    ``reply`` and ``sources`` are what :func:`answer_question` hands to the
    front-end. The rest is the working: what the router decided, every passage
    the model was given (first search and follow-ups, in the order given),
    which of those it actually cited, the tool calls the answering model made,
    and where the time went. eval/run_eval.py scores on these.
    """

    reply: str | None = None
    sources: str | None = None
    decision: RouterDecision | None = None
    docs: list = field(default_factory=list)
    cited: list = field(default_factory=list)
    # {"round": 1, "name": "find_papers", "arguments": "{...}"} per call; calls
    # made together share a round, and each round is one more model call.
    tool_calls: list = field(default_factory=list)
    # Wall-clock seconds in the router call, in the searches (the first and
    # every follow-up), and in the answering model's calls.
    seconds: dict = field(default_factory=dict)


@contextmanager
def _timed(seconds: dict, key: str):
    """Add the time spent in the block to ``seconds[key]``."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        seconds[key] = round(seconds.get(key, 0.0) + time.perf_counter() - t0, 2)


def answer_question(
    user_input: str,
    history: list[dict] | None = None,
) -> tuple[str | None, str | None]:
    """Route → execute tools → generate answer (searching again if needed).

    The router picks the first search; the answering model may then request up
    to ``FOLLOWUP_SEARCHES`` more rounds of tool calls when the passages it was
    handed do not cover the question: ``search_publications`` for more passages,
    ``find_papers`` for paper records (by topic, author and year).

    Parameters
    ----------
    user_input:
        The user's latest message.
    history:
        Prior turns as ``[{"role": "human"|"ai", "content": str}, ...]`` (most
        recent last). Only the last ``HISTORY_WINDOW`` are folded into the
        prompt for context.

    Returns
    -------
    (reply, sources):
        ``reply`` is markdown for the assistant bubble (citation tags already
        rewritten to ``[1]`` / stripped). ``sources`` is markdown for the "See
        sources" panel, or ``None`` when no search was run. ``(None, None)`` if
        the model returned an empty answer.
    """
    answer = answer_question_detailed(user_input, history)
    return answer.reply, answer.sources


def answer_question_detailed(
    user_input: str,
    history: list[dict] | None = None,
) -> Answer:
    """:func:`answer_question`, returning the full :class:`Answer`."""
    history = history or []
    result = Answer()

    with langfuse_client.start_as_current_observation(name="agent-query") as root_obs:
        root_obs.update(input=user_input)

        prev_conv = "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}" for m in history[-HISTORY_WINDOW:]
        )

        # 1. Route: decide which tools to call
        with langfuse_client.start_as_current_observation(name="route") as route_obs:
            route_obs.update(input={"user_input": user_input})
            with _timed(result.seconds, "route"):
                decision = route(user_input, prev_conv)
            route_obs.update(output=decision.model_dump())

        # 2. Execute the selected tools
        platochat_info_text = None
        retrieved_docs: list = []

        if decision.platochat_info:
            platochat_info_text = tool_get_platochat_info()

        if decision.search_query:
            with _timed(result.seconds, "search"):
                _, retrieved_docs = tool_search_publications(decision.search_query)

        # 3. Generate the final answer
        blocks: list[str] = []
        if platochat_info_text:
            blocks.append(
                "The following is the information about the chatbot and website:\n\n"
                + platochat_info_text
            )
        if retrieved_docs:
            blocks.append(
                "The following passages were retrieved from the publications. "
                "Cite each one you use with <cite>N</cite> tags, where N is the "
                "passage number shown in the metadata (e.g. <cite>3</cite>):\n\n"
                + format_docs(retrieved_docs)
            )
        if prev_conv:
            blocks.append("Preceding conversation:\n" + prev_conv)
        blocks.append("User message: " + user_input)

        answer_messages = [
            {
                "role": "system",
                "content": system_prompt
                + SEARCH_TOOL_INSTRUCTIONS.format(n=FOLLOWUP_SEARCHES)
                + (FIND_PAPERS_INSTRUCTIONS if papers_collection is not None else ""),
            },
            {"role": "user", "content": "\n\n---\n\n".join(blocks)},
        ]

        # The model may ask for more context instead of answering. Loop until it
        # answers or spends its budget; once the budget is gone we stop offering
        # the tool, which forces a plain answer and terminates the loop.
        searches_left = FOLLOWUP_SEARCHES
        raw_answer = ""

        while True:
            with _timed(result.seconds, "answer"):
                raw_answer, tool_calls = complete(
                    answer_messages,
                    tools=available_tools() if searches_left > 0 else None,
                )
            if not tool_calls:
                break

            answer_messages.append({
                "role": "assistant",
                "content": raw_answer or None,
                "tool_calls": tool_calls,
            })
            with langfuse_client.start_as_current_observation(
                name="follow-up-search"
            ) as fs_obs:
                fs_obs.update(input={"calls": [c["function"] for c in tool_calls]})
                n_before = len(retrieved_docs)
                for call in tool_calls:
                    result.tool_calls.append({
                        "round": FOLLOWUP_SEARCHES - searches_left + 1, **call["function"],
                    })
                    # retrieved_docs is the request's running passage list:
                    # a passage's number is its position in it, so follow-up
                    # passages continue where the last search stopped.
                    with _timed(result.seconds, "search"):
                        tool_content, new_docs = run_search_tool_call(call, retrieved_docs)
                    retrieved_docs.extend(new_docs)
                    answer_messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": tool_content,
                    })
                fs_obs.update(output={
                    "n_new_docs":    len(retrieved_docs) - n_before,
                    "searches_left": searches_left - 1,
                })
            searches_left -= 1

        result.decision, result.docs = decision, retrieved_docs
        if not raw_answer:
            return result

        # 4. Post-process: rewrite/strip citation tags, build the sources block.
        #    (No Streamlit-specific math rewriting — the browser renders LaTeX.)
        if retrieved_docs:
            result.reply, source_numbers = used_sources(raw_answer)
            result.sources = "**Sources**\n\n" + build_sources_text(retrieved_docs, source_numbers)
            result.cited = [d for d in cited_docs(retrieved_docs, source_numbers) if d]
        else:
            result.reply = _CITE_RE.sub("", raw_answer)

        return result
