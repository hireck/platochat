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
FOLLOWUP_SEARCHES    = int(os.environ.get("PLATO_FOLLOWUP_SEARCHES", "2"))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

system_prompt = """You are a PLATO expert at the ESA helpdesk. Your task is to help researchers learn about Plato and use Plato data products effectively in their work.

Be helpful and concise. Volunteer additional information where relevant, but
keep it brief. Do not make up facts that are not supported by the information
provided to you. If the supplied information is insufficient to answer, say
so plainly.

The user message may be accompanied by:
  - the information about the chatbot and website: a description of PLATO
    Chat and its sources, plus general information about the mission and
    useful links from the public PLATO-Pub website, and/or
  - passages retrieved from the index of published articles on ESA's PLATO
    mission, each with a `chunk_number` in its metadata.

If retrieved passages are provided, cite each passage you use by wrapping its
chunk_number in <cite>...</cite> tags, e.g. `<cite>23</cite>` --- only the
number inside the tags, and each cited chunk number in its own pair of tags.
Do not cite the information about the chatbot and website. Do not list
sources at the end; citations are inline only. Do not fabricate citations.
If NO retrieved passages are provided, do not emit any <cite> tags.

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
becomes "PLATO field of view"). You may search at most {n} more time(s); after
that, answer with what you have and state plainly what is missing.
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


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

META_FIELDS = ["chunk_number", "title", "section_headers", "parent_doc", "link"]


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
        "chunk_number":    props.get("chunk_number"),
        "title":           props.get("title"),
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


def retrieve_docs(query: str) -> list:
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

    retrieved, found_by = _union(vector_hits, bm25_hits)

    with langfuse_client.start_as_current_observation(name="rerank") as rr_obs:
        rr_obs.update(input={
            "query":        query,
            "n_candidates": len(retrieved),
            "n_vector":     len(vector_hits),
            "n_bm25":       len(bm25_hits),
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


def format_docs(docs: list) -> str:
    if not docs:
        return "No relevant documents were found."
    parts = []
    for d in docs:
        meta = {f: d.properties.get(f) for f in META_FIELDS}
        parts.append(json.dumps(meta, indent=4) + "\n" + d.properties["page_content"])
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


def build_sources_text(docs: list, source_numbers: list[str]) -> str:
    no_refs = (
        "_The information presented here does not explicitly reference the "
        "retrieved sources. Extra caution with respect to accuracy may be "
        "in order._"
    )
    if not docs or not source_numbers:
        return no_refs
    num2doc = {str(d.properties["chunk_number"]): d for d in docs}
    lines = []
    for idx, num in enumerate(source_numbers, start=1):
        doc = num2doc.get(num)
        if doc is None:
            continue
        title   = doc.properties.get("title", "Unknown title")
        headers = doc.properties.get("section_headers")
        section = (
            "Section: " + ", ".join(headers)
            if headers
            else doc.properties["page_content"][:60] + "…"
        )
        lines.append(f"[{idx}] *{title}* — {section}")
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


def run_search_tool_call(call: dict, seen_chunks: set) -> tuple[str, list]:
    """Execute one model-requested tool call.

    Returns ``(tool_message_content, new_docs)``. Passages already handed to the
    model are filtered out: ``chunk_number`` is a global running counter over the
    whole corpus (see ``get_present_papers.py``), so it is a safe dedup key
    across searches, and re-sending a chunk would waste context and let the same
    source be cited under two numbers.

    Never raises: a bad tool call is reported back to the model as text so it can
    recover on the next turn rather than taking the request down.
    """
    name = call["function"]["name"]
    if name != "search_publications":
        return f"Error: no tool named {name!r} is available.", []

    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return "Error: the tool arguments were not valid JSON. Try again.", []

    query = (args.get("query") or "").strip()
    if not query:
        return "Error: 'query' is required and must be a non-empty string.", []

    print(f"[follow-up search] {query!r}")
    _, docs = tool_search_publications(query)
    fresh = [d for d in docs if d.properties["chunk_number"] not in seen_chunks]
    seen_chunks.update(d.properties["chunk_number"] for d in fresh)

    if not fresh:
        return (
            "That search returned only passages you have already been given. "
            "Answer with what you have, and say what is missing."
        ), []

    return (
        "The following passages were retrieved from the publications. Cite each "
        "one you use with <cite>N</cite> tags, where N is the chunk_number shown "
        "in the metadata (e.g. <cite>38</cite>):\n\n" + format_docs(fresh)
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


def answer_question(
    user_input: str,
    history: list[dict] | None = None,
) -> tuple[str | None, str | None]:
    """Route → execute tools → generate answer (searching again if needed).

    The router picks the first search; the answering model may then request up
    to ``FOLLOWUP_SEARCHES`` more via the ``search_publications`` tool when the
    passages it was handed do not cover the question.

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
    history = history or []

    with langfuse_client.start_as_current_observation(name="agent-query") as root_obs:
        root_obs.update(input=user_input)

        prev_conv = "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}" for m in history[-HISTORY_WINDOW:]
        )

        # 1. Route: decide which tools to call
        with langfuse_client.start_as_current_observation(name="route") as route_obs:
            route_obs.update(input={"user_input": user_input})
            decision = route(user_input, prev_conv)
            route_obs.update(output=decision.model_dump())

        # 2. Execute the selected tools
        platochat_info_text = None
        retrieved_docs: list = []

        if decision.platochat_info:
            platochat_info_text = tool_get_platochat_info()

        if decision.search_query:
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
                "chunk_number shown in the metadata (e.g. <cite>38</cite>):\n\n"
                + format_docs(retrieved_docs)
            )
        if prev_conv:
            blocks.append("Preceding conversation:\n" + prev_conv)
        blocks.append("User message: " + user_input)

        answer_messages = [
            {
                "role": "system",
                "content": system_prompt
                + SEARCH_TOOL_INSTRUCTIONS.format(n=FOLLOWUP_SEARCHES),
            },
            {"role": "user", "content": "\n\n---\n\n".join(blocks)},
        ]

        # The model may ask for more context instead of answering. Loop until it
        # answers or spends its budget; once the budget is gone we stop offering
        # the tool, which forces a plain answer and terminates the loop.
        seen_chunks = {d.properties["chunk_number"] for d in retrieved_docs}
        searches_left = FOLLOWUP_SEARCHES
        raw_answer = ""

        while True:
            raw_answer, tool_calls = complete(
                answer_messages,
                tools=SEARCH_TOOL_SCHEMA if searches_left > 0 else None,
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
                    tool_content, new_docs = run_search_tool_call(call, seen_chunks)
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

        if not raw_answer:
            return None, None

        # 4. Post-process: rewrite/strip citation tags, build the sources block.
        #    (No Streamlit-specific math rewriting — the browser renders LaTeX.)
        if retrieved_docs:
            reply, source_numbers = used_sources(raw_answer)
            sources = "**Sources**\n\n" + build_sources_text(retrieved_docs, source_numbers)
        else:
            reply = _CITE_RE.sub("", raw_answer)
            sources = None

        return reply, sources
