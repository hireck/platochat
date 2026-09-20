"""Header-aware Markdown splitter with paragraph packing — standard library only.

Two stages, usable separately or together via :func:`split_markdown_packed`:

1. :class:`MarkdownHeaderSplitter` splits Markdown on ATX headers (#, ##, ...)
   and attaches the active header hierarchy to each chunk, so a retrieved chunk
   still knows which section / subsection it came from. A focused replacement
   for langchain's MarkdownHeaderTextSplitter.

2. :class:`ParagraphPacker` then packs each section's paragraphs into
   token-bounded chunks. Header splitting alone yields one chunk per paper
   section — 890 tokens at the median in the PLATO corpus, with a long tail into
   the tens of thousands — which is far too coarse a unit for vector retrieval.

No third-party dependencies, so nothing here breaks on a Python upgrade. The
packer takes an optional ``count_tokens`` callable so callers can supply their
embedding model's real tokenizer without this module importing one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A header line: 1-6 leading '#', at least one space, then text.
# Leading whitespace up to 3 spaces is allowed by the CommonMark spec.
_HEADER_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
# A fenced code block delimiter: ``` or ~~~ (3+), optionally with an info string.
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
# A paragraph break: a blank line (possibly carrying whitespace).
_PARA_RE = re.compile(r"\n[ \t]*\n")


@dataclass
class MarkdownChunk:
    """A chunk of markdown plus the header context it lives under."""

    content: str
    metadata: dict[str, str] = field(default_factory=dict)
    # The active header hierarchy, outermost first. Kept separate from
    # ``metadata`` so callers can add their own keys (parent_doc, chunk_number,
    # ...) without losing track of which values were headers.
    headers: list[str] = field(default_factory=list)
    # Position among the packed pieces of one section; both 1 when unpacked.
    part: int = 1
    n_parts: int = 1

    def __repr__(self) -> str:  # readable in logs / debugging
        meta = ", ".join(f"{k}={v!r}" for k, v in self.metadata.items())
        preview = (self.content[:50] + "…") if len(self.content) > 50 else self.content
        part = f", part={self.part}/{self.n_parts}" if self.n_parts > 1 else ""
        return f"MarkdownChunk({preview!r}, {{{meta}}}{part})"

    def header_path(self, sep: str = " > ") -> str:
        """Header hierarchy as one line, e.g. ``"Results > Transit depths"``.

        Useful as a prefix on the text handed to an embedding model: a chunk
        five paragraphs into a section otherwise carries no clue about what it
        is a section *of*.
        """
        return sep.join(h for h in self.headers if h)


class MarkdownHeaderSplitter:
    """Split markdown on configured header levels, tracking hierarchy.

    Parameters
    ----------
    headers_to_split_on:
        Iterable of (marker, metadata_key) pairs, e.g.
        [("#", "h1"), ("##", "h2"), ("###", "h3")].
        Only headers at these levels start a new chunk; their text is
        recorded under the given metadata key. Deeper headers reset when a
        shallower header appears (h1 clears h2, h3, ...).
    strip_headers:
        If True (default), header lines are removed from chunk content and
        live only in metadata. If False, the header line stays in the text.
    drop_empty:
        If True (default), chunks whose content is blank are discarded
        (their headers still propagate to following chunks via metadata).
    """

    def __init__(
        self,
        headers_to_split_on=(("#", "h1"), ("##", "h2"), ("###", "h3")),
        strip_headers: bool = True,
        drop_empty: bool = True,
    ) -> None:
        # marker length (number of '#') -> metadata key
        self._levels: dict[int, str] = {
            marker.count("#"): key for marker, key in headers_to_split_on
        }
        self._strip_headers = strip_headers
        self._drop_empty = drop_empty

    def split(self, text: str) -> list[MarkdownChunk]:
        chunks: list[MarkdownChunk] = []
        buffer: list[str] = []
        active: dict[int, str] = {}  # level -> header text, current context

        # code-fence state: None when outside a fence, else the delimiter run
        fence: str | None = None

        def flush() -> None:
            content = "\n".join(buffer).strip("\n")
            buffer.clear()
            if self._drop_empty and not content.strip():
                return
            levels = sorted(active)
            metadata = {self._levels[lvl]: active[lvl] for lvl in levels}
            headers = [active[lvl] for lvl in levels]
            chunks.append(
                MarkdownChunk(content=content, metadata=metadata, headers=headers)
            )

        for line in text.splitlines():
            fence_match = _FENCE_RE.match(line)
            if fence_match:
                marker = fence_match.group(1)
                if fence is None:
                    fence = marker[0] * 3  # opening fence; remember the char
                elif line.lstrip().startswith(fence):
                    fence = None  # closing fence
                buffer.append(line)
                continue

            # Inside a code block, never interpret '#' as a header.
            if fence is not None:
                buffer.append(line)
                continue

            header_match = _HEADER_RE.match(line)
            if header_match and len(header_match.group(1)) in self._levels:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                flush()  # close out the previous section
                active[level] = title
                # a header resets everything deeper than it
                for deeper in [l for l in active if l > level]:
                    del active[deeper]
                if not self._strip_headers:
                    buffer.append(line)
            else:
                buffer.append(line)

        flush()
        return chunks


# ---------------------------------------------------------------------------
# Paragraph packing
# ---------------------------------------------------------------------------

# Fallback token estimate, used when the caller passes no real tokenizer.
# Keeping this module dependency-free means it cannot import one itself, so the
# constant is calibrated on the PLATO corpus: bge-m3's tokenizer averages 2.94
# characters per token there (LaTeX markup and numerals make scientific text far
# denser than the ~4 chars/token of ordinary prose). Rounded down to 2.5 so the
# estimate over-counts and chunks land under budget rather than over.
_CHARS_PER_TOKEN = 2.5


def _estimate_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN) + 1


# Sections whose content is not worth indexing. Bibliographies are 24% of the
# PLATO corpus by token count and cannot answer a question about the mission:
# a list of citations only ever matches as a false positive, and chopped into
# 512-token pieces it becomes hundreds of them. Appendices are deliberately NOT
# dropped -- in this corpus they carry configuration files and derivations that
# a helpdesk bot should be able to quote.
#
# The plural "references" is load-bearing: this corpus has an "Appendix A
# Reference frames" section (and "A.2 The camera reference frame") describing
# instrument coordinate systems, which a singular match would throw away. The
# optional prefix absorbs section numbering -- "9 References and citations",
# "VIII Acknowledgments" -- while the anchor keeps "9.3 The list of references"
# and "9.1 Cross-referencing", which are about citing rather than citations.
DROP_SECTIONS = re.compile(
    r"\s*(?:(?:[IVXLC]+|[A-Z]|\d+)(?:\.\d+)*[.)]?\s+)?"
    r"(references|bibliography|acknowledge?ments?)\b",
    re.I,
)

# The |---|:---:|---| rule under a pipe-table header row.
_TABLE_RULE_RE = re.compile(r"^[\s|:-]*-[\s|:-]*$")

# "Table 3: Rotation period and activity indicators for all 430 targets." — the
# sentence saying what a table holds. Matched by shape rather than size because
# captions in this corpus run to a few hundred tokens, well past any sensible
# orphan threshold.
_CAPTION_RE = re.compile(r"\s*(?:table|tab\.)\s*\d+", re.I)


def is_markdown_table(para: str) -> bool:
    """True for a pipe table — rows whose meaning lives in the header row."""
    lines = [l for l in para.splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    return sum(1 for l in lines if l.count("|") >= 3) >= 0.8 * len(lines)


class ParagraphPacker:
    """Pack blank-line-separated paragraphs into token-bounded chunks.

    Splitting only on headers makes a whole paper section the retrieval unit,
    which is both too big for an embedding model to see in full and too coarse
    to rank precisely: one vector has to stand for the entire section. Packing
    paragraphs up to a budget keeps chunks semantically whole — cuts land on
    paragraph boundaries, never mid-sentence — while sizing them like an
    argument rather than a section.

    Parameters
    ----------
    max_tokens:
        Ceiling for ordinary prose chunks, enforced strictly. A prose paragraph
        longer than this is hard-split on word boundaries (the only case where
        a cut lands mid-sentence). Atomic paragraphs are exempt — see below.
    overlap_tokens:
        Trailing paragraphs repeated at the start of the following chunk, so a
        claim spanning a boundary still appears whole in one of them. The
        overlap is dropped whenever carrying it would push the next paragraph
        over ``max_tokens`` — the budget wins over the overlap.
    min_tokens:
        Floor below which a chunk is merged into a neighbour rather than
        indexed on its own. An over-long paragraph forces the buffer to flush
        early, which otherwise strands its lead-in — "Our findings can be
        summarized as follows." alone is an index entry that can only ever be a
        false match. Merging prefers the *following* chunk, since such a
        fragment introduces what comes after it.
    is_atomic:
        Predicate marking paragraphs that are their own chunk regardless of
        size. Defaults to :func:`is_markdown_table`. A table cut on word
        boundaries loses its header row, and every piece after the first is
        uninterpretable digits — worse than useless, because it still gets an
        embedding and can still be retrieved. Atomic paragraphs are exempt from
        ``max_tokens`` and never share a chunk with surrounding prose, with one
        exception: a preceding caption ("Table 3: Rotation period and activity
        indicators…") is pulled in, because it is the only thing saying what
        the columns mean. Applying this to *every* table rather than only
        oversized ones costs a little context around small inline tables, and
        buys a caption-plus-table pair that is never separated by the budget.
    atomic_max_tokens:
        Ceiling above which even an atomic paragraph has to be broken up —
        set it to the embedding model's context window, since beyond that the
        model silently truncates anyway. A table over the limit is split by
        rows with its header row repeated on every piece, so each stays
        readable. ``None`` (default) means never split them.
    count_tokens:
        Callable returning the token count of a string. Defaults to
        :func:`_estimate_tokens`; pass the embedding model's tokenizer, e.g.
        ``lambda s: len(tok.encode(s, add_special_tokens=False))``, for exact
        packing.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        min_tokens: int = 48,
        is_atomic=None,
        atomic_max_tokens: int | None = None,
        count_tokens=None,
    ) -> None:
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        if min_tokens >= max_tokens:
            raise ValueError("min_tokens must be smaller than max_tokens")
        if atomic_max_tokens is not None and atomic_max_tokens < max_tokens:
            raise ValueError("atomic_max_tokens must be at least max_tokens")
        self._max = max_tokens
        self._overlap = overlap_tokens
        self._min = min_tokens
        self._is_atomic = is_atomic if is_atomic is not None else is_markdown_table
        self._atomic_max = atomic_max_tokens
        self._count = count_tokens or _estimate_tokens

    def _hard_split(self, para: str, n_tokens: int) -> list[str]:
        """Break a single over-long prose paragraph on word boundaries."""
        words = para.split()
        if not words:
            return []
        # Words per piece, scaled from the measured token count. Aim slightly
        # under budget, then verify each piece and shrink further if the
        # tokenizer disagrees with the linear estimate (it often does on
        # formula-heavy text, where token density varies within a paragraph).
        per_piece = max(1, int(len(words) * self._max / n_tokens * 0.9))
        pieces, i = [], 0
        while i < len(words):
            piece = " ".join(words[i : i + per_piece])
            while self._count(piece) > self._max and piece.count(" "):
                piece = piece.rsplit(" ", 1)[0]
            pieces.append(piece)
            i += len(piece.split())
        return pieces

    def _split_table(self, table: str, limit: int) -> list[str]:
        """Break a table by rows, repeating the header on every piece."""
        lines = [l for l in table.splitlines() if l.strip()]
        # Header block: through the |---|---| rule if there is one, else row 1.
        head_end = 0
        for i, line in enumerate(lines[:3]):
            if "|" in line and _TABLE_RULE_RE.match(line):
                head_end = i
                break
        header = lines[: head_end + 1]
        # +1 per line for the newline joining it to the rest; the running total
        # is only a guide, and the verification pass below is what enforces the
        # limit, but over-counting here keeps that pass from having to work.
        header_tokens = self._count("\n".join(header)) + 1

        groups: list[list[str]] = []
        rows: list[str] = []
        rows_tokens = header_tokens
        for row in lines[head_end + 1 :]:
            n = self._count(row) + 1
            if rows and rows_tokens + n > limit:
                groups.append(rows)
                rows, rows_tokens = [], header_tokens
            rows.append(row)
            rows_tokens += n
        if rows:
            groups.append(rows)

        # Summing per-row counts is not the same as tokenizing the joined text,
        # so measure each piece for real and push any excess rows into the next
        # one. Without this a piece can land just over the model's window and be
        # silently truncated -- losing exactly the rows we split the table to keep.
        out: list[str] = []
        carry: list[str] = []
        for group in groups + [[]]:
            rows = carry + group
            carry = []
            if not rows:
                continue
            while len(rows) > 1 and self._count("\n".join(header + rows)) > limit:
                carry.insert(0, rows.pop())
            out.append("\n".join(header + rows))
        while carry:
            rows, carry = carry, []
            while len(rows) > 1 and self._count("\n".join(header + rows)) > limit:
                carry.insert(0, rows.pop())
            out.append("\n".join(header + rows))
        return out or [table]

    def pack(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in _PARA_RE.split(text) if p.strip()]
        out: list[str] = []
        buf: list[tuple[str, int]] = []  # (paragraph, token count)
        buf_tokens = 0

        def flush() -> None:
            nonlocal buf, buf_tokens
            if buf:
                out.append("\n\n".join(p for p, _ in buf))
                buf, buf_tokens = [], 0

        for para in paragraphs:
            # Counted after the atomic check: tokenizing a 45k-token table only
            # to discard the number is the most expensive no-op in this loop.
            if self._is_atomic(para):
                # Keep a pending caption attached to its table. Split apart,
                # the caption is retrieved without its data and the table is
                # retrieved without anything saying what its columns mean.
                prefix = ""
                if buf and buf_tokens <= self._min:
                    prefix = "\n\n".join(p for p, _ in buf) + "\n\n"
                    buf, buf_tokens = [], 0
                elif buf and _CAPTION_RE.match(buf[-1][0]):
                    # The caption is the tail of an already-packed buffer, so
                    # emit what came before it and carry only the caption over.
                    caption, caption_tokens = buf.pop()
                    buf_tokens -= caption_tokens
                    flush()
                    prefix = caption + "\n\n"
                else:
                    flush()
                limit = self._atomic_max
                if limit is None or self._count(prefix + para) <= limit:
                    out.append(prefix + para)
                else:
                    # The caption rides on the first piece, so that piece has
                    # less room for rows than the others.
                    pieces = self._split_table(para, limit - self._count(prefix))
                    pieces[0] = prefix + pieces[0]
                    out.extend(pieces)
                continue

            n = self._count(para)
            if n > self._max:
                flush()
                out.extend(self._hard_split(para, n))
                continue

            if buf and buf_tokens + n > self._max:
                out.append("\n\n".join(p for p, _ in buf))
                # Carry the trailing paragraphs forward as overlap, newest
                # first until the overlap budget is used up.
                tail: list[tuple[str, int]] = []
                tail_tokens = 0
                for p, pn in reversed(buf):
                    if tail_tokens + pn > self._overlap:
                        break
                    tail.insert(0, (p, pn))
                    tail_tokens += pn
                # Never let the overlap itself force the next chunk over budget.
                if tail_tokens + n > self._max:
                    tail, tail_tokens = [], 0
                buf, buf_tokens = tail, tail_tokens

            buf.append((para, n))
            buf_tokens += n

        flush()
        return self._coalesce(out)

    def _coalesce(self, chunks: list[str]) -> list[str]:
        """Fold undersized chunks into a neighbour, forward first."""
        if self._min <= 0:
            return chunks
        sizes = [self._count(c) for c in chunks]
        out: list[str] = []
        out_sizes: list[int] = []
        i = 0
        while i < len(chunks):
            text, n = chunks[i], sizes[i]
            if n < self._min:
                if i + 1 < len(chunks) and n + sizes[i + 1] <= self._max:
                    # Merge forward: hand the fragment to the next chunk.
                    chunks[i + 1] = text + "\n\n" + chunks[i + 1]
                    sizes[i + 1] = n + sizes[i + 1]
                    i += 1
                    continue
                if out and out_sizes[-1] + n <= self._max:
                    out[-1] += "\n\n" + text
                    out_sizes[-1] += n
                    i += 1
                    continue
            out.append(text)
            out_sizes.append(n)
            i += 1
        return out


def split_markdown_packed(
    text: str,
    *,
    headers_to_split_on=(("#", "h1"), ("##", "h2"), ("###", "h3")),
    strip_headers: bool = True,
    drop_empty: bool = True,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    min_tokens: int = 48,
    is_atomic=None,
    atomic_max_tokens: int | None = None,
    drop_sections=DROP_SECTIONS,
    count_tokens=None,
) -> list[MarkdownChunk]:
    """Header-split ``text``, drop unwanted sections, then paragraph-pack.

    Every piece of a section inherits that section's metadata and headers, and
    records its position via ``part`` / ``n_parts``. Pass ``drop_sections=None``
    to keep bibliographies and acknowledgements.
    """
    if isinstance(drop_sections, str):
        drop_sections = re.compile(drop_sections, re.I)

    splitter = MarkdownHeaderSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=strip_headers,
        drop_empty=drop_empty,
    )
    packer = ParagraphPacker(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
        is_atomic=is_atomic,
        atomic_max_tokens=atomic_max_tokens,
        count_tokens=count_tokens,
    )

    out: list[MarkdownChunk] = []
    for section in splitter.split(text):
        # Match on any level of the hierarchy: a "References" h2 keeps its
        # subsections out too, and a stray "## Acknowledgements" nested under
        # a paper title is caught the same way.
        if drop_sections and any(drop_sections.match(h.strip()) for h in section.headers):
            continue
        pieces = packer.pack(section.content)
        for i, piece in enumerate(pieces, start=1):
            out.append(
                MarkdownChunk(
                    content=piece,
                    metadata=dict(section.metadata),
                    headers=list(section.headers),
                    part=i,
                    n_parts=len(pieces),
                )
            )
    return out


# Convenience function mirroring a one-shot call style.
def split_markdown(text: str, **kwargs) -> list[MarkdownChunk]:
    return MarkdownHeaderSplitter(**kwargs).split(text)
