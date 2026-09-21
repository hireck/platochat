"""Rebuild Markdown heading levels from section numbering — standard library only.

PDF converters recover heading *text* far more reliably than heading *depth*.
marker in particular assigns the number of ``#`` almost arbitrarily: measured on
Rauer et al. 2014 (*The PLATO 2.0 Mission*, 63 pp) it emitted ``# 4.7 ...`` and
``# 5.2 ...`` as top-level headings while the genuine top-level sections ``6``,
``7`` and ``9`` came out as ``###``, and ``4.1.1`` landed at the same level as
its parent ``4.1``. It also wraps heading text in ``**bold**``/``*italic*`` and
prefixes it with ``<span id="page-9-1"></span>`` anchors.

That matters because :mod:`textsplitter` derives each chunk's section
breadcrumb from the heading levels: a heading at level *n* clears every deeper
level, so one mis-levelled heading re-parents everything after it. On the paper
above, chunking marker's raw output gave 0 of 40 chunks rooted at the paper
title and 38 with markup debris in the breadcrumb; after renumbering, 48 of 48
chunks were correctly rooted and none carried debris.

The numbering itself, however, survives intact in the heading text. So this
module throws marker's levels away and derives them from the numbers instead:
``2`` becomes ``##``, ``2.1`` ``###``, ``4.1.1`` ``####``, with ``#`` reserved
for the paper title. Headings with no number are classified rather than guessed
at (see :func:`renumber_headings`).

The same repair works on any converter that keeps section numbers in the
heading text — MinerU, for instance, flattens every section and subsection to
``##`` but numbers them correctly — so nothing here is marker-specific.

No third-party dependencies, matching :mod:`textsplitter`.
"""

from __future__ import annotations

import re

# A heading line: 1-6 leading '#', at least one space, then text.
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")

# marker embeds link anchors in heading text: '<span id="page-9-1"></span>2.2 ...'
_ANCHOR_RE = re.compile(r"<span[^>]*>\s*</span>\s*")
# ...and wraps the whole heading in emphasis: '**1 Introduction**', '*2.1 ...*'.
_EMPHASIS_RE = re.compile(r"^\s*(\*{1,3}|_{1,3})(.+?)\1\s*$")

# "2", "2.1", "4.1.1", and "2.10Planetary atmospheres" — marker sometimes drops
# the space after the number, so \s* rather than \s+. The title must start with
# a letter, which stops the date line "28.02.2014" parsing as section 28.02.20.
_NUMBERED_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s*([^\W\d_].*)$")
# Appendix sections: "A1 Planetary Transits", "A2.1 ...". A space is required
# here so ordinary capitalised words ("Abstract") cannot match.
_APPENDIX_RE = re.compile(r"^([A-Z]\d{0,2}(?:\.\d{1,2})*)\.?\s+([^\W\d_].*)$")

#: Unnumbered headings that belong at section level rather than under whichever
#: numbered section happened to precede them.
FRONT_BACK_MATTER = frozenset({
    "abstract", "references", "bibliography", "acknowledgement",
    "acknowledgements", "acknowledgments", "appendix",
})

#: A numbered section deeper than this is treated as a false positive.
_MAX_DEPTH = 4
#: Section numbers start small; a larger leading number is a date or a measurement.
_MAX_SECTION_NUMBER = 30
#: Headings longer than this are prose that the converter mistook for a heading.
_MAX_HEADING_CHARS = 120


def strip_heading_markup(text: str) -> str:
    """Remove anchor spans and surrounding emphasis from heading text.

    ``'<span id="page-9-1"></span>*2.2 Constraints on planet formation*'``
    becomes ``'2.2 Constraints on planet formation'``. Emphasis is unwrapped
    repeatedly so ``***bold italic***`` is handled too.
    """
    text = _ANCHOR_RE.sub("", text).strip()
    previous = None
    while previous != text:
        previous = text
        match = _EMPHASIS_RE.match(text)
        if match:
            text = match.group(2).strip()
    return text


def parse_section_number(text: str) -> tuple[int, str] | None:
    """Split a heading into (depth, normalised text), or None if unnumbered.

    Depth is the number of dot-separated components, so "2" -> 1, "2.1" -> 2,
    "4.1.1" -> 3. The returned text always has exactly one space between the
    number and the title, repairing marker's occasional "2.10Planetary".
    """
    for pattern, is_appendix in ((_NUMBERED_RE, False), (_APPENDIX_RE, True)):
        match = pattern.match(text)
        if match is None:
            continue
        number, title = match.group(1), match.group(2)
        parts = number.split(".")
        if len(parts) > _MAX_DEPTH:
            return None
        if not is_appendix and int(parts[0]) > _MAX_SECTION_NUMBER:
            return None
        return len(parts), f"{number} {title}"
    return None


def _is_false_heading(text: str) -> bool:
    """True for lines the converter promoted to a heading but that are not one.

    Covers the three kinds seen in the PLATO corpus: bullet items
    ("• Ages to 10%"), lead-in sentences ("PLATO 2.0 will answer fundamental
    questions such as:"), and bare dates or figure numbers with no letters.
    """
    return (
        text.startswith(("•", "-", "*", "–", "—"))
        or text.endswith(":")
        or len(text) > _MAX_HEADING_CHARS
        or not any(character.isalpha() for character in text)
    )


def renumber_headings(markdown: str, title_from_first_heading: bool = True
                      ) -> tuple[str, dict[str, int]]:
    """Rewrite every heading's level from its section number.

    Each heading is handled in one of five ways, counted in the returned stats:

    ``title``
        The first heading becomes the document's single ``#``.
    ``numbered``
        Level comes from the section number: depth + 1, so top-level sections
        sit at ``##`` beneath the title.
    ``front_back``
        Abstract / References / Appendix / Acknowledgements, placed at ``##``
        so they are siblings of the numbered sections rather than children of
        whichever section preceded them.
    ``inherited``
        An unnumbered heading that is genuinely a subsection (e.g. "Effects of
        rotation" under "A2 Asteroseismology"), placed one level below the last
        numbered heading.
    ``demoted``
        A false heading (see :func:`_is_false_heading`), rewritten as a bold
        paragraph so its text is kept but it no longer splits a chunk.

    Args:
        markdown: the converted Markdown.
        title_from_first_heading: treat the first heading as the paper title.
            Set False for a document whose first heading is a real section.

    Returns:
        (rewritten markdown, stats dict).
    """
    lines_out: list[str] = []
    stats = {"title": 0, "numbered": 0, "front_back": 0,
             "inherited": 0, "demoted": 0}
    last_numbered_level = 1
    seen_title = False

    for line in markdown.split("\n"):
        match = _HEADING_RE.match(line)
        if match is None:
            lines_out.append(line)
            continue

        text = strip_heading_markup(match.group(2))
        if not text:
            lines_out.append(line)
            continue

        if title_from_first_heading and not seen_title:
            seen_title = True
            stats["title"] += 1
            lines_out.append(f"# {text}")
            continue

        is_front_back = (text.rstrip(":").strip().lower() in FRONT_BACK_MATTER
                         or text.lower().startswith("appendix"))

        if not is_front_back and _is_false_heading(text):
            stats["demoted"] += 1
            lines_out.append(f"**{text}**")
            continue

        parsed = parse_section_number(text)
        if parsed is not None:
            depth, normalised = parsed
            level = min(depth + 1, 6)  # '#' is reserved for the title
            last_numbered_level = level
            stats["numbered"] += 1
            lines_out.append(f"{'#' * level} {normalised}")
        elif is_front_back:
            stats["front_back"] += 1
            lines_out.append(f"## {text}")
        else:
            stats["inherited"] += 1
            level = min(last_numbered_level + 1, 6)
            lines_out.append(f"{'#' * level} {text}")

    return "\n".join(lines_out), stats


if __name__ == "__main__":
    import sys

    source = open(sys.argv[1], encoding="utf-8").read()
    result, counts = renumber_headings(source)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", encoding="utf-8") as handle:
            handle.write(result)
    else:
        print(result)
    print(counts, file=sys.stderr)
