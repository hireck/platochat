#!/usr/bin/env python
"""Download a paper's full text: arXiv source, else arXiv PDF, else an open-access PDF.

Which full texts we fetch at all is a licensing decision, made in
``fulltext_sources()``: a paper's text is downloaded only if it is on arXiv or
ADS flags a copy of it as open access. The paywalled papers on the list (94 of
179 in September 2026, mostly SPIE proceedings) are indexed by their abstract
alone. Downloading them would be easy -- on the AU network the publishers hand
them over -- but quoting subscription content in a public chatbot amounts to
redistributing it, which subscription licences generally do not allow.

Sources, best first (a paper only gets the ones that apply to it):

``manual``         a PDF someone saved by hand as <data>/manual/<bibcode>.pdf,
                   for an open-access paper the script cannot fetch itself
``arxiv-source``   the LaTeX bundle from arXiv. LaTeXML makes much cleaner
                   markdown from it than any PDF converter: real headings, real
                   math, captions attached to their figures
``arxiv-pdf``      the arXiv PDF, when the LaTeX is unusable. Some authors
                   upload only a PDF, and then the "source" *is* that PDF
``publisher-pdf``  the publisher's PDF, if ADS flags it open access: the direct
                   link ADS has, else the one the landing page names in its
                   citation_pdf_url tag (Zenodo and Cambridge work that way)
``author-pdf``     an open copy the author deposited (a thesis repository)
``ads-scan``       ADS's scan of the printed pages, for old proceedings

Some publishers refuse us: Oxford University Press (MNRAS) and EPJ answer 403,
Springer serves a JavaScript bot check in place of the article page. We
identify ourselves honestly and do not work around a publisher's refusal.
update_corpus.py lists those papers, and a PDF saved by hand into the manual
folder is picked up on the next run.

Standard library only, like fetch_metadata.py -- except that arXiv is asked
through `requests`, which its CDN serves where it refuses urllib (see
ARXIV_THROTTLE). Without `requests` installed, urllib is used for arXiv too.

    python fetch_fulltext.py 2025ExA....59...26R            # try one paper
    python fetch_fulltext.py 2026AJ....171...14B --dest /tmp/x
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.client import HTTPException

try:            # see ARXIV_CLIENT
    import requests
except ImportError:
    requests = None

# Who we are, for the server logs of everyone we download from.
USER_AGENT = "PLATOChat-ingest/1.0 (+https://platopub.phys.au.dk/)"

# arXiv asks automated clients to use this mirror of the site rather than
# arxiv.org itself, and to leave three seconds between requests.
ARXIV = "https://export.arxiv.org"
ADS_RESOLVER = "https://api.adsabs.harvard.edu/v1/resolver"

# Seconds between requests to one host; arXiv's stated minimum is 3.
HOST_DELAY = {"export.arxiv.org": 5.0}
DEFAULT_DELAY = 1.0
TIMEOUT = 60                       # seconds without a byte before giving up
MAX_DOWNLOAD = 400 * 2**20         # one file; arXiv bundles with raw data run large
MAX_UNPACKED = 1500 * 2**20        # everything a source bundle unpacks to
MAX_PAGE = 20 * 2**20              # a landing page

# HTTP statuses worth trying again later. Anything else -- 403, 404, 410 -- will
# say the same thing tomorrow.
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}

# arXiv's CDN refuses Python's own HTTP client (urllib) with an empty 406
# ("Not Acceptable") on most requests for files not already in its cache.
# Measured 2026-09-21: urllib was refused even 45 s apart, while curl and the
# `requests` library, from the same machine at the same minute and with the
# same User-Agent, were served every time (4 of 4 uncached bundles); other
# projects report the same that year. arXiv's published rules for
# export.arxiv.org invite exactly this use -- programmatic downloads, at up to
# four requests a second -- so for arXiv we use `requests` (ARXIV_CLIENT). We
# still say who we are and keep five seconds between requests. A 406 that gets
# through anyway is waited out: the caller tries again a round later.
# Publishers are another matter: their refusals (403, bot checks) may be meant,
# so those are respected and reported, never worked around.
ARXIV_THROTTLE = {406}
ARXIV_CLIENT = "requests" if requests is not None else "urllib"

# The ADS property that says a given copy is free to read.
OPEN_FLAG = {
    "publisher-pdf": "PUB_OPENACCESS",
    "author-pdf":    "AUTHOR_OPENACCESS",
    "ads-scan":      "ADS_OPENACCESS",
}
SOURCES = ("manual", "arxiv-source", "arxiv-pdf", "publisher-pdf", "author-pdf", "ads-scan")


class FetchError(Exception):
    """A source that did not yield a usable file.

    ``transient`` separates "try again tomorrow" (a timeout, a 503) from "this
    will not work" (a 404, a publisher that refuses scripts, no LaTeX in the
    bundle). Only a permanent failure moves on to the next, worse source.
    """

    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient


@dataclass
class Fetched:
    source: str     # one of SOURCES
    kind: str       # "tex" (path is the main .tex of an unpacked bundle) or "pdf"
    path: str
    url: str        # where it came from, after redirects
    sha256: str     # of the file as downloaded (for a bundle: the archive)
    note: str = ""  # e.g. how the main .tex was chosen


# ---------------------------------------------------------------------------
# Policy: whose full text we may fetch
# ---------------------------------------------------------------------------

def access_of(rec: dict) -> str:
    """'arxiv', 'open access' or 'paywalled' -- full text for the first two only."""
    if rec.get("arxiv_id"):
        return "arxiv"
    if set(rec.get("property") or []) & set(OPEN_FLAG.values()):
        return "open access"
    return "paywalled"


def fulltext_allowed(rec: dict) -> bool:
    return access_of(rec) != "paywalled"


def fulltext_sources(rec: dict, manual_pdf: str | None = None) -> list[str]:
    """The sources worth trying for this paper, best first; empty if paywalled."""
    if not fulltext_allowed(rec):
        return []
    props = set(rec.get("property") or [])
    esources = set(rec.get("esources") or [])
    out = ["manual"] if manual_pdf else []
    if rec.get("arxiv_id"):
        out += ["arxiv-source", "arxiv-pdf"]
    if "PUB_OPENACCESS" in props and (esources & {"PUB_PDF", "PUB_HTML"} or rec.get("doi")):
        out.append("publisher-pdf")
    if "AUTHOR_OPENACCESS" in props and "AUTHOR_PDF" in esources:
        out.append("author-pdf")
    if "ADS_OPENACCESS" in props and "ADS_PDF" in esources:
        out.append("ads-scan")
    return out


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

_last_request: dict[str, float] = {}
# Requests arXiv has refused in a row. Past ARXIV_GIVE_UP, arXiv is not asked
# again until reset_throttle() -- update_corpus.py calls it between rounds.
_arxiv_strikes = 0
ARXIV_GIVE_UP = 8


def reset_throttle() -> None:
    global _arxiv_strikes
    _arxiv_strikes = 0


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or url


def _wait_turn(url: str) -> None:
    """Keep to each host's request rate (only the first host of a redirect chain)."""
    host = _host(url)
    wait = _last_request.get(host, 0.0) + HOST_DELAY.get(host, DEFAULT_DELAY) - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request[host] = time.monotonic()


class _Status(Exception):
    """An HTTP error status, whichever client got it."""

    def __init__(self, code: int, url: str, retry_after: str = ""):
        super().__init__(code)
        self.code, self.url, self.retry_after = code, url, retry_after or ""


def _open(url: str, headers: dict):
    """``(final url, content type, byte blocks, close)``; raises _Status on 4xx/5xx."""
    if _host(url).endswith("arxiv.org") and ARXIV_CLIENT == "requests":
        resp = requests.get(url, headers=headers, timeout=TIMEOUT, stream=True)
        if resp.status_code >= 400:
            resp.close()
            raise _Status(resp.status_code, resp.url, resp.headers.get("Retry-After"))
        return (resp.url, resp.headers.get("Content-Type", ""),
                resp.iter_content(1 << 16), resp.close)
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                      timeout=TIMEOUT)
    except urllib.error.HTTPError as exc:
        raise _Status(exc.code, exc.url or url,
                      exc.headers.get("Retry-After") if exc.headers else "") from None
    return (resp.geturl(), resp.headers.get("Content-Type", ""),
            iter(lambda: resp.read(1 << 16), b""), resp.close)


def http_get(url: str, dest: str | None = None, *, headers: dict | None = None,
             max_bytes: int = MAX_DOWNLOAD, retries: int = 1) -> tuple[str, str, bytes | None]:
    """GET ``url``; returns ``(final url, content type, body)``.

    With ``dest`` the body is streamed to that file instead (and None is
    returned for it), so a 300 MB source bundle never sits in memory. A 429 or
    5xx is retried ``retries`` times after a pause; a timeout is not, since the
    next run will try again anyway, and neither is an arXiv refusal (see
    ARXIV_THROTTLE), which the caller waits out between rounds.
    """
    global _arxiv_strikes
    arxiv = _host(url).endswith("arxiv.org")
    if arxiv and _arxiv_strikes >= ARXIV_GIVE_UP:
        raise FetchError("arXiv refused the last few requests; not asked again this round",
                         transient=True)
    for attempt in range(retries + 1):
        _wait_turn(url)
        part = dest + ".part" if dest else None
        try:
            final, ctype, blocks, close = _open(
                url, {"User-Agent": USER_AGENT, "Accept": "*/*", **(headers or {})})
            chunks, size = [], 0
            out = open(part, "wb") if part else None
            try:
                for block in blocks:
                    size += len(block)
                    if size > max_bytes:
                        raise FetchError(f"{_host(final)}: file larger than "
                                         f"{max_bytes >> 20} MB")
                    if out:
                        out.write(block)
                    else:
                        chunks.append(block)
            finally:
                close()
                if out:
                    out.close()
            if arxiv:
                _arxiv_strikes = 0
            if part:
                os.replace(part, dest)
                return final, ctype, None
            return final, ctype, b"".join(chunks)
        except _Status as exc:
            host = _host(exc.url or url)
            throttled = host.endswith("arxiv.org") and exc.code in ARXIV_THROTTLE
            err = FetchError(f"HTTP {exc.code} from {host}"
                             + (" (arXiv refusing us for now)" if throttled else ""),
                             transient=throttled or exc.code in TRANSIENT_STATUS)
            pause = (None if throttled
                     else min(int(exc.retry_after), 120) if exc.retry_after.isdigit()
                     else 10 * (attempt + 1))
        except (OSError, HTTPException) as exc:   # URLError and requests' errors are OSErrors
            reason = getattr(exc, "reason", None) or exc
            err, pause = FetchError(f"{_host(url)}: {reason}", transient=True), None
        finally:
            if part and os.path.exists(part):
                os.remove(part)
        if err.transient and pause is not None and attempt < retries:
            time.sleep(pause)
            continue
        if arxiv and "refusing" in str(err):
            _arxiv_strikes += 1
        raise err
    raise AssertionError("unreachable")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _is_pdf(path: str) -> bool:
    # The PDF spec allows junk before the header, within the first 1024 bytes.
    with open(path, "rb") as fh:
        return b"%PDF-" in fh.read(1024)


def _is_bot_check(page: str, url: str = "") -> bool:
    return bool(_BOT_CHECK_RE.search(page[:20000]) or _BOT_CHECK_RE.search(url))


def _fetch_pdf(url: str, dest: str, what: str) -> tuple[str, str]:
    """Download what should be a PDF; ``(final url, sha256)`` if it is one.

    A bot check in its place counts as temporary: IOP served the same PDF in
    the morning of 2026-09-21 and a Radware check (validate.perfdrive.com) in
    the afternoon. The caller tries again on a later run -- not by answering
    the check -- and gives up after a few (update_corpus.MAX_TRANSIENT_TRIES).
    """
    final, ctype, _ = http_get(url, dest)
    if not _is_pdf(dest):
        with open(dest, "rb") as fh:
            page = fh.read(20000).decode("utf-8", "replace")
        os.remove(dest)
        if _is_bot_check(page, final):
            raise FetchError(f"{_host(final)} served a bot check instead of the {what}",
                             transient=True)
        raise FetchError(f"the {what} at {_host(final)} is not a PDF "
                         f"({ctype.split(';')[0] or 'unknown type'})")
    return final, sha256_file(dest)


# ---------------------------------------------------------------------------
# ADS links and landing pages
# ---------------------------------------------------------------------------

def ads_link(bibcode: str, link_type: str, ads_key: str) -> str | None:
    """The URL ADS holds for one of a paper's full-text links (PUB_PDF, ...)."""
    url = f"{ADS_RESOLVER}/{urllib.parse.quote(bibcode, safe='')}/{link_type}"
    try:
        _, _, body = http_get(url, headers={"Authorization": f"Bearer {ads_key}",
                                            "Accept": "application/json"})
    except FetchError as exc:
        if str(exc).startswith("HTTP 404"):   # ADS: "did not find any records"
            return None
        raise
    data = json.loads(body or b"{}")
    if data.get("action") == "redirect":
        return data.get("link")
    records = (data.get("links") or {}).get("records") or []
    return records[0].get("url") if records else None


_META_RE = re.compile(r"<meta\s[^>]*>", re.I)
# Marks of the interstitial pages bot filters serve in place of the real one
# (Cloudflare, Fastly, Radware's perfdrive).
_BOT_CHECK_RE = re.compile(
    r"captcha|just a moment|enable javascript|challenge-platform|/_fs-ch-|cf-chl|perfdrive",
    re.I)
_ATTR_RE = re.compile(r"""([\w:.-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")


def citation_pdf_url(page: str, base: str) -> str | None:
    """The PDF a landing page declares for Google Scholar, if it declares one.

    ``<meta name="citation_pdf_url" content="...">`` is the Highwire tag that
    Zenodo, Cambridge, EPJ, A&A and most other publishers put on article pages.
    """
    for tag in _META_RE.findall(page):
        attrs = {m.group(1).lower(): html.unescape(m.group(2) if m.group(2) is not None
                                                   else m.group(3))
                 for m in _ATTR_RE.finditer(tag)}
        if attrs.get("name", "").lower() == "citation_pdf_url" and attrs.get("content"):
            return urllib.parse.urljoin(base, attrs["content"].strip())
    return None


# ---------------------------------------------------------------------------
# Picking the main .tex of an arXiv bundle
# ---------------------------------------------------------------------------

def _words(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def title_share(title: str, text: str) -> float:
    """Share of the title's longer words that occur in ``text``."""
    wanted = {w for w in _words(title) if len(w) > 3}
    if not wanted:
        return 1.0
    return len(wanted & _words(text)) / len(wanted)


_COMMENT_RE = re.compile(r"(?<!\\)%.*")
_DOCUMENTCLASS_RE = re.compile(r"\\document(?:class|style)\b")
_BEGIN_DOCUMENT_RE = re.compile(r"\\begin\s*\{\s*document\s*\}")
_TITLE_RE = re.compile(r"\\title\s*(?:\[[^\]]*\])?\s*\{")
# Compilable files that are not the paper: journal author guides and templates
# shipped inside the bundle, referee replies, cover letters.
_NOT_THE_PAPER_RE = re.compile(
    r"guide|template|sample|example|response|reply|referee|cover|letter", re.I)


def _read_tex(path: str) -> str:
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _tex_title(tex: str) -> str:
    """The argument of ``\\title{...}``, braces balanced; '' if there is none."""
    m = _TITLE_RE.search(tex)
    if not m:
        return ""
    depth, i = 1, m.end()
    while i < len(tex) and depth:
        if tex[i] in "{}" and tex[i - 1] != "\\":
            depth += 1 if tex[i] == "{" else -1
        i += 1
    return tex[m.end():i - 1]


def _readme_toplevel(src_dir: str) -> set[str]:
    """Files arXiv's own README says it compiles, relative to the bundle."""
    path = os.path.join(src_dir, "00README.json")    # arXiv's current format
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                sources = json.load(fh).get("sources") or []
        except (ValueError, OSError, AttributeError):
            return set()
        return {s["filename"] for s in sources
                if isinstance(s, dict) and s.get("usage") == "toplevel" and s.get("filename")}
    path = os.path.join(src_dir, "00README.XXX")     # the older one
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as fh:
            return {ln.split()[0] for ln in fh
                    if len(ln.split()) >= 2 and ln.split()[1] == "toplevelfile"}
    return set()


def pick_main_tex(src_dir: str, title: str) -> tuple[str | None, str]:
    """The .tex file that is the paper, and a note on how it was chosen.

    Not simply the first file with a ``\\documentclass``: Bray et al. (2023)
    ships the MNRAS author guide in its bundle, and that is what got converted
    and indexed under the paper's name. Candidates are the files that compile
    on their own (``\\documentclass`` and ``\\begin{document}``, outside
    comments); arXiv's README narrows them down where it names a top-level
    file; then the one whose ``\\title`` best matches the paper's title from
    ADS wins, with names like "guide" or "reply" and small files losing ties.
    update_corpus.py checks the converted markdown for the title again, so a
    wrong pick here is caught there.
    """
    candidates = []
    for dirpath, _dirs, files in os.walk(src_dir):
        for fn in files:
            if not fn.lower().endswith(".tex"):
                continue
            path = os.path.join(dirpath, fn)
            tex = _COMMENT_RE.sub("", _read_tex(path))
            if _DOCUMENTCLASS_RE.search(tex) and _BEGIN_DOCUMENT_RE.search(tex):
                candidates.append((path, tex))
    if not candidates:
        return None, "no .tex file with \\documentclass and \\begin{document}"
    if len(candidates) == 1:
        return candidates[0][0], "the only compilable .tex file"

    toplevel = _readme_toplevel(src_dir)
    pool = [c for c in candidates if os.path.relpath(c[0], src_dir) in toplevel] or candidates

    def rank(c):
        path, tex = c
        return (round(title_share(title, _tex_title(tex)), 2),
                not _NOT_THE_PAPER_RE.search(os.path.basename(path)),
                os.path.getsize(path))

    best = max(pool, key=rank)
    score = rank(best)[0]
    how = (f"title match {score:.2f}" if score >= 0.5
           else "no \\title matched the paper's, took the likeliest by name and size")
    others = ", ".join(sorted(os.path.relpath(p, src_dir) for p, _ in candidates if p != best[0]))
    return best[0], f"{os.path.relpath(best[0], src_dir)}: {how}; passed over {others}"


# ---------------------------------------------------------------------------
# Unpacking an arXiv e-print
# ---------------------------------------------------------------------------

def _safe_extract(tar: tarfile.TarFile, dest: str) -> None:
    """Extract regular files and directories only, and nothing outside ``dest``.

    Done by hand rather than with tarfile's ``filter="data"`` (Python 3.12+):
    no links at all, not even inside the bundle, no devices or absolute/``..``
    paths -- and a cap on the unpacked size, which the filter does not have.
    """
    root = os.path.realpath(dest)
    os.makedirs(root, exist_ok=True)
    total = 0
    for member in tar.getmembers():
        if not (member.isfile() or member.isdir()):
            continue
        target = os.path.realpath(os.path.join(root, member.name))
        if target != root and not target.startswith(root + os.sep):
            continue
        if member.isdir():
            os.makedirs(target, exist_ok=True)
            continue
        total += member.size
        if total > MAX_UNPACKED:
            raise FetchError(f"the arXiv source unpacks to more than {MAX_UNPACKED >> 20} MB")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with tar.extractfile(member) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)


def unpack_eprint(path: str, dest: str) -> tuple[str, str]:
    """Unpack an arXiv e-print; ``("tex", dest)`` or ``("pdf", path to the PDF)``.

    What arXiv serves as "source" comes in four shapes: a (gzipped) tar of the
    LaTeX tree, a single gzipped .tex file, a PDF (the authors uploaded no
    LaTeX -- Rauer et al. 2014 is one), or rarely a gzipped PDF.
    """
    if _is_pdf(path):
        return "pdf", path
    try:
        with tarfile.open(path, "r:*") as tar:
            _safe_extract(tar, dest)
        return "tex", dest
    except tarfile.ReadError:
        pass
    with open(path, "rb") as fh:
        gzipped = fh.read(2) == b"\x1f\x8b"
    with (gzip.open(path, "rb") if gzipped else open(path, "rb")) as fh:
        data = fh.read(MAX_DOWNLOAD + 1)
    if len(data) > MAX_DOWNLOAD:
        raise FetchError(f"the arXiv source unpacks to more than {MAX_DOWNLOAD >> 20} MB")
    if b"%PDF-" in data[:1024]:
        pdf = os.path.splitext(path)[0] + ".pdf"
        with open(pdf, "wb") as fh:
            fh.write(data)
        return "pdf", pdf
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "main.tex"), "wb") as fh:
        fh.write(data)
    return "tex", dest


# ---------------------------------------------------------------------------
# The sources
# ---------------------------------------------------------------------------

def fetch_arxiv_source(rec: dict, dest_dir: str) -> Fetched:
    arxiv_id = rec["arxiv_id"]
    raw = os.path.join(dest_dir, "arxiv-eprint")
    final, _, _ = http_get(f"{ARXIV}/e-print/{arxiv_id}", raw)
    sha = sha256_file(raw)
    src_dir = os.path.join(dest_dir, "source")
    if os.path.isdir(src_dir):
        shutil.rmtree(src_dir)       # a previous attempt; unpack afresh
    kind, path = unpack_eprint(raw, src_dir)
    if kind == "pdf":
        # No LaTeX to be had: this *is* the arXiv PDF, so report it as that
        # source and the caller will not download the same file again.
        pdf = os.path.join(dest_dir, "arxiv.pdf")
        os.replace(path, pdf)
        return Fetched("arxiv-pdf", "pdf", pdf, final, sha,
                       "arXiv holds no LaTeX for this paper, only the authors' PDF")
    main, note = pick_main_tex(src_dir, rec.get("title") or "")
    if main is None:
        raise FetchError(f"arXiv source unusable: {note}")
    return Fetched("arxiv-source", "tex", main, final, sha, note)


def fetch_arxiv_pdf(rec: dict, dest_dir: str) -> Fetched:
    pdf = os.path.join(dest_dir, "arxiv.pdf")
    final, sha = _fetch_pdf(f"{ARXIV}/pdf/{rec['arxiv_id']}", pdf, "arXiv PDF")
    return Fetched("arxiv-pdf", "pdf", pdf, final, sha)


def fetch_publisher_pdf(rec: dict, dest_dir: str, ads_key: str) -> Fetched:
    """The publisher's PDF: ADS's direct link first, then the landing page's."""
    bibcode, esources = rec["bibcode"], set(rec.get("esources") or [])
    pdf = os.path.join(dest_dir, "publisher.pdf")
    errors: list[str] = []
    transient = False

    if "PUB_PDF" in esources:
        try:
            url = ads_link(bibcode, "PUB_PDF", ads_key)
            if url:
                final, sha = _fetch_pdf(url, pdf, "publisher PDF")
                return Fetched("publisher-pdf", "pdf", pdf, final, sha)
        except FetchError as exc:
            errors.append(str(exc))
            transient |= exc.transient

    try:
        landing = ads_link(bibcode, "PUB_HTML", ads_key) if "PUB_HTML" in esources else None
        if not landing and rec.get("doi"):
            landing = "https://doi.org/" + urllib.parse.quote(rec["doi"][0], safe="/")
        if landing:
            final, ctype, body = http_get(landing, max_bytes=MAX_PAGE)
            if b"%PDF-" in body[:1024]:        # a "landing page" that is the PDF
                with open(pdf, "wb") as fh:
                    fh.write(body)
                return Fetched("publisher-pdf", "pdf", pdf, final, sha256_file(pdf))
            page = body.decode("utf-8", "replace")
            pdf_url = citation_pdf_url(page, final)
            if not pdf_url and _is_bot_check(page, final):
                # Springer, for one, serves scripts a 3 KB JavaScript challenge
                # in place of the article page. Temporary, as in _fetch_pdf.
                errors.append(f"{_host(final)} served a bot check instead of the article page")
                transient = True
            elif not pdf_url:
                errors.append(f"the landing page at {_host(final)} names no PDF")
            else:
                final, sha = _fetch_pdf(pdf_url, pdf, "PDF named by the landing page")
                return Fetched("publisher-pdf", "pdf", pdf, final, sha,
                               f"via the landing page {landing}")
    except FetchError as exc:
        errors.append(str(exc))
        transient |= exc.transient
    raise FetchError("; ".join(errors) or "ADS has no publisher link", transient)


def fetch_ads_link_pdf(rec: dict, dest_dir: str, ads_key: str, source: str) -> Fetched:
    """A PDF behind one ADS link: the author's copy, or ADS's own scan."""
    link_type = {"author-pdf": "AUTHOR_PDF", "ads-scan": "ADS_PDF"}[source]
    url = ads_link(rec["bibcode"], link_type, ads_key)
    if not url:
        raise FetchError(f"ADS has no {link_type} link")
    pdf = os.path.join(dest_dir, f"{source}.pdf")
    final, sha = _fetch_pdf(url, pdf, link_type)
    return Fetched(source, "pdf", pdf, final, sha)


def fetch(rec: dict, source: str, dest_dir: str, *, ads_key: str | None = None,
          manual_pdf: str | None = None) -> Fetched:
    """Download one source of one paper into ``dest_dir``; raises FetchError."""
    if source != "manual" and source not in fulltext_sources(rec):
        # The policy, enforced here as well as by the caller's choice of sources.
        raise FetchError(f"{source} is not an allowed source for this paper "
                         f"({access_of(rec)})")
    os.makedirs(dest_dir, exist_ok=True)
    if source == "manual":
        if not fulltext_allowed(rec):
            raise FetchError("paywalled: its full text is not indexed, even by hand")
        pdf = os.path.join(dest_dir, "manual.pdf")
        shutil.copyfile(manual_pdf, pdf)
        if not _is_pdf(pdf):
            raise FetchError(f"{manual_pdf} is not a PDF")
        return Fetched("manual", "pdf", pdf, "file://" + os.path.abspath(manual_pdf),
                       sha256_file(pdf))
    if source == "arxiv-source":
        return fetch_arxiv_source(rec, dest_dir)
    if source == "arxiv-pdf":
        return fetch_arxiv_pdf(rec, dest_dir)
    if not ads_key:
        raise FetchError(f"{source} needs an ADS API key")
    if source == "publisher-pdf":
        return fetch_publisher_pdf(rec, dest_dir, ads_key)
    return fetch_ads_link_pdf(rec, dest_dir, ads_key, source)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bibcodes", nargs="+")
    ap.add_argument("--dest", help="download folder (default: a new temporary one)")
    ap.add_argument("--metadata", default=os.path.join(here, "papers.json"))
    ap.add_argument("--env", default=os.path.join(here, os.pardir, "platochat", ".env"))
    ap.add_argument("--all", action="store_true",
                    help="try every source, not just until the first one works")
    args = ap.parse_args()

    sys.path.insert(0, here)
    from fetch_metadata import load_key
    ads_key = load_key(args.env)
    with open(args.metadata, encoding="utf-8") as fh:
        papers = json.load(fh)["papers"]
    aliases = {a: b for b, r in papers.items() for a in r.get("aliases", [b])}
    dest_root = args.dest or tempfile.mkdtemp(prefix="fulltext-")

    for given in args.bibcodes:
        bibcode = aliases.get(given)
        if bibcode is None:
            print(f"{given}: not on the PLATO-Pub list")
            continue
        rec = papers[bibcode]
        sources = fulltext_sources(rec)
        print(f"{bibcode}  ({access_of(rec)})  {rec.get('title', '')[:70]}")
        if not sources:
            print("  paywalled -- abstract only, nothing to download")
        for source in sources:
            if source == "manual":
                continue
            try:
                got = fetch(rec, source, os.path.join(dest_root, bibcode), ads_key=ads_key)
            except FetchError as exc:
                print(f"  {source:14} failed{' (transient)' if exc.transient else ''}: {exc}")
                continue
            print(f"  {source:14} {got.kind}: {got.path}\n  {'':14} from {got.url}"
                  + (f"\n  {'':14} {got.note}" if got.note else ""))
            if not args.all:
                break
    print(f"\nFiles are in {dest_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
