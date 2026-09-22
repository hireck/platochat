import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heading_levels import renumber_headings  # noqa: E402


def load_pdf_models(device=None, dtype=None):
    """Load and return marker's ML model dict, for reuse across many PDFs.

    marker rasterizes layout/OCR/table detection through several torch models
    that are slow to load and are downloaded from HuggingFace on first use
    (multiple GB). When converting a whole corpus, load them once here and pass
    the result to ``pdf_to_markdown(..., models=...)`` so the cost is paid a
    single time rather than per file.

    ``device``/``dtype`` are forwarded to marker (e.g. device="mps" on Apple
    Silicon, "cuda" on a GPU box, "cpu" otherwise); None lets marker pick.
    """
    from marker.models import create_model_dict

    return create_model_dict(device=device, dtype=dtype)


def pdf_to_markdown(pdf_path, image_dir=None, models=None, page_range=None,
                    fix_heading_levels=True):
    """Convert a PDF to Markdown (with images) using marker.

    marker analyses the page layout to recover reading order, headings, lists,
    tables and math, then renders GitHub-flavored Markdown. Headings come out as
    ``#``/``##`` so the result feeds straight into ``MarkdownHeaderTextSplitter``,
    the same way the LaTeXML path does. Every figure marker finds is written into
    ``image_dir`` and referenced from the Markdown as ``![](<filename>)``.

    Image links in the returned Markdown are relative to ``image_dir`` (bare
    filenames), matching the convention of ``latex_to_markdown``.

    marker's heading *text* is reliable but the number of ``#`` it assigns is
    not, which corrupts the section breadcrumbs ``textsplitter`` builds. By
    default the headings are therefore rebuilt from their section numbering --
    see ``heading_levels.renumber_headings`` for what that fixes and how.

    Args:
        pdf_path: path to the input .pdf file.
        image_dir: directory for the extracted images. Defaults to
            "<pdf_basename>_files" beside the .pdf file.
        models: a model dict from ``load_pdf_models()``. If None, models are
            loaded for this single call (slow; prefer passing a shared dict when
            converting more than one PDF).
        page_range: optional marker page-range string (e.g. "0-5" or "0,2,4")
            to convert only part of the document; None converts everything.
        fix_heading_levels: rebuild heading levels from section numbering.
            Set False to get marker's raw output (e.g. to compare the two).

    Returns:
        The Markdown as a string.
    """
    # marker pulls in torch and a stack of vision models at import time, so the
    # imports stay inside the function (mirrors the lazy-import note on the
    # LaTeXML path's heavy deps).
    try:
        from marker.converters.pdf import PdfConverter
        from marker.config.parser import ConfigParser
        from marker.output import text_from_rendered
    except ImportError as exc:
        raise RuntimeError(
            "marker not installed. Install with `pip install marker-pdf`."
        ) from exc

    pdf_path = os.path.abspath(pdf_path)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(pdf_path)

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    if image_dir is None:
        image_dir = os.path.join(os.path.dirname(pdf_path), base + "_files")
    os.makedirs(image_dir, exist_ok=True)

    if models is None:
        models = load_pdf_models()

    # ConfigParser turns a plain options dict into the config/processor/renderer
    # trio PdfConverter expects. Forcing the markdown renderer guarantees the
    # ![](...) image refs (and the (markdown, "md", images) tuple) below.
    cli_options = {"output_format": "markdown"}
    if page_range is not None:
        cli_options["page_range"] = page_range
    config_parser = ConfigParser(cli_options)

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=models,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )
    rendered = converter(pdf_path)

    # images is {filename: PIL.Image}; the markdown already references each by
    # its bare filename, so saving them into image_dir makes those refs resolve
    # relative to image_dir without rewriting the markdown.
    markdown, _ext, images = text_from_rendered(rendered)
    for filename, image in images.items():
        image.save(os.path.join(image_dir, filename))

    if fix_heading_levels:
        markdown, _stats = renumber_headings(markdown)

    return markdown


def pdf_to_markdown_mineru(pdf_path, image_dir=None, backend="pipeline",
                           lang="en"):
    """Convert a PDF to Markdown (with images) using MinerU.

    A drop-in alternative to ``pdf_to_markdown`` (the marker path): same
    signature shape, same output contract -- returns the Markdown as a string
    and writes every extracted figure into ``image_dir`` with bare-filename
    ``![](<name>)`` references, relative to ``image_dir``.

    Unlike marker's optional ``--use_llm`` refinement, MinerU needs no external
    LLM: its quality comes from offline, locally-downloaded models (layout,
    formula, table, OCR for ``pipeline``; a small self-hosted VLM for the
    ``*-engine`` backends). Nothing leaves the machine.

    MinerU's Python API shifts between releases, so -- like the LaTeXML path --
    this shells out to the stable ``mineru`` CLI and reads back what it wrote.
    The CLI loads its models on every invocation (slow), which is fine for the
    background ingestion this is meant for.

    Args:
        pdf_path: path to the input .pdf file.
        image_dir: directory for the extracted images. Defaults to
            "<pdf_basename>_files" beside the .pdf file.
        backend: MinerU backend. "pipeline" (default) is CPU-tolerant and needs
            no GPU; "vlm-engine"/"hybrid-engine" are higher quality but want a
            GPU. All run offline -- none call out to an external LLM API.
        lang: document language hint for the pipeline OCR stage (e.g. "en").

    Returns:
        The Markdown as a string.
    """
    if shutil.which("mineru") is None:
        raise RuntimeError(
            "'mineru' not found on PATH. Install with `pip install mineru` "
            "(first run downloads the offline models)."
        )

    pdf_path = os.path.abspath(pdf_path)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(pdf_path)

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    if image_dir is None:
        image_dir = os.path.join(os.path.dirname(pdf_path), base + "_files")
    os.makedirs(image_dir, exist_ok=True)

    # MinerU writes a whole tree of artifacts (the .md plus *_middle.json,
    # *_layout.pdf, an images/ dir, etc.) into <out>/<base>/<method>/. We only
    # want the markdown and the figures, so let it write into a scratch dir and
    # copy the two pieces out, leaving the debug artifacts to be discarded.
    with tempfile.TemporaryDirectory() as out_dir:
        cmd = ["mineru", "-p", pdf_path, "-o", out_dir, "-b", backend]
        if lang:
            cmd += ["-l", lang]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        # The exact subfolder (auto/ocr/txt for pipeline, vlm for the VLM
        # backends) depends on backend/method, so locate the markdown by name
        # rather than hardcoding the path. Like the LaTeXML path, success is
        # "the output file exists", not the exit code -- MinerU can report
        # recoverable per-figure problems yet still emit usable markdown.
        hits = glob.glob(os.path.join(out_dir, "**", base + ".md"),
                         recursive=True)
        if not hits:
            log = (proc.stdout + proc.stderr).strip()
            raise RuntimeError(
                f"mineru exited {proc.returncode} and wrote no {base}.md.\n"
                f"{log}"
            )
        if proc.returncode != 0:
            warnings.warn(
                f"mineru reported problems but produced {base}.md; "
                f"continuing.\n{(proc.stdout + proc.stderr).strip()}",
                stacklevel=2,
            )

        md_path = hits[0]
        markdown = open(md_path, encoding="utf-8").read()

        # MinerU stores figures in an images/ dir beside the .md and references
        # them as ![](images/<name>). Flatten them into image_dir and drop the
        # "images/" prefix so the refs match the marker path's bare-name form.
        src_images = os.path.join(os.path.dirname(md_path), "images")
        if os.path.isdir(src_images):
            for name in os.listdir(src_images):
                shutil.copy2(os.path.join(src_images, name),
                             os.path.join(image_dir, name))
        markdown = re.sub(r"(!\[[^\]]*\]\()images/", r"\1", markdown)

    return markdown


def run_jobs(jobs_path, timeout=None):
    """Convert a batch of PDFs with one load of marker's models.

    ``jobs_path`` is a JSON list of ``{"id", "pdf", "markdown", "image_dir"}``.
    Progress goes to stdout as one JSON object per line -- ``{"event": "start",
    "id": ...}`` before each PDF and ``{"event": "done", "id": ..., "ok": ...,
    "seconds": ..., "error": ...}`` after it -- which is what update_corpus.py
    reads. It runs this in a separate Python because marker and its torch need
    not be the ones in the chatbot's venv. One PDF failing does not stop the
    rest; ``timeout`` (seconds) bounds each one.
    """
    import json
    import signal
    import time

    with open(jobs_path, encoding="utf-8") as fh:
        jobs = json.load(fh)

    def emit(**msg):
        print(json.dumps(msg), flush=True)

    def on_alarm(signum, frame):
        raise TimeoutError(f"not finished within {timeout} s")

    signal.signal(signal.SIGALRM, on_alarm)
    models = load_pdf_models()
    for job in jobs:
        emit(event="start", id=job["id"])
        t0 = time.time()
        if timeout:
            signal.alarm(int(timeout))
        try:
            markdown = pdf_to_markdown(job["pdf"], image_dir=job["image_dir"], models=models)
            with open(job["markdown"], "w", encoding="utf-8") as fh:
                fh.write(markdown)
            result = {"ok": True}
        except Exception as exc:  # one bad PDF must not take the batch down
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
        finally:
            signal.alarm(0)
        emit(event="done", id=job["id"], seconds=round(time.time() - t0, 1), **result)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="PDF to markdown with marker.")
    ap.add_argument("pdf", nargs="?", help="convert one PDF and print the markdown")
    ap.add_argument("--jobs", help="JSON list of conversions to run with one model load")
    ap.add_argument("--timeout", type=float, help="seconds allowed per PDF (with --jobs)")
    args = ap.parse_args()
    if args.jobs:
        run_jobs(args.jobs, args.timeout)
    elif args.pdf:
        print(pdf_to_markdown(args.pdf))
    else:
        ap.error("give a PDF or --jobs")
