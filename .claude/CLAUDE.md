# The project: Building a chatbot and possible MCP server for PLATO Pub

## Background
The PLATO pub website https://platopub.phys.au.dk/about/platopub.php , led by Mikkel Lund and hosted at Aarhus University provides information about the PLATO space telescope.  PLATO is being build by a large consortium and will probably launch late 2026 or early 2027. For the website all publications about PLATO are collected in one place: https://platopub.phys.au.dk/about/ads_feed.php . Currently these are papers about the development and expected capabilities of PLATO. Once the telescope becomes operational, this will shift towards papers presenting results from PLATO observations.

In this project we will make the information from the PLATO publications available through a chatbot, and possibly also an MCP server. The goal is to encourage and facilitate researchers' use of PLATO data for high quality, impactful results, by reducing the time and effort they need to spend on understanding the technical details of the instrument and data. This way they can focus on the science.

## Repository layout

The whole PlatoPub folder is one git repo (remote: hireck/platochat, private).

* `platochat/` — the chatbot itself: `plato_core.py` (the UI-agnostic core), `plato_api.py` (the FastAPI backend), `plato_chat.py` (the Streamlit UI), and `.env` with the API keys
* `ingest/` — everything that turns papers into indexed chunks: `latexml_to_markdown.py` and `pdf_to_markdown.py` (source to markdown), `heading_levels.py` (repairing the heading levels marker gets wrong), `textsplitter.py` (chunking), `fetch_metadata.py` (the PLATO-Pub list plus full ADS metadata, written to `papers.json`), `reindex_weaviate.py` (chunk and load into Weaviate), plus the paper metadata JSONs — `papers.json` is the current one; `PLATOChat_papers.json` and `present_papers.json` are from March 2024 and only still used by the older scripts. `get_present_papers.py` and `chunk_tex_data_plato.py` are older code from the FAISS days, kept for reference for their markdown and chunking heuristics — the FAISS index itself and the cells that built and queried it are gone; they still expect to be run from the data directory rather than from the repo.
* `eval/` — `questions.json` (42 questions with the paper(s) that should answer each, and patterns the answer must contain) and `run_eval.py`, which scores retrieval alone (`make eval`, ~1 min, no LLM) or whole answers (`make eval-full`, ~10 min, needs the AU network). Run it before and after anything that touches retrieval, chunking or the prompts. Results land in `eval/results/` (not tracked).
* `local_site/` — a local copy of the PLATO Pub pages, for developing the chatbot front-end against
* `Makefile` — the dev runner: `make dev` starts the API and the static site together, `make metadata` refreshes `papers.json`, `make reindex` rebuilds the Weaviate collection, `make eval` runs the evaluation

## The chatbot

The PLATO chatbot shall be part of the PLAT Pub website. We'll probably want to run it in a docker.

### Information sources

* platochat_info.md: a short document with high level information about PLATO (from the website, maybe pull it from ther directly in future?) and about the chatbot itself
* a full-text RAG index with hybrid retrieval on chunks to answer detailed questions. 
* a whole-paper level index that lets users search papers by metadata, like author names and publication date
* maybe later add ESA's PLATO mission website for information about observing proposals and data access

Important: older information can be outdated. The publication date of retrieved information needs to always be made available, and the chatbot needs to be instructed to prefer more recent information.

How this is done: every chunk carries its paper's `pubdate` (from ADS); the model sees it in each passage's metadata and the system prompt tells it to go with the most recent passage where they disagree and to say so; the Sources panel shows the year and month and links each title to its ADS page (plus publisher and arXiv links). Deliberately *not* done: boosting recent chunks in the ranking, which would bury the definitive older paper on questions where age is irrelevant. The `recency` questions in the eval set (camera count, field of view, launch date — all of which changed between the 2014 and the current design) are there to show whether the prompt alone is enough.

General knowledge (decided 2026-09-21): the papers come first, but an astronomy-related question they do not cover may be answered from the LLM's general knowledge, **provided it is flagged to the user**. The system prompt makes the model say that the publications do not cover the point and put the general-knowledge part in its own paragraph under a fixed label (`GENERAL_KNOWLEDGE_LABEL` in `plato_core.py`: "General knowledge, not from the PLATO publications"), without citations. The wording is fixed rather than left to the model so that users learn to recognise it and the eval can check for it, in both directions: present where it should be (`expect_flag: true`), absent from answers the papers fully support (`expect_flag: false`). Questions unrelated to astronomy or PLATO are declined. The label matters most for facts about PLATO itself, where the model's training data can be years behind the papers.

Passage ids: chunks are stored under a stable `chunk_id` (`<bibcode>#0017`, position within the paper) so that adding or removing a paper does not renumber the rest. The model never sees those; it cites passages by their position in the current request (1, 2, 3, ...), which is easier for it to copy correctly.

Figures: We want to have the option to have figures from the paper displayed if they seem relevant for the answer, based on the caption and/or the mention of the figure in the cited text. (Extracting information from images directly is not a priority.)

### Data ingestion

The papers are listed on https://platopub.phys.au.dk/about/ads_feed.php . That page is an empty shell filled by JavaScript from a JSON endpoint, `https://platopub.phys.au.dk/about/ads_middleware.php?page=1&rows=1000&year=&author=&keywords=`, which is what `fetch_metadata.py` reads — no HTML scraping. The list is itself an ADS query (published papers only, i.e. no arXiv-only bibcodes; "PLATO" in title and abstract; "space" in the full text) and shows a "last updated" date of the same day, so it appears to refresh daily.

Bibcodes are not permanent: a preprint (`2024arXiv240104503D`) or an early-access paper (`2026MNRAS.tmp.1093M`) gets a new bibcode on publication, and the list can briefly carry both. ADS keeps the old ones as aliases of the new record, and `papers.json` stores them under `aliases`. `reindex_weaviate.py` uses those to match a markdown file converted under an old bibcode to the paper's current record; a markdown file that matches nothing is not on the list and is left out of the index.

The arXiv source bundle can contain more than one compilable `.tex` file: for Bray et al. 2023 the MNRAS author guide (`mnras_guide.tex`) was converted in place of the paper, and its chunks sat in the index under the paper's name until the eval caught it. Whatever picks the main file in the future pipeline must not just take the first `.tex` with a `\documentclass`. As a backstop, `reindex_weaviate.py` refuses a markdown file that does not contain its own paper's title.

LaTeXML evaluates `\today` at conversion time, so A&A papers carry the conversion date in their front matter — "(June 21, 2026)" on a 2019 paper. Now that the model is told to weigh publication dates, that is a false date sitting in the text. Fixed 2026-09-21 in two places, both through `strip_false_dates()` in `latexml_to_markdown.py`: the converter removes a front-matter date line that falls within its own run (a range, not "today" — PlatoSim's conversion crossed midnight), and `reindex_weaviate.py` removes one dated more than a month after the paper's ADS publication date, which covers markdown converted before the fix (four A&A papers) without editing the files. MNRAS papers had a second false date, "pubyear: 2015" on a 2021 paper — template debris LaTeXML leaves in the keyword block; it went when keyword blocks were dropped from the index.

LaTeXML writes "Abstract", "Key Words.:" and "Acknowledgements." as level-6 headings, so the chunker splits on all six levels. Before it did (it stopped at five), abstract chunks had no section name — the Sources line fell back to raw chunk text, which is how the false date surfaced in a citation — and the acknowledgements of 15 papers were indexed as the tail of their last section instead of being dropped. Keyword blocks are dropped on purpose: as chunks of their own they are bare term lists a dozen tokens long, which BM25's preference for short documents would rank highly for any query sharing a term.

Their complete metadata is then to be collected via the ADS link. There is an ADS_API_KEY in platochat/.env . There was some old code that yielded /Users/hilke/data/plato_data/OneDrive_1_11-03-2024/PLATOChat_papers.json, but this did not include publication dates, and possibly some other fields are missing as well. We want to collect all the available information.

For papers that have an arxiv ID: If the latex source is available, we will process that (ingest/latexml_to_markdown.py), otherwise we will use marker to process the pdf (ingest/pdf_to_markdown.py).

We compared marker against MinerU on two PLATO papers (2026-09-20). Text quality is close — MinerU is somewhat faster and finds a few more figures — but we went with marker for operational reasons: MinerU 4.x is a client/server tool that defaults to sending PDFs to mineru.net (local parsing is `disabled` out of the box and needs a running server plus its own model download), it silently converts only the first 10 pages unless told otherwise, and it no longer writes image files at all — figures come back as opaque locators into its internal library. marker stays a plain in-process Python call that writes the figures beside the markdown. `pdf_to_markdown_mineru()` is kept for reference but is written against the MinerU 2.x CLI and will not run against 4.x.

marker's weak point is that it assigns heading levels almost arbitrarily, which corrupts the section breadcrumbs textsplitter builds. ingest/heading_levels.py rebuilds them from the section numbering and runs by default inside pdf_to_markdown().

Page furniture (running headers, footers, publisher stamps) needs no filter for now. Checked on seven papers (~200 pages, five publishers: ExA, EPJ Web of Conferences, A&A, SPIE, MNRAS): marker's layout model discards it on its own, and across 3046 lines exactly one instance survived — a venue heading, "EPJ Web of Conferences" on Ouazzani et al. 2015. Nothing leaked into body text. A filter was written and dropped again, because a rule fitted to one example is not evidence it would catch whatever turns up next. **Re-check this once the corpus is larger**, and fit any filter to what is actually found then.

Two things worth keeping from that exercise. First, the obvious heuristics are all dangerous and were each caught destroying real content: "a heading with nothing under it is furniture" flags every section opening with a subsection ("2. Modelling strategy" then "2.1. Seismic inversions"); "a bare number line is a page number" flags the wrapped page field of a citation ("... 2012, MNRAS, 425," / "757"); "a line naming a journal is furniture" flags bibliographies — all 240 journal-abbreviation hits in one paper were citations. Any future filter must leave everything from the references heading onward alone. Second, ADS stores the expanded journal name ("European Physical Journal Web of Conferences") while PDFs print the abbreviation ("EPJ Web of Conferences"), so matching a venue name needs overlapping words, not a substring test. The dropped implementation is kept at the git tag `experiment/page-furniture` (`git show experiment/page-furniture`).

Papers that are not on arxiv often have a link to for example the publisher's page where the paper can be downloaded. This will typically be pdf, or maybe html in some cases. We will need to confirm which formats the papers are in, and figure out how we can automatically download them. Typically we will need to find the download link on the page. Some of these papers will be pay-walled. Typically, the university has access to these papers, as they pay for them through a deal with the publisher. And users will typically have access too, through their own university. The chatbot will provide links to the original papers in their original location. Unpublished materials, such as internal documents will not be included. The chatbot will be public facing.

Once the chatbot is up and running, its sources will need to be kept up-to-date. The PLATO Pub list will need to be regularly checked for new papers. (I'll need to check with Mikkel with what frequency the list is updated.) When new papers appear, they need to be processed and added to the chunk index and whole-paper index. The original documents (pdf/latex/html) and the full markdown do not need to remain saved on the server, once processing is complete.

Papers that are dropped from the list (e.g. arxiv-only preprints that have been replaced with their published version) should ideally also be dropped from the index.

### Robustness and maintainablity

We'll need to monitor the chatbot, so we can fix it if something breaks, e.g. with PM2 and UpTimeRobot.

We rely on the LLMs from the university's AI Lab. If they change the version they serve, our chatbot breaks.

We may want to switch out Weaviate against Postgress with pgvector, as the latter as less likely to disappear without a clear alternative. Similarly we should review where else we rely on non-standard packages that might become a problem for maintenace in the coming decade or so.

For LLM observability, we use LangFuse, but this may also be a risky dependency.

## MCP Server

Many astronomers use tools like Claude code and cowork to explore the literature, write code for processing data, and write their papers. Making full-text index and paper index search available via mcp, will allow researchers to access up-to-date processed information from inside their own workflow, using their own LLM (subscription). 

# Communication
Explain everything in plain English to a linguist who has experience with writing python scripts for applications of NLP techniques, data processing, exploration, QA. This was mostly prototyping, rather than production. I have some half-knowledge about software development and UX from working in a startup in cross-functional teams, and would like to keep learning.