# The project: Building a chatbot and possible MCP server for PLATO Pub

## Background
The PLATO pub website https://platopub.phys.au.dk/about/platopub.php , led by Mikkel Lund and hosted at Aarhus University provides information about the PLATO space telescope.  PLATO is being build by a large consortium and will probably launch late 2026 or early 2027. For the website all publications about PLATO are collected in one place: https://platopub.phys.au.dk/about/ads_feed.php . Currently these are papers about the development and expected capabilities of PLATO. Once the telescope becomes operational, this will shift towards papers presenting results from PLATO observations.

In this project we will make the information from the PLATO publications available through a chatbot, and possibly also an MCP server. The goal is to encourage and facilitate researchers' use of PLATO data for high quality, impactful results, by reducing the time and effort they need to spend on understanding the technical details of the instrument and data. This way they can focus on the science.

## Repository layout

The whole PlatoPub folder is one git repo (remote: hireck/platochat, private).

* `platochat/` — the chatbot itself: `plato_core.py` (the UI-agnostic core), `plato_api.py` (the FastAPI backend), `plato_chat.py` (the Streamlit UI), and `.env` with the API keys
* `ingest/` — everything that turns papers into indexed chunks: `latexml_to_markdown.py` and `pdf_to_markdown.py` (source to markdown), `textsplitter.py` (chunking), `reindex_weaviate.py` (chunk and load into Weaviate), plus the paper metadata JSONs. `get_present_papers.py` and `chunk_tex_data_plato.py` are older code from the FAISS days, kept for reference for their markdown and chunking heuristics — the FAISS index itself and the cells that built and queried it are gone; they still expect to be run from the data directory rather than from the repo.
* `local_site/` — a local copy of the PLATO Pub pages, for developing the chatbot front-end against
* `Makefile` — the dev runner: `make dev` starts the API and the static site together, `make reindex` rebuilds the Weaviate collection

## The chatbot

The PLATO chatbot shall be part of the PLAT Pub website. We'll probably want to run it in a docker.

### Information sources

* platochat_info.md: a short document with high level information about PLATO (from the website, maybe pull it from ther directly in future?) and about the chatbot itself
* a full-text RAG index with hybrid retrieval on chunks to answer detailed questions. 
* a whole-paper level index that lets users search papers by metadata, like author names and publication date
* maybe later add ESA's PLATO mission website for information about observing proposals and data access

Important: older information can be outdated. The publication date of retrieved information needs to always be made available, and the chatbot needs to be instructed to prefer more recent information.

Figures: We want to have the option to have figures from the paper displayed if they seem relevant for the answer, based on the caption and/or the mention of the figure in the cited text. (Extracting information from images directly is not a priority.)

### Data ingestion

The papers are listed on https://platopub.phys.au.dk/about/ads_feed.php . 
Their complete metadata is then to be collected via the ADS link. There is an ADS_API_KEY in platochat/.env . There was some old code that yielded /Users/hilke/data/plato_data/OneDrive_1_11-03-2024/PLATOChat_papers.json, but this did not include publication dates, and possibly some other fields are missing as well. We want to collect all the available information.

For papers that have an arxiv ID: If the latex source is available, we will process that (ingest/latexml_to_markdown.py), otherwise we will use u-miner to process the pdf (ingest/pdf_to_markdown.py).

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