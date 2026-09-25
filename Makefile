# PLATO-Pub chatbot — local dev runner.
#
#   make venv     create platochat/.venv and install requirements.txt
#   make dev      start the API (:8000) and the static site (:8090) together
#   make api      start just the FastAPI backend (:8000)
#   make web      start just the static PLATO-Pub site (:8090)
#   make streamlit run the legacy Streamlit UI instead of the web front-end
#   make kill     stop the API and static server
#   make urls     print the local URLs
#   make metadata refresh ingest/papers.json from the PLATO-Pub list and ADS
#   make update   bring the corpus up to date: refresh the list, fetch and convert
#                 new full texts, update the index paper by paper (what cron runs)
#   make status   which papers have full text, which are abstract only, what failed
#   make licences look up under what licence each paper is published, and report
#   make reindex  rebuild the Weaviate PLATO collection from the markdown corpus
#                 (DESTRUCTIVE — see the reindex target below)
#   make papers   bring the whole-paper index (PLATO_PAPERS) up to date on its own
#   make eval     score retrieval against eval/questions.json (no LLM, ~1 min)
#   make eval-full  score whole answers too (needs the AU network, ~10 min)
#
# Prerequisites: Weaviate running locally (the PLATO collection must be loaded)
# and the env vars in platochat/.env (OPENWEBUI_API_KEY, LANGFUSE_*).

API_PORT  ?= 8000
WEB_PORT  ?= 8090
WEB_DIR   := local_site
API_DIR   := platochat
# The corpus-prep scripts (markdown conversion, chunking, Weaviate load).
INGEST_DIR := ingest

# The API deps live in a venv inside platochat (see the venv target below).
# Override with:  make api PYTHON=/path/to/other/python
VENV      := $(CURDIR)/$(API_DIR)/.venv
PYTHON    ?= $(VENV)/bin/python
# Interpreter used to CREATE the venv; the deps are pinned for 3.13. (3.10 had
# to go with macOS 27: its last SciPy, 1.15.3, ships Fortran modules the new
# loader refuses, and sentence-transformers cannot be imported without them.)
BASE_PYTHON ?= python3.13
# The static site server only needs the stdlib, so it does not require the venv.
WEB_PYTHON ?= python3

# The eval questions and results.
EVAL_DIR  := eval

.PHONY: dev api web streamlit venv check-venv kill urls reindex papers metadata update status licences eval eval-full

venv:
	$(BASE_PYTHON) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(API_DIR)/requirements.txt
	@echo "venv ready: $(VENV)"

check-venv:
	@test -x $(PYTHON) || { echo "No venv at $(VENV) — run 'make venv' first."; exit 1; }

dev: check-venv
	@echo "Starting PLATO API (:$(API_PORT)) and static site (:$(WEB_PORT))…"
	@echo "Open  http://localhost:$(WEB_PORT)/about/chatbot.php  (Ctrl-C stops both)"
	@trap 'kill 0' INT TERM EXIT; \
		( cd $(API_DIR) && $(PYTHON) -m uvicorn plato_api:app --port $(API_PORT) ) & \
		$(WEB_PYTHON) $(WEB_DIR)/serve.py $(WEB_PORT) & \
		wait

api: check-venv
	cd $(API_DIR) && $(PYTHON) -m uvicorn plato_api:app --reload --port $(API_PORT)

# The static site alone; the chat shows mock replies unless the API runs too.
# `make web WEB_PORT=8091` runs a second copy beside `make dev`. (The API only
# answers pages on :8090 -- see PLATO_CORS_ORIGINS in platochat/plato_api.py.)
web:
	@echo "Open  http://localhost:$(WEB_PORT)/about/chatbot.php"
	$(WEB_PYTHON) $(WEB_DIR)/serve.py $(WEB_PORT)

streamlit: check-venv
	cd $(API_DIR) && $(PYTHON) -m streamlit run plato_chat.py

# Rebuild the vector index from scratch: the collection is dropped and every
# paper chunked and embedded again (about ten minutes for 180 papers). The
# everyday path is `make update`, which rewrites only the papers that changed;
# a rebuild is for trying other chunking settings in a collection of its own.
#
#   make reindex                              # preview only, writes nothing
#   make reindex ARGS="--collection PLATO_TEST"   # build alongside the live one
#   make reindex ARGS="--yes"                 # replace the live PLATO collection
#
# Defaults to --dry-run so a bare 'make reindex' cannot destroy the index.
ARGS ?= --dry-run
reindex: check-venv
	$(PYTHON) $(INGEST_DIR)/reindex_weaviate.py $(ARGS)

# The whole-paper index behind the chatbot's find_papers tool: one object per
# paper (title, abstract, authors, date), in <collection>_PAPERS. `make update`
# and `make reindex` keep it up to date already; this is for building it on its
# own. Takes half a minute; only papers whose record changed are rewritten.
#   make papers PAPER_ARGS="--dry-run"
#   PLATO_COLLECTION=PLATO_NEXT make papers       # PLATO_NEXT_PAPERS
PAPER_ARGS ?=
papers: check-venv
	$(PYTHON) $(INGEST_DIR)/paper_index.py $(PAPER_ARGS)

# Which papers exist and what ADS knows about them (dates, DOIs, arXiv ids).
# Run before a reindex: the index only holds papers that are on the list.
#   make metadata META_ARGS="--dry-run"    # report only, write nothing
META_ARGS ?=
metadata: check-venv
	$(PYTHON) $(INGEST_DIR)/fetch_metadata.py $(META_ARGS)

# The whole ingest in one go, doing only what is not done yet: refresh the
# list, download and convert full texts that are missing (arXiv and open access
# only -- paywalled papers are indexed by their abstract), then write the
# objects of new or changed papers and remove those of papers that left the
# list. State lives in <data>/manifest.json and in the index itself; see
# ingest/update_corpus.py. ingest/cron_update.sh runs this nightly.
#   make update UPDATE_ARGS="--dry-run"                  # say what would be done
#   PLATO_COLLECTION=PLATO_TEST make update              # another collection
#   make update UPDATE_ARGS="--only <bibcode> --redo"    # fetch one paper again
# PDFs are converted by marker, which may live in another Python; set
# PLATO_MARKER_PYTHON if `marker_single` is not on the PATH.
UPDATE_ARGS ?=
update: check-venv
	$(PYTHON) $(INGEST_DIR)/update_corpus.py $(UPDATE_ARGS)

status: check-venv
	@$(PYTHON) $(INGEST_DIR)/update_corpus.py --status

# Under what licence each paper, and the copy of it we indexed, is published
# (arXiv, Crossref, DataCite, OpenAlex; see ingest/fetch_licences.py). Looks up
# new papers and those ADS reports something new about -- `make update` does
# the same every night -- and prints the report. Recorded in the manifest only.
#   make licences UPDATE_ARGS="--recheck-licences"    # look every paper up again
#   PLATO_COLLECTION=PLATO_NEXT make licences         # object counts from that collection
licences: check-venv
	$(PYTHON) $(INGEST_DIR)/update_corpus.py --stages licences $(UPDATE_ARGS)
	@$(PYTHON) $(INGEST_DIR)/fetch_licences.py --report

# Regression check for anything that touches retrieval or the prompts.
#   make eval EVAL_ARGS="--label hybrid --against eval/results/<earlier>/results.json"
#   PLATO_COLLECTION=PLATO_TEST make eval        # score a side-by-side collection
EVAL_ARGS ?=
eval: check-venv
	$(PYTHON) $(EVAL_DIR)/run_eval.py $(EVAL_ARGS)

eval-full: check-venv
	$(PYTHON) $(EVAL_DIR)/run_eval.py --full $(EVAL_ARGS)

kill:
	-@pkill -f "uvicorn plato_api" 2>/dev/null || true
	-@pkill -f "(http\.server|serve\.py) $(WEB_PORT)" 2>/dev/null || true
	@echo "Stopped API and static server."

urls:
	@echo "Web UI : http://localhost:$(WEB_PORT)/about/chatbot.php"
	@echo "API    : http://localhost:$(API_PORT)/api/chat  (health: /health)"
