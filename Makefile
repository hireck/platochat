# PLATO-Pub chatbot — local dev runner.
#
#   make venv     create platochat/.venv and install requirements.txt
#   make dev      start the API (:8000) and the static site (:8090) together
#   make api      start just the FastAPI backend (:8000)
#   make web      start just the static PLATO-Pub site (:8090)
#   make streamlit run the legacy Streamlit UI instead of the web front-end
#   make kill     stop the API and static server
#   make urls     print the local URLs
#   make reindex  rebuild the Weaviate PLATO collection from the markdown corpus
#                 (DESTRUCTIVE — see the reindex target below)
#
# Prerequisites: Weaviate running locally (the PLATO collection must be loaded)
# and the env vars in platochat/.env (OPENWEBUI_API_KEY, LANGFUSE_*).

API_PORT  ?= 8000
WEB_PORT  ?= 8090
WEB_DIR   := local_site
API_DIR   := platochat

# The API deps live in a venv inside platochat (see the venv target below).
# Override with:  make api PYTHON=/path/to/other/python
VENV      := $(CURDIR)/$(API_DIR)/.venv
PYTHON    ?= $(VENV)/bin/python
# Interpreter used to CREATE the venv; the deps are pinned for 3.10.
BASE_PYTHON ?= python3.10
# The static site server only needs the stdlib, so it does not require the venv.
WEB_PYTHON ?= python3

.PHONY: dev api web streamlit venv check-venv kill urls reindex

venv:
	$(BASE_PYTHON) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(API_DIR)/requirements.txt
	@echo "venv ready: $(VENV)"

check-venv:
	@test -x $(PYTHON) || { echo "No venv at $(VENV) — run 'make venv' first."; exit 1; }

dev: check-venv
	@echo "Starting PLATO API (:$(API_PORT)) and static site (:$(WEB_PORT))…"
	@echo "Open  http://localhost:$(WEB_PORT)/chatbot.html  (Ctrl-C stops both)"
	@trap 'kill 0' INT TERM EXIT; \
		( cd $(API_DIR) && $(PYTHON) -m uvicorn plato_api:app --port $(API_PORT) ) & \
		( cd $(WEB_DIR) && $(WEB_PYTHON) -m http.server $(WEB_PORT) ) & \
		wait

api: check-venv
	cd $(API_DIR) && $(PYTHON) -m uvicorn plato_api:app --reload --port $(API_PORT)

web:
	@echo "Open  http://localhost:$(WEB_PORT)/chatbot.html"
	cd $(WEB_DIR) && $(WEB_PYTHON) -m http.server $(WEB_PORT)

streamlit: check-venv
	cd $(API_DIR) && $(PYTHON) -m streamlit run plato_chat.py

# Rebuild the vector index. Chunks the markdown corpus and re-embeds it with
# bge-m3, which is 1024-dim against the old 384, so the collection cannot be
# updated in place -- it is dropped and recreated. Takes a few minutes.
#
#   make reindex                              # preview only, writes nothing
#   make reindex ARGS="--collection PLATO_TEST"   # build alongside the live one
#   make reindex ARGS="--yes"                 # replace the live PLATO collection
#
# Defaults to --dry-run so a bare 'make reindex' cannot destroy the index.
ARGS ?= --dry-run
reindex: check-venv
	$(PYTHON) reindex_weaviate.py $(ARGS)

kill:
	-@pkill -f "uvicorn plato_api" 2>/dev/null || true
	-@pkill -f "http.server $(WEB_PORT)" 2>/dev/null || true
	@echo "Stopped API and static server."

urls:
	@echo "Web UI : http://localhost:$(WEB_PORT)/chatbot.html"
	@echo "API    : http://localhost:$(API_PORT)/api/chat  (health: /health)"
