# PLATO-Pub local site sandbox (chatbot dev)

A local, offline copy of the **relevant parts** of the public PLATO-Pub site
(`https://platopub.phys.au.dk/about/platopub.php`) so the chatbot UI can be
developed and tested locally — with a native HTML/JS chat panel **instead of
Streamlit**, living exactly where it would sit on the real site.

## What's here

| Path | Source | Notes |
|---|---|---|
| `chatbot.html` | new | The chatbot page: a faithful copy of the site chrome (header, top nav, sidebar) with a new **PLATO Chatbot** sidebar entry, hosting the chat widget in the `#contents` area. |
| `css/style.css`, `css/unsemantic-grid-responsive-tablet.css` | mirrored from the live site | The real site styling — do not edit. |
| `js/scripts.js` | mirrored | The site's own JS (mobile menu etc.). |
| `css/fonts/*`, `images/plato-logo-noname.png`, `favicon.ico` | mirrored | Assets referenced by `style.css`. |
| `css/chatbot.css` | **new** | The chat widget styling. Matches the site palette. |
| `js/chatbot.js` | **new** | The chat controller (replaces Streamlit). |

Stylesheets use absolute paths (`/css/...`), so **serve this folder as the web
root**.

## Run it

From the **Platopub repo root** (one level up):

```bash
make dev      # starts the API (:8000) + this static site (:8090) together
# then open http://localhost:8090/chatbot.html
make kill     # stops both
```

Or serve just the static page (UI-only, no backend) from this folder:

```bash
cd local_site && python3 -m http.server 8090
```

With **no backend running** the page still works — it shows mock replies so you
can iterate on the UI. jQuery/jQuery-UI/fonts/marked/MathJax load from CDNs (so
you need internet for those); everything PLATO-Pub-specific is local.

The backend needs **Weaviate running** (with the `PLATO` collection loaded) and
the env vars in `../platochat/.env`.

## The backend

`js/chatbot.js` POSTs to `window.PLATO_CHAT_API` (default
`http://localhost:8000/api/chat`):

```
request : { "message": str, "history": [ {"role":"human"|"ai","content":str}, ... ] }
response: { "reply": "<markdown>", "sources": "<markdown>"|null }
```

This is served by `../platochat/plato_api.py` (FastAPI), a thin wrapper over the
shared RAG pipeline in `../platochat/plato_core.py` (`route` →
`tool_search_publications` → answer → `used_sources`/`build_sources_text`,
returning `reply` + `sources` strings). The legacy Streamlit UI
(`plato_chat.py`) is now just another thin client of the same `plato_core`.

> Note: the live site is PHP. On the server this page becomes `chatbot.php`
> reusing the same header/sidebar includes; `chatbot.html` here is the static
> stand-in for local dev.
