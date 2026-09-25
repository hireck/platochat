# PLATO-Pub local site sandbox (chatbot dev)

A local, offline copy of the **relevant parts** of the public PLATO-Pub site
(`https://platopub.phys.au.dk/about/platopub.php`) so the chatbot UI can be
developed and tested locally — with a native HTML/JS chat panel **instead of
Streamlit**, living exactly where it would sit on the real site.

## What's here

| Path | Source | Notes |
|---|---|---|
| `chatbot.html` | new | The chatbot page (`/about/chatbot.php` on the site): a faithful copy of the site chrome (header, top nav, sidebar) with a new **PLATO Chatbot** tab and sidebar entry, hosting the chat widget in the `#contents` area. |
| `chatbot_about.html` | **new** | The chatbot's About page (`/about/chatbot_about.php`): what it knows, how to read its answers, what it cannot do. Linked from the sidebar, under PLATO Chatbot, and from the foot of the chat page. |
| `css/style.css`, `css/unsemantic-grid-responsive-tablet.css` | mirrored from the live site | The real site styling — do not edit. |
| `js/scripts.js` | mirrored | The site's own JS (mobile menu etc.). Its `$.tablesorter` call fails here, because that plugin is not copied; the error in the console is harmless. |
| `css/fonts/*`, `images/plato-logo-noname.png`, `favicon.ico` | mirrored | Assets referenced by `style.css`. |
| `css/chatbot.css` | **new** | The chat widget styling. Matches the site palette. |
| `js/chatbot.js` | **new** | The chat controller (replaces Streamlit). |
| `serve.py` | dev only | The local web server (see below). Not for the site. |

Stylesheets use absolute paths (`/css/...`), so **serve this folder as the web
root**.

The pages link to each other by their URLs on the site (`/about/chatbot.php`,
`/about/chatbot_about.php`), so their markup can be copied over unchanged.
`serve.py` answers those URLs from the `.html` files here, and sends any other
`.php` link (Home, Published PLATO Papers, New member ...) to the live site.

The sidebar is the site's own, with two entries added under "PLATO-Pub open
access": **PLATO Chatbot**, beside Published PLATO Papers, and **About the
chatbot** under it. On phones the tabs are hidden and the Menu button opens the
sidebar, so there the sidebar entry is the only way to the chatbot.

## Run it

From the **Platopub repo root** (one level up):

```bash
make dev      # starts the API (:8000) + this static site (:8090) together
# then open http://localhost:8090/about/chatbot.php
make kill     # stops both
```

Or serve just the static pages (UI-only, no backend):

```bash
make web                  # http://localhost:8090/about/chatbot.php
make web WEB_PORT=8091    # a second copy, beside a running `make dev`
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

The API is stateless: the page sends the last few messages with every
question. The conversation itself is kept in the tab's `sessionStorage`, so
that it survives the visitor following a link in the menus and coming back,
or reloading; closing the tab clears it, and so does the **New conversation**
button under the input. An exchange is stored once its answer is in.

> Note: the live site is PHP. On the server these pages become
> `/about/chatbot.php` and `/about/chatbot_about.php`, reusing the site's
> header and sidebar includes; the two sidebar entries go into the shared
> sidebar, so that every page shows them. `chatbot.html` and
> `chatbot_about.html` here are the static stand-ins for local dev. The About
> page has two highlighted placeholders (what is logged and for how long; whom
> to contact) to fill in before it goes public.
