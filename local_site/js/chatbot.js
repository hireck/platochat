/*
 * PLATO Chatbot — front-end controller (replaces the Streamlit UI).
 *
 * Talks to a small HTTP backend that wraps the existing RAG pipeline
 * (route -> retrieve+rerank -> answer) from platochat/plato_chat.py.
 *
 * Backend contract (JSON):
 *   POST {CHAT_API}
 *   request : { "message": str, "history": [ { "role": "human"|"ai", "content": str }, ... ] }
 *   response: { "reply": "<markdown>", "sources": "<markdown>" }
 *
 * If the backend is unreachable the widget falls back to MOCK replies so the
 * page can be developed/tested on its own.
 */
(function () {
	"use strict";

	// Point this at the FastAPI endpoint you build from plato_chat.py.
	// Override at runtime with:  window.PLATO_CHAT_API = "http://host:port/api/chat"
	var CHAT_API = window.PLATO_CHAT_API || "http://localhost:8000/api/chat";
	var HISTORY_WINDOW = 8; // messages sent back to the server for context

	var log = document.getElementById("chat-log");
	var form = document.getElementById("chat-form");
	var input = document.getElementById("chat-input");
	var sendBtn = document.getElementById("chat-send");
	var statusEl = document.getElementById("chat-status");
	var examples = document.getElementById("chat-examples");

	var history = []; // [{role:'human'|'ai', content:str}]

	// ---- rendering helpers ---------------------------------------------------

	function typeset(el) {
		if (window.MathJax && window.MathJax.typesetPromise) {
			window.MathJax.typesetPromise([el]).catch(function () {});
		}
	}

	// Only what markdown itself produces is allowed through: no SVG/MathML (the
	// math is typeset by MathJax afterwards, from the $...$ text), and no CSS or
	// form controls, which cannot run code but can restyle the page or fake a
	// login box.
	var SANITIZE_OPTIONS = {
		USE_PROFILES: { html: true },
		FORBID_TAGS: ["style", "form", "input", "button", "textarea", "select"],
		FORBID_ATTR: ["style"]
	};

	// marked passes raw HTML in its input straight through, and that input is
	// text we do not control -- the LLM's answer, and through it passages from
	// the indexed papers. So marked's output is only used once DOMPurify has
	// stripped what could execute (<script>, onerror=, javascript: links). If
	// either library failed to load, the text is shown escaped instead: plain,
	// but inert.
	function renderMarkdown(text) {
		try {
			if (window.marked && window.DOMPurify && window.DOMPurify.isSupported) {
				return window.DOMPurify.sanitize(window.marked.parse(text), SANITIZE_OPTIONS);
			}
		} catch (e) {}
		return escapeHtml(text);
	}

	function escapeHtml(s) {
		return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
	}

	function addMessage(role, markdown, sourcesMarkdown) {
		var row = document.createElement("div");
		row.className = "chat-msg " + role;

		var bubble = document.createElement("div");
		bubble.className = "chat-bubble";
		if (role === "human") {
			// What the visitor typed is shown as typed, never parsed as markup.
			bubble.textContent = markdown;
		} else {
			bubble.innerHTML = renderMarkdown(markdown);
		}
		row.appendChild(bubble);

		if (role === "ai" && sourcesMarkdown) {
			var details = document.createElement("details");
			details.className = "chat-sources";
			var summary = document.createElement("summary");
			summary.textContent = "See sources";
			var body = document.createElement("div");
			body.className = "sources-body";
			body.innerHTML = renderMarkdown(sourcesMarkdown);
			details.appendChild(summary);
			details.appendChild(body);
			bubble.appendChild(details);
		}

		// The sources link out to ADS and the publishers. The conversation lives
		// only in this page's memory, so following a link in the same tab would
		// lose it -- open them in a new one.
		var links = bubble.querySelectorAll("a[href]");
		for (var i = 0; i < links.length; i++) {
			links[i].target = "_blank";
			links[i].rel = "noopener noreferrer";
		}

		log.appendChild(row);
		typeset(bubble);
		log.scrollTop = log.scrollHeight;
		return bubble;
	}

	function addThinking() {
		var row = document.createElement("div");
		row.className = "chat-msg ai";
		row.innerHTML =
			'<div class="chat-bubble thinking">Analysing your question' +
			'<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>';
		log.appendChild(row);
		log.scrollTop = log.scrollHeight;
		return row;
	}

	// ---- network -------------------------------------------------------------

	function callBackend(message) {
		return fetch(CHAT_API, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				message: message,
				history: history.slice(-HISTORY_WINDOW)
			})
		}).then(function (r) {
			if (!r.ok) throw new Error("HTTP " + r.status);
			return r.json();
		});
	}

	function mockReply(message) {
		return Promise.resolve({
			reply:
				"**(mock reply — backend not connected)**\n\nYou asked: _" +
				message +
				"_\n\nOnce the FastAPI backend wrapping `plato_chat.py` is running at `" +
				CHAT_API +
				"`, real answers with citations like [1] will appear here.",
			sources: "[1] *Example paper* — Section: Introduction"
		});
	}

	// ---- conversation flow ---------------------------------------------------

	function send(message) {
		message = (message || "").trim();
		if (!message) return;

		if (examples) examples.style.display = "none";
		addMessage("human", message);
		history.push({ role: "human", content: message });

		input.value = "";
		input.disabled = true;
		sendBtn.disabled = true;
		statusEl.textContent = "";
		var thinking = addThinking();

		callBackend(message)
			.catch(function (err) {
				statusEl.textContent = "Backend unreachable (" + err.message + ") — showing mock reply.";
				return mockReply(message);
			})
			.then(function (data) {
				log.removeChild(thinking);
				var reply = (data && data.reply) || "Sorry, something went wrong. Please try again.";
				var sources = data && data.sources;
				addMessage("ai", reply, sources);
				history.push({ role: "ai", content: reply });
			})
			.finally(function () {
				input.disabled = false;
				sendBtn.disabled = false;
				input.focus();
			});
	}

	// ---- wiring --------------------------------------------------------------

	form.addEventListener("submit", function (e) {
		e.preventDefault();
		send(input.value);
	});

	Array.prototype.forEach.call(document.querySelectorAll(".example-btn"), function (btn) {
		btn.addEventListener("click", function () {
			send(btn.textContent);
		});
	});

	// greeting
	addMessage("ai", "Welcome to the PLATO chatbot — how can I help you?");
})();
