"""
PLATO chatbot — Streamlit UI (LEGACY).

This is now a thin presentation layer over the shared pipeline in
``plato_core.py``; all retrieval / routing / answering logic lives there and is
also served by the web API (``plato_api.py``). It is kept working during the
migration to the web front-end and can be deleted once that replaces it.

Run:  streamlit run plato_chat.py
"""

import streamlit as st
from langchain.memory import StreamlitChatMessageHistory
from langchain_core.messages.base import BaseMessage

import plato_core
from plato_core import HISTORY_WINDOW


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    expander = st.expander("About")
    expander.write(
        "This is a initial demo version of what a chatbot for PLATO could look "
        "like. It is a RAG system, using an LLM to generate answers based on "
        "relevant paragraphs provided by a retrieval component (a search engine "
        "using embeddings). The paragraphs come from a limited number of papers."
    )


# ---------------------------------------------------------------------------
# Pipeline call + rendering
# ---------------------------------------------------------------------------

def answer_question(user_input: str) -> None:
    """Send the question through the core pipeline and render the reply."""
    # History as core expects it: [{"role": "human"|"ai", "content": str}, ...],
    # taken *before* the current message is appended.
    history = [
        {"role": m.type, "content": m.content}
        for m in msgs.messages[-HISTORY_WINDOW:]
    ]
    msgs.add_message(BaseMessage(type="human", content=user_input))

    with st.spinner("🧠 Thinking…"):
        reply, sources = plato_core.answer_question(user_input, history)

    if not reply:
        st.write("Oops, something went wrong. Please try again.")
        return

    sources_text = sources or "No relevant sources were found."
    with st.chat_message("ai"):
        st.write(reply)
        with st.expander("See sources"):
            st.write(sources_text)

    ai_msg = BaseMessage(type="ai", content=reply)
    setattr(ai_msg, "sources", sources_text)
    msgs.add_message(ai_msg)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("PLATO chatbot")

if "clicked" not in st.session_state:
    st.session_state.clicked = False
if "chosen_example" not in st.session_state:
    st.session_state.chosen_example = ""


def hide_buttons(ex: str = "") -> None:
    st.session_state.clicked = True
    st.session_state.chosen_example = ex


msgs = StreamlitChatMessageHistory(key="langchain_messages")

if len(msgs.messages) == 0:
    msgs.add_message(
        BaseMessage(type="ai", content="Welcome to the PLATO chatbot – How can I help you?")
    )

# Replay history (keep each answer's sources expander).
for msg in msgs.messages:
    if msg.type == "ai" and hasattr(msg, "sources"):
        with st.chat_message("ai"):
            st.write(msg.content)
            with st.expander("See sources"):
                st.write(msg.sources)
    elif msg.type == "human":
        with st.chat_message("human"):
            st.write(msg.content)
    else:
        st.chat_message(msg.type).write(msg.content)

# Example prompts (only on a fresh conversation).
if len(msgs.messages) == 1 and not st.session_state.clicked:
    st.write(
        "Here are some examples of questions you can ask. Or you can ask your "
        "own question in the input field below"
    )
    for ex in ["When will PLATO become operational?", "What kind of science can I do with PLATO?"]:
        st.button(ex, on_click=hide_buttons, args=[ex])

st.chat_input(key="content", on_submit=hide_buttons)

if st.session_state.chosen_example:
    answer_question(st.session_state.chosen_example)
    st.session_state.chosen_example = ""
if content := st.session_state.content:
    answer_question(content)
