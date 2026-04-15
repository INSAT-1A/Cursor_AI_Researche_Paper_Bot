import json
import logging
import os
from datetime import datetime

import streamlit as st
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

logger = logging.getLogger("streamlit_app")


st.set_page_config(
    page_title="Cursor Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


CURSOR_STYLE = """
<style>
:root {
    --cursor-bg: #0f111a;
    --cursor-panel: #161924;
    --cursor-border: #2a2f3a;
    --cursor-text: #e6e6e6;
    --cursor-muted: #a8afc1;
    --cursor-accent: #4f8cff;
}

.stApp {
    background-color: var(--cursor-bg);
    color: var(--cursor-text);
}

[data-testid="stSidebar"] {
    background-color: var(--cursor-panel);
    border-right: 1px solid var(--cursor-border);
}

.chat-shell {
    border: 1px solid var(--cursor-border);
    border-radius: 14px;
    padding: 0.7rem 1rem;
    background: rgba(255, 255, 255, 0.02);
    margin-bottom: 0.6rem;
}

.chat-user {
    border-left: 3px solid var(--cursor-accent);
}

.chat-assistant {
    border-left: 3px solid #7f8aa3;
}

.chat-role {
    color: var(--cursor-muted);
    font-size: 0.78rem;
    margin-bottom: 0.2rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.footer-note {
    color: var(--cursor-muted);
    font-size: 0.8rem;
    margin-top: 1rem;
}
</style>
"""
st.markdown(CURSOR_STYLE, unsafe_allow_html=True)


def ensure_state() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = {"New Chat": []}
    if "active_chat" not in st.session_state:
        st.session_state.active_chat = "New Chat"
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = "You are a helpful assistant."
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "top_p" not in st.session_state:
        st.session_state.top_p = 1.0
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 700


def get_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def render_message(role: str, content: str) -> None:
    role_class = "chat-user" if role == "user" else "chat-assistant"
    st.markdown(
        f"""
        <div class="chat-shell {role_class}">
            <div class="chat-role">{role}</div>
            <div>{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def conversation_for_model() -> list[dict]:
    messages = [{"role": "system", "content": st.session_state.system_prompt}]
    messages.extend(st.session_state.conversations[st.session_state.active_chat])
    return messages


def create_new_chat() -> None:
    name = f"Chat {len(st.session_state.conversations) + 1}"
    st.session_state.conversations[name] = []
    st.session_state.active_chat = name


def export_active_chat() -> str:
    payload = {
        "chat_name": st.session_state.active_chat,
        "exported_at": datetime.now().isoformat(),
        "messages": st.session_state.conversations[st.session_state.active_chat],
    }
    return json.dumps(payload, indent=2)


ensure_state()

with st.sidebar:
    st.title("Cursor Chat")
    st.caption("ChatGPT-like UI in Streamlit")

    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()

    chat_names = list(st.session_state.conversations.keys())
    st.session_state.active_chat = st.selectbox(
        "Conversation",
        options=chat_names,
        index=chat_names.index(st.session_state.active_chat),
    )

    st.divider()
    st.subheader("Model Settings")
    model = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"],
        index=0,
    )
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.5, st.session_state.temperature, 0.1)
    st.session_state.top_p = st.slider("Top P", 0.1, 1.0, st.session_state.top_p, 0.1)
    st.session_state.max_tokens = st.slider("Max Tokens", 128, 2048, st.session_state.max_tokens, 64)
    st.session_state.system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=90,
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.conversations[st.session_state.active_chat] = []
    with col2:
        st.download_button(
            "⬇ Export",
            data=export_active_chat(),
            file_name=f"{st.session_state.active_chat.replace(' ', '_').lower()}.json",
            mime="application/json",
            use_container_width=True,
        )


st.title("ChatGPT-style Assistant")
st.caption("Cursor-inspired theme + streaming responses")

for msg in st.session_state.conversations[st.session_state.active_chat]:
    render_message(msg["role"], msg["content"])

client = get_client()
if not client:
    st.warning("Set `OPENAI_API_KEY` to enable model responses.")

prompt = st.chat_input("Message Cursor Chat...")
if prompt:
    st.session_state.conversations[st.session_state.active_chat].append({"role": "user", "content": prompt})
    render_message("user", prompt)

    if client:
        response_placeholder = st.empty()
        assembled = ""
        try:
            with st.spinner("Thinking..."):
                stream = client.chat.completions.create(
                    model=model,
                    messages=conversation_for_model(),
                    temperature=st.session_state.temperature,
                    top_p=st.session_state.top_p,
                    max_tokens=st.session_state.max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        assembled += delta
                        response_placeholder.markdown(
                            f"""
                            <div class="chat-shell chat-assistant">
                                <div class="chat-role">assistant</div>
                                <div>{assembled}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        except RateLimitError as exc:
            logger.warning("OpenAI rate limit: %s", exc)
            st.error("The API rate limit was hit. Wait a moment and try again.")
            assembled = ""
        except APITimeoutError as exc:
            logger.warning("OpenAI timeout: %s", exc)
            st.error("The request timed out. Try again with a shorter message or lower max tokens.")
            assembled = ""
        except APIError as exc:
            logger.exception("OpenAI API error")
            st.error(f"API error: {exc}")
            assembled = assembled or ""
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception("Network or parse error during chat")
            st.error("Something went wrong while talking to the model. Check your connection and try again.")
            assembled = assembled or ""
        except Exception as exc:
            logger.exception("Unexpected error during chat completion")
            st.error(f"Unexpected error: {exc}")
            assembled = assembled or ""

        st.session_state.conversations[st.session_state.active_chat].append(
            {"role": "assistant", "content": assembled}
        )

st.markdown(
    '<div class="footer-note">Tip: Use the sidebar to start chats, tweak model settings, and export history.</div>',
    unsafe_allow_html=True,
)
