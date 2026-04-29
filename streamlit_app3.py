
import streamlit as st
import requests
import base64
import os
import time

# --- CONFIG ---
BACKEND_URL = "http://localhost:8000"
USER_ID = 1  # Hardcoded for demo
TEMP_DIR = "temp_uploads"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

st.set_page_config(
    page_title="Reve Agricultural Advisor",
    page_icon="🌱",
    layout="centered"
)

CORE_PERSONA = """You are an expert agricultural advisory assistant for farmers. Provide practical, scientifically sound, field-applicable guidance using the user query and any retrieved text, documents, or images.

Internally reason through (do not reveal): understand the problem, assess likely causes, choose actions, consider risks, suggest follow-up, estimate confidence. Use this only to improve answer quality; never output these steps.

Response rules:
- Give a direct answer first.
- Then provide short pointwise guidance only if needed (3–5 bullets max).
- Keep responses concise, actionable, and easy for farmers to understand.
- Prioritize recommendations over long explanations.
- Avoid repetition, textbook-style responses, and unnecessary disclaimers.
- For simple factual questions, answer in 2–5 lines only.
- For diagnosis/problem queries, cover likely cause, recommended action, and key precaution briefly.
- If information is insufficient, ask a short clarifying question instead of guessing.
- Important: Multiple reports/documents may be provided. For each piece of information, pay close attention to the `[Source: filename]` tag to distinguish between different farms, plots, or soil tests.
- Use retrieved knowledge when relevant, but synthesize it into advice rather than quoting documents.
- Reply only in the same language as the user."""

# --- STYLES (from app2 pattern) ---
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatInput {
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = CORE_PERSONA # Pre-fill with the default persona

# --- HELPERS ---

def clear_chat():
    st.session_state.messages = []
    st.session_state.thread_id = None
    st.rerun()

def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=1)
        return r.status_code == 200
    except: return False

def get_threads():
    try:
        r = requests.get(f"{BACKEND_URL}/threads/{USER_ID}", timeout=1)
        return r.json() if r.status_code == 200 else []
    except: return []

def load_thread(thread_id):
    try:
        r = requests.get(f"{BACKEND_URL}/thread/{thread_id}", timeout=2)
        if r.status_code == 200:
            st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in r.json()]
            st.session_state.thread_id = thread_id
    except: pass

    st.rerun()

    st.rerun()

@st.dialog("⚙️ AI Customization", width="large")
def show_settings():
    st.subheader("System Prompt Editor")
    
    st.info("The text below is the complete set of instructions given to the AI. You can modify it or replace it entirely.")
    
    with st.expander("📖 View Original Default Persona (for reference)", expanded=False):
        st.markdown(f"```text\n{CORE_PERSONA}\n```")
    
    new_prompt = st.text_area(
        "Current System Instructions:",
        value=st.session_state.system_prompt,
        height=400,
        placeholder="Enter the full AI persona instructions here...",
        help="This entire text will be sent to the AI as its system instructions."
    )
    if new_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = new_prompt
        st.toast("System prompt saved!")
        
    if st.button("🔄 Reset to Default Persona"):
        st.session_state.system_prompt = CORE_PERSONA
        st.rerun()

# --- SIDEBAR (from app2 design) ---
with st.sidebar:
    st.title("🌱 Reve Advisor")
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        clear_chat()
        
    st.divider()
    st.subheader("⚙️ System Status")
    if check_backend():
        st.success("✅ Backend Connected")
    else:
        st.error("❌ Backend Disconnected")
    
    st.divider()
    st.subheader("📎 Attachments")
    uploaded_files = st.file_uploader(
        "Upload reports (PDF/TXT) or images (JPG/PNG)",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload multiple files for context-aware advice."
    )
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files selected")
        for f in uploaded_files:
            # Preview if image
            if f.type.startswith("image/"):
                st.image(f, caption=f"Preview: {f.name}", use_container_width=True)

    st.divider()
    st.subheader("📚 Chat History")
    threads = get_threads()
    for t in threads:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(f"Chat {t['id']}", key=f"t_{t['id']}", use_container_width=True):
                load_thread(t['id'])
                st.rerun()
        with col2:
            # Simple delete logic
            if st.button("🗑️", key=f"del_{t['id']}"):
                requests.delete(f"{BACKEND_URL}/thread/{t['id']}")
                if st.session_state.thread_id == t['id']:
                    st.session_state.thread_id = None
                    st.session_state.messages = []
                st.rerun()

    st.divider()
    if st.button("🗑️ Clear Current", use_container_width=True):
        clear_chat()

    st.divider()
    if st.button("⚙️ AI Customization", use_container_width=True):
        show_settings()

# --- MAIN UI ---
st.title("Reve AI Agricultural Advisor")
st.markdown("---")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input (static at bottom)
if prompt := st.chat_input("Ask about crops, soil, or pests..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process attachments
    file_paths = []
    image_b64 = None
    if uploaded_files:
        for f in uploaded_files:
            file_bytes = f.read()
            if f.type.startswith("image/"):
                # Use only the first image for visual intent for now, or last
                image_b64 = base64.b64encode(file_bytes).decode('utf-8')
            else:
                f_path = os.path.join(TEMP_DIR, f.name)
                with open(f_path, "wb") as out_f:
                    out_f.write(file_bytes)
                file_paths.append(os.path.abspath(f_path))

    # Call Backend
    with st.chat_message("assistant"):
        with st.spinner("Reve is thinking..."):
            try:
                payload = {
                    "user_id": USER_ID,
                    "thread_id": st.session_state.thread_id,
                    "message": prompt,
                    "file_paths": file_paths, # Corrected to plural
                    "image_b64": image_b64,
                    "system_prompt": st.session_state.system_prompt
                }
                response = requests.post(f"{BACKEND_URL}/chat", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("error"):
                        st.error(f"Error: {data['error']}")
                    else:
                        reply = data["reply"]
                        st.markdown(reply)
                        st.session_state.thread_id = data["thread_id"]
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        st.rerun()
                else:
                    st.error("Backend request failed.")
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("Powered by Reve AI Advisor • Your expert farming companion.")
