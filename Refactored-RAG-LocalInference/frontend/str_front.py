import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import requests
import os

# Backend API URL
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Microservice Frontend", layout="wide")

st.title('🚀 Reve Nano Science Chatbot')
st.markdown("---")


# Session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [AIMessage(content='Connected to RAG Backend. How can I help?')]

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message('assistant'):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message('user'):
            st.write(message.content)

# User Management
if 'user_id' not in st.session_state:
    st.session_state.user_id = "vats_user"

# Sidebar with Controls
with st.sidebar:
    st.header("👤 User Profile")
    st.session_state.user_id = st.text_input("Enter User ID", value=st.session_state.user_id)
    st.info(f"Currently acting as: `{st.session_state.user_id}`")
    
    st.markdown("---")
    st.header("Admin Controls")
    if st.button("Trigger Ingestion"):
        with st.spinner("Processing new files in background..."):
            try:
                response = requests.post(f"{BACKEND_URL}/ingest")
                if response.status_code == 200:
                    st.success("Ingestion complete!")
                else:
                    st.error(f"Ingestion failed: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
                
    if st.button("Clear Chat"):
        st.session_state.chat_history = [AIMessage(content='Chat cleared. How can I help?')]
        st.rerun()

    st.markdown("---")
    st.header("📂 Personal Knowledge")
    uploaded_file = st.file_uploader("Upload PDF, Word, or Excel", type=["pdf", "docx", "xlsx", "xls"])
    
    if uploaded_file:
        st.info(f"📄 **{uploaded_file.name}** selected. It will be indexed when you send your next query.")

# User input
user_query = st.chat_input("Ask a question...")

if user_query:
    # 1. Check for uploaded files and index them first
    if uploaded_file:
        with st.status(f"Indexing {uploaded_file.name}...", expanded=False) as status:
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                params = {"user_id": st.session_state.user_id}
                response = requests.post(f"{BACKEND_URL}/upload", params=params, files=files)
                if response.status_code == 200:
                    status.update(label=f"✅ {uploaded_file.name} indexed", state="complete")
                else:
                    st.error(f"❌ Failed to index {uploaded_file.name}: {response.text}")
                    status.update(label="Index Failed", state="error")
                    st.stop()
            except Exception as e:
                st.error(f"Connection error during indexing: {e}")
                status.update(label="Connection Error", state="error")
                st.stop()

    with st.chat_message('user'):
        st.markdown(user_query)
    
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message('assistant'):
        with st.spinner("Assistant is thinking..."):
            try:
                payload = {
                    "query": user_query,
                    "user_id": st.session_state.user_id,
                    "thread_id": "thread_1"
                }
                
                response = requests.post(f"{BACKEND_URL}/chat", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    st.write(answer)
                    
                    if sources:
                        st.markdown("---")
                        st.markdown("### 🔍 Sources Used")
                        for i, source in enumerate(sources):
                            src_name = os.path.basename(source['metadata'].get('source', 'Unknown'))
                            with st.expander(f"Source {i + 1}: {src_name}"):
                                st.write(source["content"])
                    
                    st.session_state.chat_history.append(AIMessage(content=answer))
                else:
                    st.error(f"Error from backend: {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
