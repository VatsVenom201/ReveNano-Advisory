import streamlit as st
import requests
import base64
import os
from PIL import Image
import io

# Configuration
BASE_URL = "http://127.0.0.1:8000"
USER_ID = 1  # Default User ID
TEMP_DIR = "temp_uploads"

# Ensure temp directory exists
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

st.set_page_config(page_title="Reve Agricultural Advisor", page_icon="🌱", layout="centered")

# Custom CSS for a premium feel
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

st.title("🌱 Reve Agricultural Advisor")
st.markdown("---")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# Connection Health Check
def check_backend_connection():
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Sidebar for file uploads
with st.sidebar:
    st.header("⚙️ System Status")
    if check_backend_connection():
        st.success("✅ Backend Connected")
    else:
        st.error("❌ Backend Disconnected")
        st.info("Make sure the FastAPI server is running on port 8000.")

    st.header("📎 Attachments")
    uploaded_file = st.file_uploader(
        "Upload a document or image", 
        type=["pdf", "docx", "xlsx", "jpg", "jpeg", "png"],
        help="Upload soil reports (PDF/Excel) or crop images for analysis."
    )
    
    if uploaded_file:
        st.success(f"Attached: {uploaded_file.name}")
        # Preview if image
        if uploaded_file.type.startswith("image/"):
            st.image(uploaded_file, caption="Preview", width="stretch")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.thread_id = None
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about your crops, soil, or upload a report..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process Attachments
    file_path = None
    image_b64 = None

    if uploaded_file:
        file_bytes = uploaded_file.read()
        
        # Save to temp location for backend (if it's a doc)
        if not uploaded_file.type.startswith("image/"):
            file_path = os.path.join(TEMP_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            file_path = os.path.abspath(file_path) # Send absolute path to backend
        
        # Encode as base64 if it's an image
        else:
            image_b64 = base64.b64encode(file_bytes).decode('utf-8')

    # Call Backend API
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                payload = {
                    "user_id": USER_ID,
                    "thread_id": st.session_state.thread_id,
                    "message": prompt,
                    "file_path": file_path,
                    "image_b64": image_b64
                }
                
                response = requests.post(f"{BASE_URL}/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                
                if data.get("error"):
                    st.error(data["error"])
                else:
                    reply = data.get("reply", "No response received.")
                    st.markdown(reply)
                    st.session_state.thread_id = data.get("thread_id")
                    st.session_state.messages.append({"role": "assistant", "content": reply})
            
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

# Footer
st.markdown("---")
st.caption("Powered by RAIO AI Advisor • Connect to your farm's future.")
