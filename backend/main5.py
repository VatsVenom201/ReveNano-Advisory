import os
import datetime
import json
import time
from typing import List, Optional
from contextlib import asynccontextmanager
import tiktoken

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv
from openai import OpenAI

import models5 as models
from database5 import engine, get_db

# -------------------------------------------------
# INIT
# -------------------------------------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

models.Base.metadata.create_all(bind=engine)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

MASTER_VECTOR_STORE_NAME = "Unified_Agriculture_RAG"
MASTER_VECTOR_STORE_ID = None

AGRI_ADVISOR_INSTRUCTIONS = (
    """You are an expert agricultural advisory assistant for farmers. Provide practical, scientifically sound, field-applicable guidance using the user query and any retrieved text, documents, or images.

    Response rules:
    - Give a direct answer first.
    - Then provide short pointwise guidance only if needed.
    - Keep responses concise, actionable, and easy for farmers to understand.
    - Use retrieved knowledge when relevant, but synthesize it into advice rather than quoting documents.
    - Reply only in the same language as the user."""
)

FILENAME_CACHE = {}

def get_filename(file_id: str):
    """Retrieve filename from cache or API"""
    if not file_id: return "Unknown"
    if file_id in FILENAME_CACHE:
        return FILENAME_CACHE[file_id]
    try:
        f = client.files.retrieve(file_id)
        FILENAME_CACHE[file_id] = f.filename
        return f.filename
    except Exception:
        return f"File-{file_id[:8]}"

# -------------------------------------------------
# VECTOR STORE HELPERS
# -------------------------------------------------

def get_or_create_master_vector_store():
    stores = client.vector_stores.list()
    for vs in stores.data:
        if getattr(vs, "name", None) == MASTER_VECTOR_STORE_NAME:
            return vs.id
    new_vs = client.vector_stores.create(name=MASTER_VECTOR_STORE_NAME)
    return new_vs.id

def add_uploaded_file_to_store(file_path):
    global MASTER_VECTOR_STORE_ID
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    # Simple check/upload
    existing_files = client.files.list(purpose="user_data")
    file_id = next((f.id for f in existing_files.data if f.filename == file_name and f.bytes == file_size), None)

    if not file_id:
        with open(file_path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="user_data")
            file_id = uploaded.id

    # Attach to MASTER store
    vs_files = client.vector_stores.files.list(vector_store_id=MASTER_VECTOR_STORE_ID)
    if not any(f.id == file_id for f in vs_files.data):
        client.vector_stores.file_batches.create_and_poll(
            vector_store_id=MASTER_VECTOR_STORE_ID,
            file_ids=[file_id]
        )
    return file_id

# -------------------------------------------------
# RETRIEVAL HELPER
# -------------------------------------------------

def retrieve_context(user_query: str):
    """
    Smart retrieval from shared store with source identification
    """
    global MASTER_VECTOR_STORE_ID
    all_chunks = []
    
    # 1. Search Master Store
    try:
        # Search the shared store
        print(f"DEBUG: Searching Shared Store for: '{user_query}'")
        results = client.vector_stores.search(
            vector_store_id=MASTER_VECTOR_STORE_ID,
            query=user_query
        )
        if hasattr(results, "data"):
            print(f"DEBUG: Found {len(results.data)} raw chunks. Processing top 5...")
            for i, r in enumerate(results.data[:5]):
                file_id = getattr(r, "file_id", None)
                fname = get_filename(file_id)
                chunk_text = r.content[0].text
                
                # Terminal Debug
                print(f"--- CHUNK {i+1} [FROM: {fname}] ---")
                print(f"TEXT: {chunk_text[:200]}...")
                
                all_chunks.append(f"[Source: {fname}]\n{chunk_text}")
    except Exception as e:
        print(f"Retrieval error: {e}")
            
    print(f"DEBUG: Retrieval complete. Total accepted chunks: {len(all_chunks)}")
    return "\n\n".join(all_chunks)

# -------------------------------------------------
# LIFESPAN
# -------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MASTER_VECTOR_STORE_ID
    MASTER_VECTOR_STORE_ID = get_or_create_master_vector_store()
    print(f"Server started with Shared Vector Store: {MASTER_VECTOR_STORE_ID}")
    yield

app = FastAPI(title="AI Chat Microservice V5 - Simpler Shared Store", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# SCHEMAS
# -------------------------------------------------

class MessageSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    role: str
    content: str
    timestamp: datetime.datetime

class ChatThreadSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    user_id: int
    created_at: datetime.datetime

class ChatInput(BaseModel):
    user_id: int
    thread_id: Optional[int] = None
    message: str
    file_paths: List[str] = []
    image_b64: Optional[str] = None
    system_prompt: Optional[str] = None

class ChatResponse(BaseModel):
    reply: Optional[str] = None
    thread_id: int
    user_id: int
    error: Optional[str] = None

# -------------------------------------------------
# DB HELPERS
# -------------------------------------------------

def get_or_create_user(db: Session, user_id: int):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        user = models.User(id=user_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

# -------------------------------------------------
# ENDPOINTS
# -------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(chat_input: ChatInput, db: Session = Depends(get_db)):
    user = get_or_create_user(db, chat_input.user_id)

    if chat_input.thread_id:
        thread = db.query(models.ChatThread).filter(
            models.ChatThread.id == chat_input.thread_id,
            models.ChatThread.user_id == user.id
        ).first()
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
    else:
        thread = models.ChatThread(user_id=user.id)
        db.add(thread)
        db.commit()
        db.refresh(thread)

    # Retrieval
    if chat_input.file_paths:
        for f_path in chat_input.file_paths:
            if os.path.exists(f_path):
                print(f"DEBUG: Adding {f_path} to Shared store.")
                add_uploaded_file_to_store(f_path)

    context = retrieve_context(chat_input.message)

    # History (Token-Based Limit: 1500 tokens)
    # Replaces the old count-based history:
    # history_messages = db.query(models.Message).filter(
    #     models.Message.thread_id == thread.id
    # ).order_by(models.Message.timestamp.desc()).limit(4).all()    # 4 is the number of messages to be included in the history
    # history_messages.reverse()                                    # 2 ai, 2 human
    
    # NEW Token Logic:
    all_messages = db.query(models.Message).filter(
        models.Message.thread_id == thread.id
    ).order_by(models.Message.timestamp.desc()).all()
    
    encoding = tiktoken.get_encoding("o200k_base")
    history_limit = 1500
    current_tokens = 0
    limited_history = []
    
    for msg in all_messages:
        role_label = "Farmer" if msg.role == "user" else "Advisor"
        msg_text = f"{role_label}: {msg.content}\n"
        msg_tokens = len(encoding.encode(msg_text))
        
        if current_tokens + msg_tokens <= history_limit:
            current_tokens += msg_tokens
            limited_history.insert(0, msg_text) # Keep chronological (oldest to newest)
        else:
            print(f"DEBUG: History token limit ({history_limit}) reached. Omitted older messages.")
            break # Reached token limit
            
    history_str = "".join(limited_history)
    print(f"DEBUG: Included {len(limited_history)} messages in history ({current_tokens} tokens)")

    # Payload
    combined_prompt = f"""
History:
{history_str}

Retrieved Knowledge:
{context}

Query:
{chat_input.message}
"""
    
    instructions = chat_input.system_prompt or AGRI_ADVISOR_INSTRUCTIONS

    input_content = [{"type": "input_text", "text": combined_prompt}]
    if chat_input.image_b64:
        img = chat_input.image_b64
        if not img.startswith("data:image"):
            img = f"data:image/jpeg;base64,{img}"
        input_content.append({"type": "input_image", "image_url": img})

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=instructions,
            input=[{"role": "user", "content": input_content}]
        )
        bot_reply = response.output_text
        response_id = response.id
        
        # Save
        db.add(models.Message(thread_id=thread.id, role="user", content=chat_input.message))
        db.add(models.Message(thread_id=thread.id, role="assistant", content=bot_reply, response_id=response_id))
        db.commit()

        return {"reply": bot_reply, "thread_id": thread.id, "user_id": user.id}
    except Exception as e:
        return {"thread_id": thread.id, "user_id": user.id, "error": str(e)}

@app.get("/threads/{user_id}", response_model=List[ChatThreadSchema])
def get_threads(user_id: int, db: Session = Depends(get_db)):
    return db.query(models.ChatThread).join(models.Message).filter(
        models.ChatThread.user_id == user_id
    ).distinct().order_by(models.ChatThread.created_at.desc()).all()

@app.get("/thread/{thread_id}", response_model=List[MessageSchema])
def get_messages(thread_id: int, db: Session = Depends(get_db)):
    thread = db.query(models.ChatThread).filter(models.ChatThread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread.messages

@app.delete("/thread/{thread_id}")
def delete_thread(thread_id: int, db: Session = Depends(get_db)):
    thread = db.query(models.ChatThread).filter(models.ChatThread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    db.delete(thread)
    db.commit()
    return {"message": "Thread deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
