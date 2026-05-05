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

import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# INIT
# -------------------------------------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

models.Base.metadata.create_all(bind=engine)

# -------------------------------------------------
# LOCAL RAG CONFIG (ChromaDB + Sentence Transformers)
# -------------------------------------------------

MODEL_DIR = os.path.join(os.getcwd(), "embedding_models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("INFO: Loading local embedding model (all-MiniLM-L6-v2)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: Using device: {device}")

# Load model locally
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device, cache_folder=MODEL_DIR)

# Init ChromaDB
CHROMA_PATH = os.path.join(os.getcwd(), "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_user_history_collection(user_id: int):
    """
    Get or create a dedicated collection for a specific user to ensure physical isolation.
    """
    collection_name = f"user_history_{user_id}"
    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

def index_message_locally(user_id: int, thread_id: int, content: str, role: str):
    """
    Store message in ChromaDB for semantic retrieval.
    Includes chunking for longer messages and thread tracking.
    """
    try:
        # Simple Chunking: 500 chars with 50 overlap
        chunk_size = 500
        overlap = 50
        #overlap = 100
        chunks = []
        
        if len(content) <= chunk_size:
            chunks = [content]
        else:
            # Basic sliding window chunking
            start = 0
            while start < len(content):
                end = start + chunk_size
                chunks.append(content[start:end])
                start += (chunk_size - overlap)
        
        for i, chunk in enumerate(chunks):
            # Manually generate embedding to ensure we use our GPU model and D: drive cache
            chunk_emb = embed_model.encode([chunk]).tolist()
            
            # Get the user's specific collection
            user_col = get_user_history_collection(user_id)
            
            msg_id = f"msg_{user_id}_{int(time.time() * 1000)}_{i}"
            user_col.add(
                embeddings=chunk_emb,
                documents=[chunk],
                metadatas=[{
                    "user_id": user_id, 
                    "thread_id": thread_id,
                    "role": role, 
                    "timestamp": str(datetime.datetime.now(datetime.timezone.utc)),
                    "original_msg_id": f"{user_id}_{int(time.time())}"
                }],
                ids=[msg_id]   # msg_id is manuyally generated, can be done response_id from openai
            )
        print(f"DEBUG: Indexed {len(chunks)} local chunks for Thread {thread_id} (User {user_id}).")
    except Exception as e:
        print(f"DEBUG: Local indexing error: {e}")

def retrieve_local_history(query: str, user_id: int, threshold: float = 0.35):
    """
    Semantic search over past messages
    """
    try:
        query_emb = embed_model.encode([query]).tolist()
        
        # Get the user's specific collection
        user_col = get_user_history_collection(user_id)
        
        # Max 5 chunks as requested
        # We NO LONGER need the 'where' filter because this collection ONLY has this user's data
        results = user_col.query(
            query_embeddings=query_emb,
            n_results=5
        )
        
        relevant_chunks = []
        if results['documents'] and len(results['documents'][0]) > 0:
            print(f"DEBUG: Local Search found {len(results['documents'][0])} potential historical chunks.")
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                similarity = 1 - distance
                
                if similarity >= threshold:
                    meta = results['metadatas'][0][i]
                    role = meta.get('role', 'unknown')
                    tid = meta.get('thread_id', 'unknown')
                    
                    print(f"   - [LOCAL MATCH] Score: {similarity:.3f} | Thread: {tid} | {role}: {doc[:200]}...")
                    relevant_chunks.append(f"[Local History - Thread {tid}] ({role}): {doc}")
                    
        return relevant_chunks
    except Exception as e:
        print(f"DEBUG: Local RAG error: {e}")
        return []

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

def infer_visual_intent(query: str, image_b64: Optional[str]):
    """
    Decide if the image is actually relevant to the current question.
    """
    if not image_b64:
        return False
        
    q = query.lower()
    visual_keywords = [
        "image", "photo", "picture", "this", "see", "look", 
        "show", "leaf", "plant", "spot", "disease", "what is this",
        "diagnosis", "view", "crop", "report"
    ]
    
    # If the user mentions visual words or asks a diagnostic question
    return any(k in q for k in visual_keywords)

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
        batch = client.vector_stores.file_batches.create_and_poll(
            vector_store_id=MASTER_VECTOR_STORE_ID,
            file_ids=[file_id]
        )

        print("DEBUG: Batch processing completed.")

        # EXTRA SAFETY CHECK
        max_wait = 30
        # reduce to 15
        start = time.time()

        while time.time() - start < max_wait:

            vs_file = client.vector_stores.files.retrieve(
                vector_store_id=MASTER_VECTOR_STORE_ID,
                file_id=file_id
            )

            status = vs_file.status
            print(f"DEBUG: File status = {status}")

            if status == "completed":
                print("DEBUG: File fully indexed and searchable.")
                break

            elif status == "failed":
                raise Exception("Vector store indexing failed.")

            time.sleep(2)
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
            for i, r in enumerate(results.data[:5]):   # top5 from master store
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

app = FastAPI(title="AI Chat Microservice V2 - Local RAG & Shared VS", lifespan=lifespan)

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

    # NEW: Local History Retrieval
    print(f"DEBUG: Starting local history search for User {user.id}...")
    local_history = retrieve_local_history(chat_input.message, user.id, threshold=0.35)
    if local_history:
        context += "\n\n[Relevant Past Context]\n" + "\n".join(local_history)
        print(f"DEBUG: Found {len(local_history)} semantic matches from local history.")

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
    # 1500 tokens limit for history context
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
MEM:
{history_str}

CTX:
{context}

Q:
{chat_input.message}
"""
    
    instructions = chat_input.system_prompt or AGRI_ADVISOR_INSTRUCTIONS

    input_content = [{"type": "input_text", "text": combined_prompt}]
    if chat_input.image_b64:
        # Smart Check: Only send image if the query refers to it
        if infer_visual_intent(chat_input.message, chat_input.image_b64):
            print("DEBUG: Visual intent detected. Sending image to LLM.")
            img = chat_input.image_b64
            if not img.startswith("data:image"):
                img = f"data:image/jpeg;base64,{img}"
            input_content.append({"type": "input_image", "image_url": img})
        else:
            print("DEBUG: Image provided but no visual intent detected in query. Skipping image payload.")

    try:
        print(f"DEBUG: Calling OpenAI Responses API (Model: gpt-4.1-mini)...")
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=instructions,
            temperature=0.8,
            max_output_tokens=800,
            store=False, # Disable server side storage if API supports it
            tools=[],
            input=[{"role": "user", "content": input_content}]
        )
        print("DEBUG: OpenAI API call successful.")
        bot_reply = response.output_text
        response_id = response.id
        
        # Save
        db.add(models.Message(thread_id=thread.id, role="user", content=chat_input.message))
        db.add(models.Message(thread_id=thread.id, role="assistant", content=bot_reply, response_id=response_id))
        db.commit()

        # NEW: Index locally with thread awareness
        # SAVE in local vectordb
        index_message_locally(user.id, thread.id, chat_input.message, "user")
        index_message_locally(user.id, thread.id, bot_reply, "assistant")

        print(f"DEBUG: Final API Usage: {response.usage}")

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
