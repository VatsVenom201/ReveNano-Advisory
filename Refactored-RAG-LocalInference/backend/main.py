from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import os
import sys

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chat import ChatEngine
from ingest import run_ingestion, get_status
from utils.database import ChatDatabase
from utils.config import DB_PATH
from utils.file_processing import extract_text_from_file

app = FastAPI(title="RAG Microservice API")

# Initialize Chat Engine and Database once at startup
engine = ChatEngine()
db = ChatDatabase(DB_PATH)

@app.on_event("startup")
async def startup_event():
    print("\n--- Startup: Checking for Knowledge Base updates ---")
    try:
        stats = run_ingestion()
        print(f"Startup Sync Complete: Added {stats['processed']}, Deleted {stats['deleted']}\n")
    except Exception as e:
        print(f"WARNING: Startup sync failed: {e}\n")

class QueryRequest(BaseModel):
    query: str
    user_id: str
    thread_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    user_id: str

@app.get("/")
async def root():
    return {"message": "RAG Microservice is running"}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        # 1. Fetch recent history for this USER
        recent_history = db.get_recent_history(request.user_id, limit=50)
        
        # 3. Get response from engine
        answer, sources = engine.response_llm(
            user_query=request.query,
            user_id=request.user_id,
            thread_id=request.thread_id,
            chronological_history_list=recent_history
        )
        
        # 4. Store in SQL (for permanent logging)
        db.add_message(request.user_id, request.thread_id, "user", request.query)
        db.add_message(request.user_id, request.thread_id, "assistant", answer)
        
        # 5. Store in Vector DB (Isolated history store as Q&A pairs)
        engine.add_to_history(request.user_id, request.query, answer)
        
        return QueryResponse(answer=answer, sources=sources, user_id=request.user_id)
        # the response_llm doesn't return user_id, we directly use the id which we gave to it as input.
    except Exception as e:
        print(f"ERROR in chat_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_endpoint():
    try:
        stats = run_ingestion()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(user_id: str, file: UploadFile = File(...)):
    import hashlib
    print(f"\n>>> RECEIVED UPLOAD REQUEST for {file.filename} (User: {user_id}) <<<")
    try:
        from utils.config import USER_FILES_ROOT
        user_folder = os.path.join(USER_FILES_ROOT, user_id)
        os.makedirs(user_folder, exist_ok=True)
        
        content = await file.read()
        incoming_hash = hashlib.md5(content).hexdigest()
        
        # 1. Fast-path duplicate check: file already on disk with same content?
        file_path = os.path.join(user_folder, file.filename)
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                existing_hash = hashlib.md5(f.read()).hexdigest()
            if existing_hash == incoming_hash:
                print(f"DEBUG: '{file.filename}' is identical to the stored copy — skipping disk write and re-index.")
                return {
                    "filename": file.filename,
                    "status": "already_exists",
                    "detail": "This exact file is already uploaded and indexed. No changes made.",
                    "user_id": user_id
                }
        
        # 2. Save Physical File (new upload or updated content)
        with open(file_path, "wb") as f:
            f.write(content)
        print(f"DEBUG: Saved {file.filename} to {file_path}")
        
        # 3. Extract Text
        text = extract_text_from_file(content, file.filename)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file.")
            
        # 4. Index & Persist — engine deduplicates by hash inside ChromaDB
        engine.add_user_document(user_id, file.filename, text, file_size=len(content), raw_content=content)
        
        return {"filename": file.filename, "status": "Successfully uploaded, indexed, and persisted.", "user_id": user_id}
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in upload_file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status_endpoint():
    try:
        status = get_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
