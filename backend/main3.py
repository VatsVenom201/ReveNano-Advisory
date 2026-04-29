import os
import datetime
import json
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv
from openai import OpenAI

import models
from database import engine, get_db

# -------------------------------------------------
# INIT
# -------------------------------------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Chat Microservice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# UNIFIED VECTOR STORE CONFIG
# -------------------------------------------------

MASTER_VECTOR_STORE_NAME = "Unified_Agriculture_RAG"
MASTER_VECTOR_STORE_ID = None

AGRI_ADVISOR_INSTRUCTIONS = (
    """You are an expert agricultural advisory assistant for farmers. Provide practical, scientifically sound, field-applicable guidance using the user query and any retrieved text, documents, or images.

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
)

SEED_FILES = [
    {
        "id": "file-5REcXM6ddmgkLaLCVC5NzD",
        "attributes": {
            "content_description": (
                "Agriculture knowledgebase covering soil science, crops, nutrients, diseases, treatments"
            ),
            "topic_area": "agriculture",
            "document_type": "knowledgebase",
            "target_audience": "farmers",
            "data_scope": "general_reference"
        }
    }
]

SOIL_REPORT_ATTRIBUTES = {
    "content_description": (
        "User uploaded soil report containing pH nutrient analysis"
    ),
    "topic_area": "soil_reports",
    "document_type": "user_report",
    "target_audience": "farmers",
    "data_scope": "personal_data"
}


# -------------------------------------------------
# VECTOR STORE HELPERS
# -------------------------------------------------

def get_or_create_master_vector_store():
    stores = client.vector_stores.list()

    for vs in stores.data:
        if getattr(vs, "name", None) == MASTER_VECTOR_STORE_NAME:
            return vs.id

    new_vs = client.vector_stores.create(
        name=MASTER_VECTOR_STORE_NAME
    )
    print(f"Created vector store: {new_vs.id}")
    return new_vs.id


def seed_vector_store(vs_id):
    """
    Ensure seed files exist in vector store
    and attach metadata attributes.
    """
    try:
        existing = client.vector_stores.files.list(
            vector_store_id=vs_id
        )

        existing_ids = {f.id for f in existing.data}

        for item in SEED_FILES:
            file_id = item["id"]

            if file_id not in existing_ids:
                client.vector_stores.files.create(
                    vector_store_id=vs_id,
                    file_id=file_id
                )
                print(f"Added seed file {file_id}")

            # attach/update attributes
            # SDK versions may vary slightly here.
            try:
                client.vector_stores.files.update(
                    vector_store_id=vs_id,
                    file_id=file_id,
                    attributes=item["attributes"]
                )
                print(f"Updated attributes for {file_id}")
            except Exception as e:
                print(
                    f"Attribute update warning for {file_id}: {e}"
                )

    except Exception as e:
        print("Vector store seed error:", e)


def add_uploaded_file_to_store(file_path):
    """
    Deduplicate against OpenAI files by name+size,
    then ensure in vector store,
    inherit soil-report metadata.
    """
    global MASTER_VECTOR_STORE_ID

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    file_id = None

    # Check already on OpenAI servers
    existing_files = client.files.list(
        purpose="user_data"
    )

    for f in existing_files.data:
        if (
            f.filename == file_name
            and f.bytes == file_size
        ):
            file_id = f.id
            print(f"Reusing file {file_id}")
            break

    # upload if absent
    if not file_id:
        with open(file_path, "rb") as f:
            uploaded = client.files.create(
                file=f,
                purpose="user_data"
            )
            file_id = uploaded.id
            print(f"Uploaded new file {file_id}")

    # ensure in vector store
    vs_files = client.vector_stores.files.list(
        vector_store_id=MASTER_VECTOR_STORE_ID
    )

    existing_ids = {f.id for f in vs_files.data}

    if file_id not in existing_ids:
        print(f"Adding {file_id} to vector store using file batch...")
        
        # Use the official OpenAI pooling helper for reliable ingestion
        client.vector_stores.file_batches.create_and_poll(
            vector_store_id=MASTER_VECTOR_STORE_ID,
            file_ids=[file_id]
        )
        
        try:
            # Update attributes for manual filtering search
            # We must include the filename in the description to match what the retriever looks for
            file_attributes = SOIL_REPORT_ATTRIBUTES.copy()
            file_attributes["content_description"] = f"Soil report: {file_name}"
            
            client.vector_stores.files.update(
                vector_store_id=MASTER_VECTOR_STORE_ID,
                file_id=file_id,
                attributes=file_attributes
            )
            print(f"Successfully updated attributes for {file_id} (Source: {file_name})")
            
            # PROACTIVE POLLING: Wait until the file is actually searchable with its new metadata
            print(f"DEBUG: Waiting for {file_id} ({file_name}) to become searchable...")
            search_ready = False
            for i in range(15): # Max 30 seconds
                try:
                    test_results = client.vector_stores.search(
                        vector_store_id=MASTER_VECTOR_STORE_ID,
                        query=file_name, # Use filename as query to ensure high relevance
                        filters={
                            "key": "content_description",
                            "type": "eq",
                            "value": f"Soil report: {file_name}"
                        }
                    )
                    if hasattr(test_results, "data") and len(test_results.data) > 0:
                        print(f"DEBUG: File {file_id} is now searchable! (Attempt {i+1})")
                        search_ready = True
                        break
                except Exception as ex:
                    print(f"DEBUG: Search poll error: {ex}")
                
                print(f"DEBUG: Still waiting for indexing... (Attempt {i+1}/15)")
                time.sleep(2)
            
            if not search_ready:
                print(f"WARNING: File {file_id} did not become searchable within timeout. Proceeding anyway.")

            refresh_file_map()
            print(f"Ingestion complete for {file_id}")
        except Exception as e:
            print(f"Attribute update warning: {e}")

    return file_id

    return file_id


# -------------------------------------------------
# RETRIEVAL HELPERS
# -------------------------------------------------

FILE_ID_MAP = {} # Cache for file_id -> filename

def refresh_file_map():
    global FILE_ID_MAP
    try:
        # Remove purpose="user_data" to include ALL files (seed files are purpose="assistants")
        files = client.files.list()
        for f in files.data:
            FILE_ID_MAP[f.id] = f.filename
    except Exception as e:
        print(f"File map refresh error: {e}")

def infer_metadata_filter(query: str):
    global FILE_ID_MAP
    
    def normalize(text: str):
        return text.lower().replace("_", " ").replace("-", " ").strip()

    q_norm = normalize(query)

    # Step 1: Detect all mentioned filenames
    # We look through all cached filenames
    target_file_id = None
    for fid, fname in FILE_ID_MAP.items():
        name_only = normalize(fname.split(".")[0])
        
        # If user mentions the specific filename or its stem
        if name_only in q_norm and len(name_only) > 3:
            target_file_id = fid
            print(f"DEBUG: Target file detected in query: {fname}")
            break

    # Step 2: Return file-specific filter if detected
    # if target_file_id:
    #     return {
    #         "key": "file_id",  # Not all SDKs support direct file_id filtering in search
    #         "type": "eq",
    #         "value": target_file_id
    #     }
    # Currently, we use the description field for smartest results if filename is detected
    if target_file_id:
        detected_name = FILE_ID_MAP[target_file_id]
        return {
            "key": "content_description",
            "type": "eq",
            "value": f"Soil report: {detected_name}"
        }

    # Step 3: Fallback to general data_scope filters
    personal_keywords = [
        "my soil report",
        "my report",
        "my soil",
        "soil test",
        "ph value",
        "my nutrient",
        "this soil",
        "report"
    ]

    if any(k in q_norm for k in personal_keywords):
        return {
            "key": "data_scope",
            "type": "eq",
            "value": "personal_data"
        }

    return {
        "key": "topic_area",
        "type": "eq",
        "value": "agriculture"
    }


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
        "diagnosis", "view", "crop"
    ]
    
    # If the user mentions visual words or asks a diagnostic question
    return any(k in q for k in visual_keywords)


def retrieve_context(user_query: str):
    """
    Metadata filtered retrieval with multi-source tracking and comparison support.
    """
    global MASTER_VECTOR_STORE_ID, FILE_ID_MAP

    def normalize(text: str):
        return text.lower().replace("_", " ").replace("-", " ").strip()

    q_norm = normalize(user_query)
    
    # Step 1: Detect all mentioned filenames
    if not FILE_ID_MAP:
        refresh_file_map()

    detected_files = [] # List of (fid, fname)
    for fid, fname in FILE_ID_MAP.items():
        name_only = normalize(fname.split(".")[0])
        if name_only in q_norm and len(name_only) > 3:
            detected_files.append((fid, fname))

    # Step 2: Determine Search Filters
    # We will perform searches for each detected file, or one general search
    search_tasks = []
    
    if len(detected_files) > 0:
        names = [f[1] for f in detected_files]
        print(f"DEBUG: Multi-file detection active. Targeted files: {names}")
        for fid, fname in detected_files:
            search_tasks.append({
                "label": f"File-Specific Search: {fname}",
                "filter": {
                    "key": "content_description",
                    "type": "eq",
                    "value": f"Soil report: {fname}"
                },
                "limit": 10
            })
    else:
        # Fallback to general filters if no specific files named
        filt = infer_metadata_filter(user_query)
        print(f"DEBUG: No specific files named. Falling back to General Search (Filter: {filt})")
        search_tasks.append({
            "label": "General Search (Personal/Agri)",
            "filter": filt,
            "limit": 10
        })

    # Step 3: Execute searches and combine
    all_chunks = []
    
    for task in search_tasks:
        try:
            print(f"DEBUG: Executing {task['label']} (Initial Retrieval Limit: {task['limit']})...")
            results = client.vector_stores.search(
                vector_store_id=MASTER_VECTOR_STORE_ID,
                query=user_query,
                filters=task["filter"]
            )
            
            if hasattr(results, "data"):
                raw_chunks = results.data
                print(f"DEBUG: Found {len(raw_chunks)} raw chunks for {task['label']}")
                
                # --- RERANKING & FILTERING ---
                # We skip the thresholding step to avoid missing potentially relevant data 
                # that might have a lower score than "garbage" chunks.
                # Results are already pre-sorted by score from the vector store.
                reranked_chunks = list(raw_chunks)
                
                # Sort descending by score (explicit sort)
                reranked_chunks.sort(key=lambda x: getattr(x, "score", 0.0), reverse=True)
                
                # 3. Take Top 5 for this file/task
                final_file_chunks = reranked_chunks[:5]
                print(f"DEBUG: Retrieval complete. {len(final_file_chunks)} chunks accepted for {task['label']}")
                
                for i, r in enumerate(final_file_chunks):
                    file_id = getattr(r, "file_id", "unknown")
                    source_name = FILE_ID_MAP.get(file_id, "Unknown Source")
                    score = getattr(r, "score", 0.0)
                    
                    if hasattr(r, "content"):
                        chunk_text = r.content[0].text
                        # Show preview in terminal
                        preview = " ".join(chunk_text.split()[:50])
                        print(f"--- Chunk {i+1} [Source: {source_name}] [Score: {score:.4f}] ---\n{preview}...\n")
                        
                        all_chunks.append(f"[Source: {source_name}]\n{chunk_text}")
        except Exception as e:
            print(f"Search error for {task['label']}: {e}")

    context = "\n\n".join(all_chunks)
    print(f"DEBUG: Total combined retrieval size: {len(all_chunks)} chunks.")
    return context


# -------------------------------------------------
# STARTUP
# -------------------------------------------------

@app.on_event("startup")
def startup_event():
    global MASTER_VECTOR_STORE_ID

    MASTER_VECTOR_STORE_ID = (
        get_or_create_master_vector_store()
    )

    seed_vector_store(
        MASTER_VECTOR_STORE_ID
    )
    
    refresh_file_map() # Pre-load filenames


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
    file_paths: List[str] = [] # Changed from file_path to file_paths
    image_b64: Optional[str] = None
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    reply: Optional[str] = None
    thread_id: int
    user_id: int
    error: Optional[str] = None


# -------------------------------------------------
# DB HELPER
# -------------------------------------------------

def get_or_create_user(db: Session, user_id: int):
    user = (
        db.query(models.User)
        .filter(models.User.id == user_id)
        .first()
    )

    if not user:
        user = models.User(id=user_id)
        db.add(user)
        db.commit()
        db.refresh(user)

    return user


# -------------------------------------------------
# HEALTH
# -------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


# -------------------------------------------------
# CHAT ENDPOINT
# -------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(
    chat_input: ChatInput,
    db: Session = Depends(get_db)
):

    user = get_or_create_user(
        db,
        chat_input.user_id
    )

    if chat_input.thread_id:
        thread = (
            db.query(models.ChatThread)
            .filter(
                models.ChatThread.id == chat_input.thread_id,
                models.ChatThread.user_id == user.id
            )
            .first()
        )

        if not thread:
            raise HTTPException(
                status_code=404,
                detail="Thread not found"
            )
    else:
        thread = models.ChatThread(
            user_id=user.id
        )
        db.add(thread)
        db.commit()
        db.refresh(thread)


    last_assistant = (
        db.query(models.Message)
        .filter(
            models.Message.thread_id == thread.id,
            models.Message.role == "assistant"
        )
        .order_by(
            models.Message.timestamp.desc()
        )
        .first()
    )

    previous_id = (
        last_assistant.response_id
        if last_assistant else None
    )


    # -------------------------
    # Construct Payload
    # -------------------------
    
    # Base instructions are permanent. User prompt is appended if provided.
    # 4. Consolidate Instructions
    # If a system_prompt is provided from the UI, it replaces the default AGRI_ADVISOR_INSTRUCTIONS entirely.
    instructions = chat_input.system_prompt or AGRI_ADVISOR_INSTRUCTIONS


    # -------------------------
    # Handle multiple file uploads
    # -------------------------
    if chat_input.file_paths:
        for f_path in chat_input.file_paths:
            if os.path.exists(f_path):
                print(f"DEBUG: Processing upload: {f_path}")
                add_uploaded_file_to_store(f_path)


    # -------------------------
    # metadata-filtered retrieval
    # -------------------------

    retrieved_context = retrieve_context(
        chat_input.message
    )


    # -------------------------
    # Construct Manual History String (Last 3 queries = ~6 messages)
    # -------------------------
    
    history_messages = (
        db.query(models.Message)
        .filter(models.Message.thread_id == thread.id)
        .order_by(models.Message.timestamp.desc())
        .limit(6) # Last 6 messages (3 users, 3 assistants)
        .all()
    )
    history_messages.reverse()

    history_str = ""
    for msg in history_messages:
        role_label = "Farmer" if msg.role == "user" else "Advisor"
        history_str += f"{role_label}: {msg.content}\n"


    # -------------------------
    # Current interaction
    # -------------------------

    current_user_content = []

    combined_prompt = f"""
Below is the recent conversation history for context:
{history_str}

Retrieved Agriculture Knowledge/Reports:
{retrieved_context}

New User Query:
{chat_input.message}
"""

    current_user_content.append(
        {
            "type":"input_text",
            "text": combined_prompt
        }
    )


    if chat_input.image_b64:
        # Smart Check: Only send the image if the query actually refers to it
        if infer_visual_intent(chat_input.message, chat_input.image_b64):
            print("DEBUG: Visual intent detected. Sending image to LLM.")
            img = chat_input.image_b64
            if not img.startswith("data:image"):
                img = f"data:image/jpeg;base64,{img}"
            current_user_content.append(
                {"type":"input_image", "image_url":img}
            )
        else:
            print("DEBUG: Image provided but no visual intent detected in query. Skipping image payload.")

    # Final Payload for Responses API
    openai_payload_input = [
        {
            "role": "user",
            "content": current_user_content
        }
    ]


    bot_reply = None
    response_id = None
    error_msg = None

########################################################################################################
#                         MODEL IMPLEMENTATION
########################################################################################################
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=instructions,
            # previous_response_id=previous_id, # REMOVED for token efficiency
            temperature=0.8,
            max_output_tokens=800,
            # store=False, # Disable server side storage if API supports it
            tools=[],
            input=openai_payload_input
        )

        bot_reply = response.output_text
        response_id = response.id

        print(f"Response ID: {response.id}")
        print(f"Token Usage: \n{response.usage}\n")
        print(f"Response: \n{response.output}\n")
        
        # Print file search results if present in the response
        if hasattr(response, "file_search_call"):
            print(f"DEBUG: OpenAI File Search Tool Results: {json.dumps(response.file_search_call, indent=2)}")
        else:
            print("DEBUG: No direct file_search_call tool results in response object.")

    except Exception as e:
        print("OpenAI Error:",e)
        error_msg = str(e)


    # -------------------------
    # Save messages
    # -------------------------

    if not error_msg:
        db.add(
            models.Message(
                thread_id=thread.id,
                role="user",
                content=chat_input.message
            )
        )

        db.add(
            models.Message(
                thread_id=thread.id,
                role="assistant",
                content=bot_reply,
                response_id=response_id
            )
        )

        db.commit()


    return {
        "reply": bot_reply,
        "thread_id": thread.id,
        "user_id": user.id,
        "error": error_msg
    }


# -------------------------------------------------
# THREAD ROUTES
# -------------------------------------------------

@app.get(
    "/threads/{user_id}",
    response_model=List[ChatThreadSchema]
)
def get_threads(
    user_id:int,
    db:Session=Depends(get_db)
):
    return (
        db.query(models.ChatThread)
        .join(models.Message) # This ensures only threads with messages are returned
        .filter(
            models.ChatThread.user_id == user_id
        )
        .distinct()
        .order_by(
            models.ChatThread.created_at.desc()
        )
        .all()
    )


@app.get(
    "/thread/{thread_id}",
    response_model=List[MessageSchema]
)
def get_messages(
    thread_id:int,
    db:Session=Depends(get_db)
):

    thread=(
        db.query(models.ChatThread)
        .filter(
            models.ChatThread.id==thread_id
        )
        .first()
    )

    if not thread:
        raise HTTPException(
            status_code=404,
            detail="Thread not found"
        )

    return thread.messages


@app.delete("/thread/{thread_id}")
def delete_thread(
    thread_id:int,
    db:Session=Depends(get_db)
):

    thread=(
        db.query(models.ChatThread)
        .filter(
            models.ChatThread.id==thread_id
        )
        .first()
    )

    if not thread:
        raise HTTPException(
            status_code=404,
            detail="Thread not found"
        )

    db.delete(thread)
    db.commit()

    return {
        "message":"Thread deleted"
    }


# -------------------------------------------------
# RUN
# -------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
