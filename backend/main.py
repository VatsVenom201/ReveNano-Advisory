import os
import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict
from openai import OpenAI
from dotenv import load_dotenv

import models
from database import engine, get_db

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Chat Microservice")

# Enable CORS for internal requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
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

class ChatResponse(BaseModel):
    reply: Optional[str] = None
    thread_id: int
    user_id: int
    error: Optional[str] = None

# Helpers
def get_or_create_user(db: Session, user_id: int):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        user = models.User(id=user_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

# Endpoints
@app.post("/chat", response_model=ChatResponse)
def chat(chat_input: ChatInput, db: Session = Depends(get_db)):
    # 1. Ensure User exists
    user = get_or_create_user(db, chat_input.user_id)
    
    # 2. Get or Create Thread
    if chat_input.thread_id:
        thread = db.query(models.ChatThread).filter(
            models.ChatThread.id == chat_input.thread_id,
            models.ChatThread.user_id == user.id
        ).first()
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found or ownership mismatch")
    else:
        thread = models.ChatThread(user_id=user.id)
        db.add(thread)
        db.commit()
        db.refresh(thread)

    # 3. Get last assistant message for continuity
    last_assistant_msg = db.query(models.Message).filter(
        models.Message.thread_id == thread.id,
        models.Message.role == "assistant"
    ).order_by(models.Message.timestamp.desc()).first()
    
    previous_id = last_assistant_msg.response_id if last_assistant_msg else None
    
    # 4. Call OpenAI Responses API
    instructions = (
            "You are a professional agricultural advisor. Provide clear, actionable, and scientific advice to farmers. "
    "Understand the user's query and respond with practical, field-applicable guidance. "
    "Keep explanations simple but accurate.\n\n"

    "1. UNDERSTANDING THE PROBLEM\n"
    "Explain the user’s issue clearly.\n\n"

    "2. POSSIBLE CAUSES / INSIGHTS\n"
    "List likely scientific or environmental causes.\n\n"

    "3. RECOMMENDED ACTIONS\n"
    "Provide clear step-by-step solutions.\n\n"

    "4. RISKS / PRECAUTIONS\n"
    "Mention risks and what to avoid.\n\n"

    "5. NEXT STEPS / MONITORING\n"
    "Explain how to track progress or improvements.\n\n"

    "6. CONFIDENCE LEVEL\n"
    "State confidence (Low/Medium/High) with a short reason."
    
    "Do NOT repeat any section. Avoid duplicate explanations. Keep response concise and structured."
    "Only reply in the language that user used to ask the query. Keep language simple"

    )

    error_msg = None
    bot_reply = None
    response_id = None

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=instructions,
            previous_response_id=previous_id,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": chat_input.message}
                    ]
                }
            ]
        )
        
        try:
            bot_reply = response.output_text
            usage = response.usage
            print(f"Usage for this query: \n{usage}\n")
        except (AttributeError, IndexError):
            bot_reply = "No response generated"
            
        response_id = response.id
        
    except Exception as e:
        error_name = type(e).__name__
        if "insufficient_quota" in str(e).lower():
            error_msg = "Error 429: API Quota Exceeded. Please check your billing/balance."
        elif "rate_limit" in str(e).lower():
            error_msg = "Error 429: Rate Limit Reached. Please wait a moment."
        elif "authentication" in str(e).lower():
            error_msg = "Error 401: Invalid API Key or Authentication failed."
        else:
            error_msg = f"{error_name}: {str(e)[:100]}..."
        
        print(f"OpenAI API Error ({error_name}): {e}")

    # 5. Save messages to DB (only if successful)
    if not error_msg:
        user_msg = models.Message(
            thread_id=thread.id, 
            role="user", 
            content=chat_input.message
        )
        bot_msg = models.Message(
            thread_id=thread.id, 
            role="assistant", 
            content=bot_reply,
            response_id=response_id
        )
        
        db.add(user_msg)
        db.add(bot_msg)
        db.commit()
    
    return {
        "reply": bot_reply, 
        "thread_id": thread.id, 
        "user_id": user.id,
        "error": error_msg
    }

@app.get("/threads/{user_id}", response_model=List[ChatThreadSchema])
def get_threads(user_id: int, db: Session = Depends(get_db)):
    threads = db.query(models.ChatThread).filter(
        models.ChatThread.user_id == user_id
    ).order_by(models.ChatThread.created_at.desc()).all()
    return threads

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
