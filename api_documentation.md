# AI Microservice API Documentation (v2)

This internal service handles agricultural advisory chats and thread management. It is designed to be called by another backend service.

---

## 1. Chat Endpoint (`/chat`)
**Purpose**: Send a user message and get an AI response.

### Expected Input
- **Method**: `POST`
- **URL**: `/chat`
- **Body (JSON)**:
```json
{
  "user_id": 101,       // Unique ID of the user (e.g. from your database)
  "thread_id": null,    // null for new chat OR integer for existing thread
  "message": "text"     // The user's question
}
```

### Expected Output
- **Status**: `200 OK`
- **Body (JSON)**:
```json
{
  "reply": "AI response text",
  "thread_id": 42,      // The ID of the thread (save this to continue chat)
  "user_id": 101        // Confirms which user this belongs to
}
```

---

## 2. List User Threads (`/threads/{user_id}`)
**Purpose**: Get all previous conversation threads for a specific user.

### Expected Input
- **Method**: `GET`
- **URL**: `/threads/101`  (where 101 is the `user_id`)
- **Body**: None

### Expected Output
- **Status**: `200 OK`
- **Body (JSON)**:
```json
[
  {
    "id": 42,                 // This is the THREAD_ID
    "user_id": 101,           // The ID of the user
    "created_at": "timestamp" // When the thread was created
  },
  {
    "id": 41,                 // Another THREAD_ID for the same user
    "user_id": 101,
    "created_at": "timestamp"
  }
]
```

---

## 3. Get Thread Messages (`/thread/{thread_id}`)
**Purpose**: Get the full conversation history (all messages) of a single thread.

### Expected Input
- **Method**: `GET`
- **URL**: `/thread/42`  (where 42 is the `thread_id`)
- **Body**: None

### Expected Output
- **Status**: `200 OK`
- **Body (JSON)**:
```json
[
  {
    "role": "user",
    "content": "What is the best soil for rice?",
    "timestamp": "timestamp"
  },
  {
    "role": "assistant",
    "content": "Rice grows best in clayey soil...",
    "timestamp": "timestamp"
  }
]
```

---

## 4. Delete Thread (`/thread/{thread_id}`)
**Purpose**: Permanently delete a thread and all its messages.

### Expected Input
- **Method**: `DELETE`
- **URL**: `/thread/42`
- **Body**: None

### Expected Output
- **Status**: `200 OK`
- **Body (JSON)**:
```json
{
  "message": "Thread deleted"
}
```
