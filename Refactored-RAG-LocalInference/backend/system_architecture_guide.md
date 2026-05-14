# 🌾 Reve Nano Science: RAG System Architecture Guide

Welcome to the technical overview of the Agricultural RAG Microservice. This document explains how the system works, from the basic flow to the advanced multi-user isolation and retrieval logic.

---

## 1. System Overview
The **Reve Nano Science Chatbot** is a production-grade RAG (Retrieval-Augmented Generation) system. It combines general agricultural knowledge from textbooks with a farmer's personal field data to provide highly accurate, tailored advice.

### Inputs & Outputs
- **Inputs**: 
  - **Natural Language Queries**: "When should I plant my wheat?"
  - **Personal Documents**: PDFs, Word files, or Excel sheets (e.g., Soil Reports).
  - **Shared Knowledge Base**: Multiple heavy PDF textbooks and research papers containing core agricultural knowledge.
  - **Conversation Context**: The system remembers what you said previously.
- **Outputs**:
  - **Contextual Responses**: Answers verified against personal and shared data.
  - **Sources**: Transparent links to exactly which document/chunk was used.

---

## 2. Core Architecture
The system is built as a de-coupled **Microservice Architecture**:
1. **Frontend (Streamlit)**: A user-friendly chat interface with file upload and user identity management.
2. **Backend (FastAPI)**: The engine room that handles logic, database operations, and LLM orchestration.
3. **Storage Tier**:
   - **SQL Database**: Stores persistent chat history and audit logs.
   - **Vector Database (ChromaDB)**: Stores high-dimensional embeddings for semantic search.

---

## 3. Why Three Vector Databases?
To ensure high performance and strict data isolation, the system manages three distinct "logical" vector stores:

### A. Shared Knowledge Base (Global)
- **Content**: General agricultural textbooks, manuals, and datasets.
- **Purpose**: Provides a baseline of farming knowledge accessible to all users.
- **Deduplication**: Automatically handles folder-level sync (add/remove) for admins using **MD5 content hashing** to prevent redundant indexing of same-content files.

### B. User Personal Documents (Isolated)
- **Content**: Farmer-specific files (soil analysis, yield reports, farm maps).
- **Purpose**: Acts as the "Ground Truth" for personal questions.
- **Isolation**: Stored in a collection specifically named after the `user_id` (e.g., `user_docs_vats123`). No other user can search this data.

### C. Conversation History (Isolated)
- **Content**: Past Q&A pairs from the current user.
- **Purpose**: Allows semantic search over past conversations (e.g., "What did I ask about my pH last month?").
- **Isolation**: Each user has their own history vector store, preventing "cross-talk" between users.

---

## 4. Multi-Tenant Isolation
User privacy is enforced through **Logical Data Partitioning**:
- **Collection Naming**: When a User Alpha queries the system, the backend dynamically routes the search to `user_docs_Alpha`.
- **Retrieval Guardrails**: The search functions are hard-coded to require a `user_id`, ensuring it is impossible for User B to even accidentally retrieve User A's data.

---

## 5. The Retrieval Workflow
When a user asks a question, the system follows these steps:

### Step 1: Intelligent Sync (mtime & Hash optimization)
Before searching, the system checks if new files exist. To stay fast, it uses a two-layer verification:
1.  **mtime (modification time) check**: if your folder hasn't changed since your last message, it skips the expensive folder scan entirely.
2.  **MD5 Content Hashing**: If a folder change is detected, the system calculates the MD5 hash of files to ensure only *genuinely new or modified* content is processed.
**This prevents duplicate indexing even if files are renamed or re-uploaded, keeping the system efficient and the vector database clean.**

### Step 2: Routing Logic
The system analyzes the query to decide which databases to use:
- **Report Mode**: If you ask "Explain my soil report," the system prioritizes Section A (Personal Docs) and skips Section B (Shared Knowledge) to reduce noise.
- **General Mode**: For questions like "How to treat fungus?", it searches both bases.

### Step 3: Retrieval & Reranking
- **Initial Search**: The system finds the top 10 most similar text chunks from the relevant databases.
- **Reranking**: It uses a **Cross-Encoder Model** (BGE-Reranker) to score these 10 chunks against your query. Only the most relevant top N chunks are kept. This filters out "weak" matches.

### Step 4: Prompt Engineering (The Labeled Context)
The final prompt is structured with clear labels, following the exact sequence used in the engine:
- `[SECTION A — YOUR PERSONAL DOCUMENTS]` (Priority Ground Truth)
- `[SECTION B — SHARED AGRICULTURE KNOWLEDGE]` (General Context)
- `[SECTION C — PRIOR CONVERSATION HISTORY]` (Conversational Context)
- `[SECTION D — YOUR TASK: ANSWER THIS QUESTION]` (The User's current query)

This tells the LLM exactly which data is "Ground Truth" (Personal) and which is "Context" (History), ensuring the final answer in Section D is as accurate as possible.

---

## 6. Concurrency & Performance
The system is designed for **High Throughput**:
- **Granular Locking**: We replaced a single "Global Lock" with multiple micro-locks.
  - User A can sync files while User B is chatting.
  - Shared Knowledge sync only blocks a few milliseconds for a timestamp check.
- **Thread-Safety**: The backend handles multiple simultaneous requests (Parallel Processing) by allowing the most time-consuming step (LLM Generation) to run outside of any locks.

---

## 7. Future Proofing
While we currently use **ChromaDB** (local), the architecture is ready to move to **Qdrant** or **Milvus** for horizontal scaling. The separation of the three stores makes it easy to migrate specific user data without affecting the whole system.

---

## 8. Data Integrity & Deduplication (MD5)
The system uses **MD5 Hex Digests** as a "digital fingerprint" for every file to ensure reliability:
- **API Guardrails**: The `/upload` endpoint performs a "Fast-Path" check. If an incoming file's hash matches an existing one in the user's folder, disk writes and re-indexing are skipped.
- **Automated Chunk Cleanup**: When a file is modified, the system uses the `source` and `file_hash` metadata to identify and delete orphaned chunks before re-indexing, preventing "hallucinations" caused by stale data.
- **Physical Isolation**: Even with deduplication, hashes are checked within the scope of a `user_id`, maintaining strict privacy controls.
