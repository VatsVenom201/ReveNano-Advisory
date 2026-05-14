import os
import sys
import time
import hashlib
import datetime
import tiktoken
import threading
from groq import Groq
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings  # will be deprecated
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# Add parent directory to sys.path to allow relative imports from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import (
    VECTOR_DB_DIR, HISTORY_DB_DIR, USER_DATA_DB_DIR, USER_FILES_ROOT,
    EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME,
    GROQ_API_KEY, GROQ_MODEL, RETRIEVAL_K, FETCH_K,
    RERANK_TOP_N, COLLECTION_NAME, CACHE_DIR, 
    HISTORY_THRESHOLD,
    HISTORY_TOKEN_LIMIT, HISTORY_CHUNK_SIZE, HISTORY_CHUNK_OVERLAP,
    CHUNK_SIZE, CHUNK_OVERLAP, KNOWLEDGE_BASE_FOLDER
)
from utils.file_processing import extract_text_from_file, extract_soil_report_data
from utils.retrieval import rerank_documents, remove_duplicate_docs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingest import run_ingestion

# Set HuggingFace Cache
os.environ["HF_HOME"] = CACHE_DIR

class ChatEngine:
    def __init__(self):
        print("Initializing Chat Engine (Multi-Store)...")
        # 1. Models
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)
        self.llm = Groq(api_key=GROQ_API_KEY)
        
        # 2. Main Store (Shared Knowledge)
        self.knowledge_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=self.embedding_model,
            collection_name=COLLECTION_NAME
        )
        self.retriever = self.knowledge_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RETRIEVAL_K, "fetch_k": FETCH_K}
        )
        
        # 3. History Store (Isolated User Collections)
        self.history_store_dir = HISTORY_DB_DIR
        os.makedirs(self.history_store_dir, exist_ok=True)
        
        # 4. User Personal Document Store
        self.user_docs_dir = USER_DATA_DB_DIR
        os.makedirs(self.user_docs_dir, exist_ok=True)
        
        # 5. Tools
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", ". ", "\n"]
        )
        self.kb_sync_lock = threading.Lock()
        self.user_locks = {}
        self.user_locks_mutex = threading.Lock()
        
        # 6. Instance Cache (Now thread-safe)
        self.user_doc_collections = {}
        self.user_history_collections = {}
        
        # 7. Sync Throttling & Efficiency
        self.last_kb_sync_time = 0
        self.kb_last_mtime = 0
        self.user_last_mtime = {} # Tracks directory mtime per user_id

    def _get_user_lock(self, user_id):
        """Returns a unique threading lock for a specific user_id."""
        with self.user_locks_mutex:
            if user_id not in self.user_locks:
                self.user_locks[user_id] = threading.Lock()
            return self.user_locks[user_id]

    def _get_user_docs_db(self, user_id):
        """Returns a cached Chroma instance for user documents (thread-safe)."""
        with self.user_locks_mutex:
            if user_id not in self.user_doc_collections:
                collection_name = f"user_docs_{user_id.replace('-', '_')}"
                self.user_doc_collections[user_id] = Chroma(
                    persist_directory=self.user_docs_dir,
                    embedding_function=self.embedding_model,
                    collection_name=collection_name
                )
            return self.user_doc_collections[user_id]

    def _get_user_history_db(self, user_id):
        """Returns a cached Chroma instance for user history (thread-safe)."""
        with self.user_locks_mutex:
            if user_id not in self.user_history_collections:
                collection_name = f"user_history_{user_id.replace('-', '_')}"
                self.user_history_collections[user_id] = Chroma(
                    persist_directory=self.history_store_dir,
                    embedding_function=self.embedding_model,
                    collection_name=collection_name
                )
            return self.user_history_collections[user_id]
        
    def add_to_history(self, user_id, query, answer):
        """Pairs Query & Answer and stores them in a user-specific history collection."""
        user_lock = self._get_user_lock(user_id)
        with user_lock:
            user_history_db = self._get_user_history_db(user_id)
            
            try:
                # Combine Q and A for semantic pairing
                full_exchange = f"Q: {query}\nA: {answer}"
                
                chunks = []
                if len(full_exchange) <= HISTORY_CHUNK_SIZE:
                    chunks = [full_exchange]
                else:
                    start = 0
                    while start < len(full_exchange):
                        end = start + HISTORY_CHUNK_SIZE
                        chunks.append(full_exchange[start:end])
                        start += (HISTORY_CHUNK_SIZE - HISTORY_CHUNK_OVERLAP)
                
                metadatas = [{
                    "user_id": user_id, 
                    "timestamp": str(datetime.datetime.now(datetime.timezone.utc))
                } for _ in chunks]
                
                user_history_db.add_texts(texts=chunks, metadatas=metadatas)
                user_history_db.persist()
                print(f"DEBUG: Stored and persisted Q&A pair for {user_id} in semantic history.")
            except Exception as e:
                print(f"ERROR in add_to_history: {e}")

    def _compute_hash(self, content: bytes) -> str:
        """Returns the MD5 hex digest of raw file bytes."""
        return hashlib.md5(content).hexdigest()

    def _delete_doc_chunks(self, user_docs_db, filename: str):
        """Deletes all existing vector DB chunks for a given filename."""
        try:
            existing = user_docs_db.get(where={"source": filename})
            ids_to_delete = existing.get("ids", [])
            if ids_to_delete:
                user_docs_db.delete(ids=ids_to_delete)
                print(f"DEBUG: Deleted {len(ids_to_delete)} stale chunks for '{filename}'.")
        except Exception as e:
            print(f"WARNING: Could not delete stale chunks for '{filename}': {e}")

    def add_user_document(self, user_id, filename, text, file_size=None, raw_content: bytes = None):
        """Chunks and stores a personal document for a specific user.
        
        Deduplication logic:
          - If the same file (same name + same MD5 hash) is already indexed → skip.
          - If the same filename exists but with a different hash (file updated) → delete stale chunks, re-index.
        """
        user_lock = self._get_user_lock(user_id)
        with user_lock:
            print(f"\n--- UPLOAD START: {filename} for {user_id} ---")
            user_docs_db = self._get_user_docs_db(user_id)
            
            try:
                # --- Strong Deduplication Check (MD5 Content Hash) ---
                file_hash = self._compute_hash(raw_content) if raw_content else None
                if file_hash:
                    # 1. First, check if this EXACT content (hash) already exists in DB
                    # (This catches duplicates even if the filename is different)
                    existing_by_hash = user_docs_db.get(where={"file_hash": file_hash})
                    if existing_by_hash["ids"]:
                        print(f"--- UPLOAD SKIPPED (Duplicate Content): Hash {file_hash[:8]}... is already indexed. ---\n")
                        return

                    # 2. If content is different but filename is same, delete old version
                    existing_by_name = user_docs_db.get(where={"source": filename})
                    if existing_by_name["ids"]:
                        print(f"DEBUG: '{filename}' exists but content has changed. Removing stale chunks...")
                        self._delete_doc_chunks(user_docs_db, filename)
                # -----------------------------------------------------

                print(f"DEBUG: Extracting and chunking {filename}...")
                from langchain_core.documents import Document as LCDocument
                
                metadata = {"source": filename, "user_id": user_id}
                if file_size is not None:
                    metadata["size"] = file_size
                if file_hash:
                    metadata["file_hash"] = file_hash
                
                # Specialized Extraction for Soil Reports
                soil_data = extract_soil_report_data(text)
                if soil_data:
                    print(f"DEBUG: Specialized Soil Report detected for {filename}. Storing as clean chunk.")
                    metadata["type"] = "soil_report"
                    user_docs_db.add_texts(texts=[soil_data], metadatas=[metadata])
                    user_docs_db.persist()
                    print(f"--- UPLOAD COMPLETE (Specialized): {filename} indexed ---\n")
                    return

                # Default RAG Workflow (Recursive Chunking)
                raw_doc = [LCDocument(page_content=text, metadata=metadata)]
                chunks = self.text_splitter.split_documents(raw_doc)
                
                if chunks:
                    print(f"DEBUG: Embedding {len(chunks)} chunks into vector store...")
                    user_docs_db.add_documents(chunks)
                    print("DEBUG: Force persisting vector store...")
                    user_docs_db.persist()
                    print(f"--- UPLOAD COMPLETE: {filename} indexed and persisted ---\n")
            except Exception as e:
                print(f"ERROR in add_user_document: {e}")

    def sync_user_documents(self, user_id):
        """Folder-backed sync for a specific user's personal documents (Optimized)."""
        from utils.file_processing import extract_text_from_file
        
        user_folder = os.path.join(USER_FILES_ROOT, user_id)
        os.makedirs(user_folder, exist_ok=True)
        
        # --- Optimization: Directory mtime check ---
        current_mtime = os.path.getmtime(user_folder)
        last_mtime = self.user_last_mtime.get(user_id, 0)
        
        if current_mtime <= last_mtime:
            # print(f"DEBUG [{user_id}]: Personal folder unchanged (mtime). Skipping sync.")
            return
            
        print(f"DEBUG [{user_id}]: Syncing personal folder... [mtime changed]")
        
        user_docs_db = self._get_user_docs_db(user_id)
        
        try:
            # 1. Get existing hashes from DB (Fast lookup)
            from utils.helpers import get_file_hash
            existing_data = user_docs_db.get()
            existing_hashes = set()
            if existing_data['metadatas']:
                for m in existing_data['metadatas']:
                    h = m.get('file_hash')
                    if h:
                        existing_hashes.add(h)
            
            # 2. Scan folder for new or changed files
            files = [f for f in os.listdir(user_folder) if os.path.isfile(os.path.join(user_folder, f))]
            to_process = []
            for fname in files:
                fpath = os.path.join(user_folder, fname)
                fhash = get_file_hash(fpath)
                
                if fhash not in existing_hashes:
                    fsize = os.path.getsize(fpath)
                    to_process.append((fname, fpath, fsize, fhash))
            
            if to_process:
                print(f"DEBUG [{user_id}]: Found {len(to_process)} new/updated personal files to index.")
                for fname, fpath, fsize, fhash in to_process:
                    with open(fpath, "rb") as f:
                        content = f.read()
                    text = extract_text_from_file(content, fname)
                    if text.strip():
                        from langchain_core.documents import Document as LCDocument
                        metadata = {"source": fname, "user_id": user_id, "size": fsize, "file_hash": fhash}
                        
                        # Specialized Extraction for Soil Reports (Sync)
                        soil_data = extract_soil_report_data(text)
                        if soil_data:
                            print(f"DEBUG [{user_id}]: Specialized Soil Report detected (Sync) for {fname}.")
                            metadata["type"] = "soil_report"
                            user_docs_db.add_texts(texts=[soil_data], metadatas=[metadata])
                        else:
                            # Normal Chunking
                            raw_doc = [LCDocument(page_content=text, metadata=metadata)]
                            chunks = self.text_splitter.split_documents(raw_doc)
                            if chunks:
                                user_docs_db.add_documents(chunks)
                                print(f"DEBUG [{user_id}]: Indexed {len(chunks)} chunks from {fname} (Size: {fsize}).")
                
                user_docs_db.persist()
            
            # --- Update last sync tracking ---
            self.user_last_mtime[user_id] = current_mtime
            print(f"DEBUG [{user_id}]: Personal sync complete.\n")
        except Exception as e:
            print(f"ERROR [{user_id}] in sync_user_documents: {e}")

    def retrieve_relevant_history(self, user_query, user_id):
        """Semantic search over the user's isolated history collection."""
        try:
            user_history_db = self._get_user_history_db(user_id)
            # Use k=10 to gather more candidates for global reranking
            results = user_history_db.similarity_search_with_score(user_query, k=10)
            
            # Tag history docs as such in metadata
            history_docs = []
            for doc, score in results:
                if score >= HISTORY_THRESHOLD:
                    doc.metadata["store"] = "history"
                    history_docs.append(doc)
            return history_docs
        except Exception as e:
            print(f"ERROR in retrieve_relevant_history: {e}")
            return []

    def retrieve_user_documents(self, user_query, user_id):
        """Semantic search over the user's personal documents."""
        print(f"DEBUG: Retrieving personal documents for {user_id}... (k=5)")
        try:
            user_docs_db = self._get_user_docs_db(user_id)
            
            # Use k=10 as initial search
            results = user_docs_db.similarity_search_with_score(user_query, k=10)
            
            # Noise filter: Skip chunks with legal boilerplate or metadata noise
            noise_keywords = [
                "Legal Terms", "Disclaimer", "rights reserved", 
                "legally binding", "not a substitute", " Ahmedabad",
                "info@reve", "grievance officer"
            ]
            
            filtered_docs = []
            for doc, score in results:
                content = doc.page_content.lower()
                if any(kw.lower() in content for kw in noise_keywords):
                    continue
                filtered_docs.append(doc)
                
            # Limit to top 5 clean chunks
            relevant_docs = filtered_docs[:5]
            print(f"DEBUG: Found {len(relevant_docs)} clean personal document chunks.")
            return relevant_docs
        except Exception as e:
            print(f"ERROR in retrieve_user_documents: {e}")
            return []

    def sync_knowledge_base(self):
        """Lightweight sync of the shared knowledge base (Throttled + mtime check)."""
        with self.kb_sync_lock:
            # 1. Directory mtime check (Fastest) - skips walk if folder is unchanged
            try:
                current_mtime = os.path.getmtime(KNOWLEDGE_BASE_FOLDER)
                if current_mtime <= self.kb_last_mtime:
                    # print("DEBUG: KB Sync skipped (Folder unchanged).")
                    return
            except Exception as e:
                print(f"WARNING: Could not check KB folder mtime: {e}")
                current_mtime = 0

            # 2. Throttling (Coarse check to prevent high-frequency folder walks)
            now = time.time()
            if now - self.last_kb_sync_time < 300: # 5 minutes
                # print("DEBUG: KB Sync skipped (Throttled).")
                return
                
            print("DEBUG: Checking Knowledge Base for new files... [WAITING]")
            stats = run_ingestion(existing_model=self.embedding_model, existing_db=self.knowledge_db)
            if stats["processed"] > 0 or stats["deleted"] > 0:
                print(f"DEBUG: Knowledge Base Synced. Added: {stats['processed']}, Deleted: {stats['deleted']}")
            else:
                print("DEBUG: Knowledge Base is up to date.")
            
            self.last_kb_sync_time = now
            self.kb_last_mtime = current_mtime

    def format_chronological_history(self, all_messages):
        """
        Filters messages to fit within HISTORY_TOKEN_LIMIT.
        all_messages should be a list of models.Message or dicts from SQL.
        """
        current_tokens = 0
        limited_history = []
        
        # Assume messages are ordered newest first (standard for token limiting)
        for msg in all_messages:
            role = msg.get("role") if isinstance(msg, dict) else msg.role
            content = msg.get("content") if isinstance(msg, dict) else msg.content
            
            role_label = "Farmer" if role == "user" else "Advisor"
            msg_text = f"{role_label}: {content}\n"
            msg_tokens = len(self.encoding.encode(msg_text))
            
            if current_tokens + msg_tokens <= HISTORY_TOKEN_LIMIT:
                current_tokens += msg_tokens
                limited_history.insert(0, msg_text) # Keep chronological (oldest to newest)
            else:
                break
                
        return "".join(limited_history), current_tokens

    def response_llm(self, user_query, user_id, thread_id, chronological_history_list):
        # 0. Sync Store Layers (Granular Sequential Sync)
        # Syncing KB and User docs are independent; each uses its own lock.
        self.sync_knowledge_base()
        
        user_lock = self._get_user_lock(user_id)
        with user_lock:
            self.sync_user_documents(user_id)
        
        # 1. Start RAG Retrieval (Lock-Free)
        # Multiple users can now perform retrieval and LLM calls in parallel.
        print(f"\n--- CHAT START (User: {user_id} | Thread: {thread_id}) ---")
        print(f"QUERY [{user_id}]: {user_query}")
        
        # 1. Determine Routing Mode (Flexible)
        q_low = user_query.lower()
        is_report_query = (
            ("my" in q_low and "report" in q_low) or
            ("this" in q_low and "report" in q_low) or
            ("my" in q_low and "soil" in q_low) or
            ("this" in q_low and "soil" in q_low) or
            ("soil" in q_low and "report" in q_low) or
            ("analysis" in q_low and any(w in q_low for w in ["my", "this", "from", "report"]))
        )
        
        # 2. Personal Document Retrieval (Priority)
        personal_docs = self.retrieve_user_documents(user_query, user_id)
        personal_context = ""
        for d in personal_docs:
            src = os.path.basename(d.metadata.get('source', 'unknown'))
            personal_context += f"--- Document: {src} ---\n{d.page_content}\n\n"

        # 3. Knowledge Base Retrieval (Skip if Report Mode)
        shared_context = ""
        retrieved_docs = []
        if not is_report_query:
            print(f"DEBUG [{user_id}]: Searching shared knowledge base...")
            candidate_docs = self.retriever.invoke(user_query)
            unique_kb_candidates = remove_duplicate_docs(candidate_docs)
            retrieved_docs = rerank_documents(self.reranker, user_query, unique_kb_candidates, top_n=RERANK_TOP_N)
            print(f"DEBUG [{user_id}]: Found {len(retrieved_docs)} reranked shared document chunks.")
            for doc in retrieved_docs:
                src = os.path.basename(doc.metadata.get('source', 'unknown'))
                shared_context += f"[Source: {src}]\n{doc.page_content}\n\n"
        else:
            print(f"DEBUG [{user_id}]: Report Keywords Detected. Skipping Knowledge Base.")

        # 4. History Assembly (Two-Layer)
        # Layer A: Semantic RAG — top 3 relevant past Q&A chunks
        semantic_history_docs = self.retrieve_relevant_history(user_query, user_id)
        semantic_history_docs = semantic_history_docs[:3]  # hard cap at 3 chunks

        # Layer B: Recency anchor — last 4 SQL messages (guaranteed recent context)
        # chronological_history_list comes in NEWEST-first from SQL
        recent_msgs_raw = chronological_history_list[:4]   # take 4 newest
        recent_msgs_raw = list(reversed(recent_msgs_raw))  # flip to oldest→newest

        # Build history block — token-budgeted
        HISTORY_TOKEN_BUDGET = 600  # hard ceiling to protect context window
        history_lines = []
        history_tokens_used = 0

        # A) Semantic chunks first (background context)
        if semantic_history_docs:
            for doc in semantic_history_docs:
                chunk_text = doc.page_content.strip()
                chunk_tokens = len(self.encoding.encode(chunk_text))
                if history_tokens_used + chunk_tokens <= HISTORY_TOKEN_BUDGET:
                    history_lines.append(chunk_text)
                    history_tokens_used += chunk_tokens
                else:
                    break

        # B) Recent messages (always appended if budget allows)
        for msg in recent_msgs_raw:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            label = "Farmer" if role == "user" else "Advisor"
            line = f"{label}: {content}"
            line_tokens = len(self.encoding.encode(line))
            if history_tokens_used + line_tokens <= HISTORY_TOKEN_BUDGET:
                history_lines.append(line)
                history_tokens_used += line_tokens

        history_block = "\n".join(history_lines) if history_lines else None
        print(f"DEBUG [{user_id}]: History block — {history_tokens_used} tokens, {len(history_lines)} entries.")

        # 5. Build Final Labeled Prompt
        history_section = (
            f"""════════════════════════════════════════
[SECTION C — CONVERSATION HISTORY — READ-ONLY CONTEXT]
⚠️  PURPOSE: Use for conversational context and meta-questions about the conversation (e.g. "list my questions", "what did I ask before", pronouns like "it", "that report").
⚠️  DO NOT re-answer questions already answered here. DO NOT treat history values as current ground truth.
────────────────────────────────────────
{history_block}
────────────────────────────────────────
════════════════════════════════════════"""
            if history_block else
            "[SECTION C — CONVERSATION HISTORY]: No prior history available."
        )

        combined_prompt = f"""Below are labeled reference sections. Read ALL sections before responding.

════════════════════════════════════════
[SECTION A — USER PERSONAL DOCUMENTS]  (HIGHEST PRIORITY)
(Farmer's own uploaded files: soil reports, field data. Use as ground truth for personal questions.)
────────────────────────────────────────
{personal_context.strip() if personal_context else "No personal documents found for this query."}
════════════════════════════════════════

════════════════════════════════════════
[SECTION B — SHARED AGRICULTURE KNOWLEDGE BASE]  (SECOND PRIORITY)
(General agriculture textbook knowledge. Use for general questions.)
────────────────────────────────────────
{shared_context.strip() if shared_context else "No shared knowledge used for this query."}
════════════════════════════════════════

{history_section}

████████████████████████████████████████
[SECTION D — YOUR TASK: ANSWER ONLY THIS QUESTION]
Your one and only job is to answer the question below.
If the question is about the conversation itself (e.g. listing past questions), use Section C.
For all factual/agricultural questions, use Section A or B.
████████████████████████████████████████
QUESTION: {user_query}
████████████████████████████████████████
"""
        print(f"\n--- FINAL PROMPT SENT TO LLM [{user_id}] ---")
        print(combined_prompt)
        print("--------------------------------\n")

        history_msgs = [
            {
                "role": "system",
                "content": (
                    """You are a Senior Agricultural Advisor. Your responses are shown directly to the farmer — write for them, not for an AI system.

You receive four labeled context sections. Follow these rules EXACTLY:

━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION USAGE RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━
[SECTION A — USER PERSONAL DOCUMENTS]
- Use ONLY if the farmer's question explicitly mentions THEIR OWN soil, field, or report (e.g. "my soil", "my report", "my farm").
- For ALL other questions (general knowledge, comparisons, how-to), IGNORE Section A entirely.
- Do NOT volunteer soil report data, parameters, or personal analysis unless the farmer directly asked for it.

[SECTION B — SHARED KNOWLEDGE BASE]
- Use for general agriculture questions (soil types, crop advice, farming techniques, comparisons).
- Cross-reference with Section A only when the farmer explicitly asks to compare their data against general knowledge.

[SECTION C — CONVERSATION HISTORY]
- Use to resolve conversational references (e.g. "it", "that crop", "last time").
- Use to answer meta-questions about the conversation (e.g. "list my questions", "what did I ask?").
- NEVER re-answer questions already answered in history. NEVER treat history values as current ground truth.

[SECTION D — THE FARMER'S QUESTION]
- This is THE ONLY thing you are answering. Focus 100% on this.

━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━
- Answer ONLY what was asked. Do not add unrequested tables, parameters, or advice sections.
- NEVER narrate your reasoning. Never say "I will now...", "Based on the context...", "Since the question asks...", or "I will do the following". Go straight to the answer.
- NEVER explain what you are about to do. Just do it.
- For a full soil analysis (only when explicitly asked): use ### 📊 Soil Parameters (table), ### 🔬 Key Findings (critical values only), ### 🚜 Actionable Advice (top 3-5 points).
- For general questions: answer directly using Section B. Do not include personal soil tables.
- Use Markdown formatting (bold, tables, bullet points) where it improves clarity.
- If a value is missing from documents, say so — do not guess.
- Reply in the same language as the farmer's question."""
                )
            },
            {"role": "user", "content": combined_prompt}
        ]
        
        # 5. LLM Call
        response = self.llm.chat.completions.create(
            messages=history_msgs,
            model=GROQ_MODEL,
            temperature=0.8,
        )
        
        output = response.choices[0].message.content
        print(f"\n--- RAW LLM OUTPUT [{user_id}] ---")
        print(output)
        print("----------------------\n")
        
        # Merge sources for UI visibility
        all_source_docs = personal_docs + retrieved_docs
        sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in all_source_docs]
        
        print(f"--- CHAT COMPLETE: {user_id} ---\n")
        return output, sources

# For unified access
_engine = None

def response_llm(user_query, user_id, thread_id, chronological_history_list):
    global _engine
    if _engine is None:
        _engine = ChatEngine()
    return _engine.response_llm(user_query, user_id, thread_id, chronological_history_list)

if __name__ == "__main__":
    # Test
    engine = ChatEngine()
    
