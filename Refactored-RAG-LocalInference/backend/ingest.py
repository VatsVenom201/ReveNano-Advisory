import os
import sys
import gc

# Add parent directory to sys.path to allow relative imports from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from utils.config import (
    KNOWLEDGE_BASE_FOLDER, VECTOR_DB_DIR, EMBEDDING_MODEL_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME, CACHE_DIR
)
from utils.cleaning import clean_and_normalize_docs
from utils.helpers import get_supported_files, clear_memory, get_file_hash
from utils.file_processing import extract_text_from_file

# Set HuggingFace Cache
os.environ["HF_HOME"] = CACHE_DIR

def run_ingestion(existing_model=None, existing_db=None):
    print("--- Starting Ingestion Pipeline ---")
    stats = {"total_files": 0, "processed": 0, "skipped": 0, "deleted": 0, "failed": 0}
    
    # Initialize/Reuse Embedding Model
    embedding_model = existing_model if existing_model else HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Initialize Recursive Character Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", ". "]
    )
    
    # Initialize/Load ChromaDB
    if existing_db:
        vector_db = existing_db
    else:
        vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME
        )
    
    # Get existing hashes in DB to skip duplicates
    existing_entries = vector_db.get()
    existing_hashes = set()
    if existing_entries['metadatas']:
        for meta in existing_entries['metadatas']:
            h = meta.get('file_hash')
            if h:
                existing_hashes.add(h)
    
    # Track existing sources only for deletion cleanup
    existing_unique_sources = set()
    if existing_entries['metadatas']:
        for meta in existing_entries['metadatas']:
            if 'source' in meta:
                existing_unique_sources.add(meta['source'])
    
    all_files = get_supported_files(KNOWLEDGE_BASE_FOLDER)
    current_files_set = set(all_files)
    
    orphaned_sources = existing_unique_sources - current_files_set
    for orphan in orphaned_sources:
        try:
            filename = os.path.basename(orphan)
            print(f"Cleaning up deleted file from DB: {filename}")
            vector_db.delete(where={"source": orphan})
            stats["deleted"] += 1
        except Exception as e:
            print(f"FAILED to delete orphan {orphan}: {e}")
            
    # 2. Check for Additions (Sync Folder with DB)
    stats["total_files"] = len(all_files)
    print(f"Found {len(all_files)} supported files in knowledge base.")
    
    for i, file_path in enumerate(all_files):
        filename = os.path.basename(file_path)
        fhash = get_file_hash(file_path)
        
        # Check if file content already exists in DB (Fast Hash Match)
        if fhash in existing_hashes:
            stats["skipped"] += 1
            continue
            
        fsize = os.path.getsize(file_path)
        print(f"[{i+1}/{len(all_files)}] New or Updated File Detected: {filename} ({fsize} bytes)")
        
        try:
            # 1. Read Content and Extract Text (Unified Processing)
            with open(file_path, "rb") as f:
                content = f.read()
            
            text = extract_text_from_file(content, filename)
            
            if not text.strip():
                print(f"   - FAILED: No text extracted from {filename}")
                stats["failed"] += 1
                continue
            
            # 2. Chunking Logic
            from langchain_core.documents import Document
            metadata = {"source": file_path, "size": fsize, "file_hash": fhash}
            raw_doc = [Document(page_content=text, metadata=metadata)]
            
            # 3. Clean and Normalize (Optional, but kept for consistency)
            # docs = clean_and_normalize_docs(raw_doc)
            
            # 4. Chunk
            chunks = text_splitter.split_documents(raw_doc)
            
            # 5. Add to Vector DB
            if chunks:
                vector_db.add_documents(chunks)
                stats["processed"] += 1
                print(f"   - Successfully added {len(chunks)} chunks to ChromaDB")
            
            # 6. Free Memory
            del raw_doc
            del chunks
            clear_memory()
            
        except Exception as e:
            stats["failed"] += 1
            print(f"   - FAILED processing {filename}: {e}")
    
    print(f"\n--- Sync Completed: {stats['processed']} added, {stats['deleted']} deleted ---")
    return stats

def get_status():
    """Returns a dictionary containing the sync status of the knowledge base."""
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )
    
    data = vector_db.get()
    embedded_sources = set()
    if data['metadatas']:
        for meta in data['metadatas']:
            if 'source' in meta:
                embedded_sources.add(meta['source'])
    
    local_files = get_supported_files(KNOWLEDGE_BASE_FOLDER)
    local_files_set = set(local_files)
    
    status = {
        "embedded": [],
        "pending": [],
        "orphaned": []
    }
    
    # Track which sources we've found in the DB
    found_sources = set()
    
    for meta in data['metadatas']:
        src = meta.get('source')
        if src and src not in found_sources:
            found_sources.add(src)
            if src in local_files_set:
                count = sum(1 for m in data['metadatas'] if m.get('source') == src)
                status["embedded"].append({"filename": os.path.basename(src), "chunks": count})
            else:
                status["orphaned"].append(os.path.basename(src))
                
    for src in local_files:
        if src not in found_sources:
            status["pending"].append(os.path.basename(src))
            
    return status

if __name__ == "__main__":
    run_ingestion()
