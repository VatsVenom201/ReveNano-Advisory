import numpy as np
from sentence_transformers import CrossEncoder

def rerank_documents(reranker, query, docs, top_n=3):
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_n]]

def remove_duplicate_docs(docs):
    unique_docs = []
    seen = set()
    for doc in docs:
        # Use first 200 chars as fingerprint for lightweight dedup
        fingerprint = doc.page_content.strip()[:200]
        if fingerprint not in seen:
            unique_docs.append(doc)
            seen.add(fingerprint)
    return unique_docs
