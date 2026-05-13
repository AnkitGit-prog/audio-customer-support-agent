# RAG Implementation Guide

## Overview

This document explains the Retrieval-Augmented Generation (RAG) architecture
used in the Audio Customer Support Agent.

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that improves LLM
responses by injecting relevant, factual context retrieved from a knowledge
base _before_ asking the model to generate an answer.

Without RAG:
```
User question → LLM → (may hallucinate or be outdated) → Answer
```

With RAG:
```
User question → Vector DB search → Top-K relevant docs
                                          ↓
             LLM (question + docs as context) → Grounded Answer
```

---

## Architecture in This Project

```
User Question (text)
        │
        ▼
┌─────────────────────────────────┐
│  CustomerSupportAgent._rag_search() │
│                                   │
│  1. Send query to ChromaDB        │
│  2. Retrieve top 3 documents      │
│     by cosine similarity          │
│  3. Format as context string      │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  CustomerSupportAgent.process_query() │
│                                   │
│  1. Receive context from RAG      │
│  2. Build system + user prompt    │
│  3. Call GPT-3.5-turbo API        │
│  4. Return generated answer       │
└─────────────────────────────────┘
        │
        ▼
   Final Answer
```

---

## ChromaDB Setup

### Persistence

ChromaDB uses a `PersistentClient` that saves data to disk (default: `./chroma_db`).
The knowledge base is only ingested **once** on the first server startup and
is automatically reloaded on subsequent restarts.

### Collection Configuration

```python
collection = client.get_or_create_collection(
    name="customer_support_kb",
    metadata={"hnsw:space": "cosine"},  # cosine similarity
)
```

### Why Cosine Similarity?

Cosine similarity measures the angle between two embedding vectors, making it
robust to different text lengths. It is the standard choice for semantic
search in RAG systems.

---

## Knowledge Base Documents

The knowledge base contains **16 documents** covering:

| # | Topic | ID |
|---|-------|----|
| 1 | Return Policy | kb_001 |
| 2 | Shipping Information | kb_002 |
| 3 | Warranty Policy | kb_003 |
| 4 | Payment Methods | kb_004 |
| 5 | Order Tracking | kb_005 |
| 6 | Customer Support Contact | kb_006 |
| 7 | Product Exchanges | kb_007 |
| 8 | Refund Process | kb_008 |
| 9 | International Shipping | kb_009 |
| 10 | Damaged Items Policy | kb_010 |
| 11 | Subscription Cancellation | kb_011 |
| 12 | Bulk Orders | kb_012 |
| 13 | Gift Cards | kb_013 |
| 14 | Account Management | kb_014 |
| 15 | Privacy Policy | kb_015 |
| 16 | Technical Support | kb_016 |

---

## Document Ingestion

Documents are upserted (insert-or-update) into ChromaDB using:

```python
collection.upsert(
    ids=[doc["id"] for doc in KNOWLEDGE_BASE],
    documents=[doc["content"] for doc in KNOWLEDGE_BASE],
    metadatas=[{"title": doc["title"], "id": doc["id"]} for doc in KNOWLEDGE_BASE],
)
```

ChromaDB automatically generates embeddings using its built-in
`all-MiniLM-L6-v2` model (via `sentence-transformers`) when no custom
embedding function is supplied.

---

## RAG Query Flow

```python
results = collection.query(
    query_texts=[query],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)
```

### Result structure:
```python
{
  "documents": [["doc1 text", "doc2 text", "doc3 text"]],
  "metadatas": [[{"title": "..."}, {"title": "..."}, {"title": "..."}]],
  "distances": [[0.12, 0.25, 0.38]]   # lower = more similar (cosine)
}
```

### Formatting retrieved context:

```
**Return Policy** (relevance: 0.88)
Our return policy allows customers to return any item within 30 days …

**Refund Process** (relevance: 0.74)
Approved refunds are processed within 3-5 business days …

**Product Exchanges** (relevance: 0.68)
Exchanges are accepted within 30 days of purchase …
```

---

## LLM Prompt Design

### System Prompt

```
You are a helpful and professional customer support agent.
Use ONLY the information provided in the context below to answer
the customer's question. Be concise, friendly, and accurate.
If the context does not contain enough information to answer
confidently, say so politely and direct the customer to contact
support directly.
```

### User Prompt Template

```
Context from our knowledge base:
{retrieved_documents}

Customer question: {user_query}

Please provide a helpful, accurate answer based on the context above.
```

---

## Distance Scores Explained

ChromaDB returns **cosine distance** (not similarity):
- `distance = 0.0` → identical vectors (perfect match)
- `distance = 1.0` → orthogonal vectors (completely unrelated)
- `distance = 2.0` → opposite vectors (maximum dissimilarity)

Convert to similarity: `similarity = 1 - distance`

**Typical good scores:**
- `distance < 0.3` → highly relevant
- `distance 0.3–0.6` → moderately relevant
- `distance > 0.6` → weakly relevant

---

## Extending the Knowledge Base

To add new documents:

1. Add entries to the `KNOWLEDGE_BASE` list in `src/llm/agent.py`:
   ```python
   {
       "id": "kb_017",
       "title": "Your New Topic",
       "content": "Full document text …",
   },
   ```

2. Delete the `./chroma_db` folder to force re-ingestion:
   ```bash
   rm -rf ./chroma_db
   ```

3. Restart the server — documents will be re-ingested automatically.

---

## Performance Tips

- Use `STT_MODEL=small` or `medium` for better transcription accuracy
  at the cost of more RAM/CPU.
- Keep individual documents under 500 tokens for best embedding quality.
- For production, consider using `OpenAIEmbeddings` from LangChain instead
  of ChromaDB's default embedder for higher-quality vector representations.
- Add a re-ranking step (e.g., cross-encoder) after initial retrieval to
  further improve result relevance.
