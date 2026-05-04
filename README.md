# Hybrid Search for Agentic Systems Demo

A self-contained Jupyter notebook demonstrating how to combine **BM25 sparse retrieval** and **kNN dense retrieval** into a hybrid search pipeline, then wrap it as a tool for an **agentic RAG loop** powered by Claude.

> Based on: [How to Build Agentic RAG with Hybrid Search — Towards Data Science](https://towardsdatascience.com/how-to-build-agentic-rag-with-hybrid-search/)

---

## What's Inside

| Section | What it covers |
|---------|---------------|
| **Keyword (BM25)** | Inverted-index keyword retrieval with TF-IDF+ scoring |
| **Semantic (Dense)** | Sentence embeddings (`all-MiniLM-L6-v2`) + FAISS flat index |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) and weighted convex combination |
| **GraphRAG** | Graph Augmented Retrieval using extracted conceptual entities |
| **Agentic RAG** | Claude tool-use loop with dynamic query rewriting and alpha selection |

---

## Why Hybrid?

Pure vector search fails on exact identifiers:

```
"Error code ER_NO_DEFAULT_FOR_FIELD"   ← BM25 wins
"What does SKU-4821 cost?"             ← BM25 wins
"How do I speed up slow queries?"      ← Dense wins
```

Hybrid search consistently outperforms either method alone — **+26–31% NDCG** vs dense-only and **+18.5% MRR** vs BM25-only on vocabulary-mismatched domains (BEIR benchmark).

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/ronybot/Hybrid-Search-for-Agentic-Systems-Demo.git
cd Hybrid-Search-for-Agentic-Systems-Demo
```

### 2. Create a `.env` file

```bash
cp .env.example .env
# then edit .env and add your key
```

```
ANTHROPIC_API_KEY=sk-ant-...
```

> ⚠️ **Never commit your `.env` file.** It is listed in `.gitignore`.

### 3. Open the notebook

Run cells top-to-bottom. The first cell installs all dependencies automatically via `%pip install`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `rank-bm25` | BM25 sparse retrieval |
| `sentence-transformers` | Dense embeddings |
| `faiss-cpu` | ANN vector index |
| `anthropic` | Claude API client |
| `numpy` / `scikit-learn` | Numerics & evaluation |
| `python-dotenv` | Load `ANTHROPIC_API_KEY` from `.env` |

---

## Decision Guide — Which Strategy to Use?

| Methodology                               | Recommended Alpha/Strategy           |
|-------------------------------------------|--------------------------------------|
| **Exact identifiers** (error codes)       | Convex α ≈ 0.1 or pure Keyword       |
| **Conceptual/Semantic queries**           | Convex α ≈ 0.9 or pure Semantic      |
| **Mixed queries / No labels**             | RRF                                  |
| **Graph connections needed**              | GraphRAG                             |
| **Versatile search for changing reality** | Give the LLM control over alpha!     |

---

## Key Concepts

### Reciprocal Rank Fusion (RRF)

Score-agnostic fusion — uses only rank positions, not raw scores:

$$\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}$$

Best default when you have no labelled data.

### Convex Combination

Weighted score fusion after min-max normalisation:

$$\text{score}(d) = \alpha \cdot \hat{s}_{\text{dense}}(d) + (1-\alpha) \cdot \hat{s}_{\text{sparse}}(d)$$

| α | Behaviour |
|---|-----------|
| 0.1 | Keyword-heavy (error codes, SKUs) |
| 0.5 | Balanced |
| 0.9 | Semantic-heavy (conversational) |

### Graph Augmented Retrieval (GraphRAG)

Extracts conceptual entities to build a relationship layout, then retrieves documents via graph-traversed neighborhoods. Expands on traditional dense and sparse search by taking hops through connected domain knowledge.

### Agentic RAG

The LLM autonomously picks the query, alpha, and whether to retrieve again:

```
User → Claude ─┬─ hybrid_search(query_1, alpha=0.1) → ...
               ├─ hybrid_search(query_2, alpha=0.8) → ...
               └─ synthesise → Final answer
```

---

## Sources

- [How to Build Agentic RAG with Hybrid Search — Towards Data Science](https://towardsdatascience.com/how-to-build-agentic-rag-with-hybrid-search/)
- [Hybrid Search for RAG: BM25, SPLADE, and Vector Search Combined — PremAI Blog](https://blog.premai.io/hybrid-search-for-rag-bm25-splade-and-vector-search-combined/)
- [Hybrid Search Explained — Weaviate](https://weaviate.io/blog/hybrid-search-explained)
- [Introducing Reciprocal Rank Fusion — OpenSearch](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/)
- [Full-text search for RAG apps: BM25 & hybrid search — Redis](https://redis.io/blog/full-text-search-for-rag-the-precision-layer/)
