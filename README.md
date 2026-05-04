# Hybrid Search for Agentic Systems Demo

A self-contained Jupyter notebook demonstrating how to combine **BM25 sparse retrieval** and **kNN dense retrieval** into a hybrid search pipeline, then wrap it as a tool for an **agentic RAG loop** powered by an LLM.

> Based on: [How to Build Agentic RAG with Hybrid Search — Towards Data Science](https://towardsdatascience.com/how-to-build-agentic-rag-with-hybrid-search/)

---

## What's Inside

| Section | What it covers |
|---------|---------------|
| **Keyword (BM25)** | Inverted-index keyword retrieval with TF-IDF+ scoring |
| **Semantic (Dense)** | Sentence embeddings (`all-MiniLM-L6-v2`) + FAISS flat index |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) and weighted convex combination |
| **GraphRAG** | Graph Augmented Retrieval using extracted conceptual entities |
| **Agentic RAG** | LLM tool-use loop with dynamic query rewriting and alpha selection |

---

## Methodologies Compared

This repository explores several search paradigms using a purposefully challenging corpus (a mock banking knowledgebase filled with precise status codes and overlapping concepts):

*   **Keyword (Sparse / BM25)**: Excels at exact term matching (e.g., error codes like `PENDING_KB200`, IDs, specific jargon) but struggles with synonyms or implied concepts.
*   **Semantic (Dense / kNN)**: Excels at understanding user intent and broad concepts even when exact words don't match, but frequently fails on exact identifiers.
*   **Hybrid Search**: Merges sparse and dense ranked lists, filling the gaps of each method. It consistently outperforms either method alone on vocabulary-mismatched domains.
*   **GraphRAG**: Broadens context by extracting conceptual entities and traversing their relationship neighborhoods.

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
| `anthropic` | LLM API client (configurable for others) |
| `numpy` / `scikit-learn` | Numerics & evaluation |
| `python-dotenv` | Load API keys from `.env` |

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

Weighted score fusion after min-max normalization:

$$\text{score}(d) = \alpha \cdot \hat{s}_{\text{dense}}(d) + (1-\alpha) \cdot \hat{s}_{\text{sparse}}(d)$$

| α | Behavior |
|---|-----------|
| 0.1 | Keyword-heavy (error codes, SKUs) |
| 0.5 | Balanced |
| 0.9 | Semantic-heavy (conversational) |

### Graph Augmented Retrieval (GraphRAG)

Extracts conceptual entities to build a relationship layout, then retrieves documents via graph-traversed neighborhoods. Expands on traditional dense and sparse search by taking hops through connected domain knowledge.

### Agentic RAG

To handle complex and ambiguous queries, we wrap the hybrid search engine as a functional tool for an LLM. The agentic loop allows the LLM to autonomously navigate the tricky dataset by:
1. Formulating and refining the **query** based on the results of previous search attempts.
2. Dynamically adjusting the **alpha** weight (e.g., using a lower alpha specifically to hunt for exact error codes, or higher for broad contexts).
3. Retrying the search until sufficient context is retrieved to synthesize a definitive answer.

```
User → LLM ─┬─ hybrid_search(query="PENDING_KB200 status", alpha=0.1) → ...
            ├─ hybrid_search(query="check clearance in progress", alpha=0.6) → ...
            └─ synthesise → Final answer
```

---

## Sources

- [How to Build Agentic RAG with Hybrid Search — Towards Data Science](https://towardsdatascience.com/how-to-build-agentic-rag-with-hybrid-search/)
- [Hybrid Search for RAG: BM25, SPLADE, and Vector Search Combined — PremAI Blog](https://blog.premai.io/hybrid-search-for-rag-bm25-splade-and-vector-search-combined/)
- [Hybrid Search Explained — Weaviate](https://weaviate.io/blog/hybrid-search-explained)
- [Introducing Reciprocal Rank Fusion — OpenSearch](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/)
- [Full-text search for RAG apps: BM25 & hybrid search — Redis](https://redis.io/blog/full-text-search-for-rag-the-precision-layer/)
