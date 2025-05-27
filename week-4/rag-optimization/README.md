# RAG Optimization Techniques

This module demonstrates four advanced techniques to optimize Retrieval-Augmented Generation (RAG) systems for improved performance, accuracy, and relevance. These techniques can significantly enhance your RAG pipeline's ability to retrieve and generate high-quality responses.

## Overview

RAG systems often fall short due to suboptimal retrieval steps. This module implements four cutting-edge optimization techniques:

1. **Contextual Retrieval** - Adds context to chunks before embedding (Anthropic)
2. **Query Expansion** - Generates multiple related queries for broader coverage
3. **Self-Query** - Extracts metadata filters from natural language queries
4. **Reranking with Cohere** - Reorders results using advanced relevance scoring

## Prerequisites

### Required API Keys

- **OpenAI API Key**: For embeddings and LLM generation
- **Cohere API Key**: For reranking functionality (get free key at [dashboard.cohere.com](https://dashboard.cohere.com/api-keys))

### Database Setup

Ensure you have the PostgreSQL database with pgvector extension running (see `../pgvector-setup/` for setup instructions).

## Techniques Explained

### 1. Contextual Retrieval (`1-contextual-retrieval.py`)

**What it does**: Adds contextual information to each document chunk before embedding, preserving important context that might be lost during chunking.

**Key Benefits**:
- Reduces retrieval failure rate by 35% [(Anthropic's findings)](https://www.anthropic.com/news/contextual-retrieval)
- Preserves document context lost during chunking
- Improves semantic understanding of isolated chunks

**How it works**:
```python
# Original chunk
"The company's revenue grew by 3% over the previous quarter."

# With contextual retrieval
"This chunk is from an SEC filing on ACME corp's performance in Q2 2023; 
the previous quarter's revenue was $314 million. 

The company's revenue grew by 3% over the previous quarter."
```

### 2. Query Expansion (`2-query-expansion.py`)

**What it does**: Generates multiple related queries from the original user query to increase retrieval coverage and handle vocabulary mismatches.

**Key Benefits**:
- Increases recall by covering semantic variations
- Handles vocabulary mismatch between query and documents
- Improves retrieval for ambiguous or short queries
- Works especially well with keyword-based search (BM25)

**How it works**:
```python
# Step 1: Generate expanded queries
original_query = "How does document parsing work?"
expanded_queries = [
    "How does document parsing work?",
    "What is text extraction from documents?", 
    "Document processing and analysis methods",
    "OCR and document digitization techniques"
]

# Step 2: Search database with each query
all_results = []
for query in expanded_queries:
    docs = vector_search(query, k=3)  # Get top 3 for each query
    all_results.extend(docs)

# Step 3: Deduplicate results (remove duplicates)
unique_docs = deduplicate_by_content_hash(all_results)

# Step 4: Sort by relevance and return top results
final_docs = sort_by_similarity(unique_docs)[:5]
```

**Key advantage**: Instead of searching once with the original query, we search multiple times with different phrasings, then combine and deduplicate the results. This dramatically increases coverage by capturing documents that might use different terminology than the user's original query.

### 3. Self-Query (`3-self-query.py`)

**What it does**: Extracts structured metadata filters from natural language queries, enabling more precise retrieval by combining semantic search with filtering.

**Key Benefits**:
- Combines semantic search with structured filtering
- Extracts metadata filters from natural language queries
- Improves precision by filtering on document attributes
- Handles complex queries with multiple constraints

**How it works**:
```python
# Natural language query
"Find research papers about machine learning methodology"

# Extracted components
{
    "semantic_query": "machine learning methodology",
    "filters": {
        "document_type": "research_paper",
        "section": "methodology",
        "topic": "machine_learning"
    }
}
```

### 4. Cohere Reranking (`4-cohere-reranking.py`)

**What it does**: Uses Cohere's rerank API to reorder retrieved documents based on their relevance to the query, providing more nuanced relevance scoring than simple similarity metrics.

**Key Benefits**:
- Improves precision by reordering results by relevance
- Reduces retrieval failure rate by up to 67% when combined with other techniques
- Works with any initial retrieval method (semantic, keyword, hybrid)
- Provides more nuanced relevance scoring than cosine similarity

**How it works**:
```python
# Initial retrieval (top 20 documents)
initial_docs = retrieve_documents(query, k=20)

# Rerank using Cohere
reranked_docs = cohere_rerank(query, initial_docs, top_n=5)

# Use top 5 reranked documents for generation
response = generate_response(query, reranked_docs)
```

## Integration with Existing RAG Pipeline

These techniques are designed to work with the existing RAG pipeline in `../rag-pipeline/`. They:

- Reuse the same vector database setup
- Leverage existing document processing
- Maintain compatibility with the RAG system architecture
- Can be integrated incrementally

## Additional Resources

- [Anthropic's Contextual Retrieval Paper](https://www.anthropic.com/news/contextual-retrieval)
- [Cohere Rerank Documentation](https://docs.cohere.com/docs/rerank)
- [Query Expansion Techniques](https://en.wikipedia.org/wiki/Query_expansion)
- [Self-Query in LangChain](https://python.langchain.com/docs/how_to/self_query/)
