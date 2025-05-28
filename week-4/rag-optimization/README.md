# RAG Optimization Techniques

This module demonstrates four advanced techniques to optimize Retrieval-Augmented Generation (RAG) systems for improved performance, accuracy, and relevance. These techniques address common RAG failures and can significantly enhance your pipeline's ability to retrieve and generate high-quality responses.

## Overview

RAG systems often struggle with retrieval quality, leading to poor response generation. Common issues include:
- **Context loss during chunking**: Important document structure and relationships are lost
- **Vocabulary mismatch**: User queries use different terms than documents
- **Lack of structured filtering**: Cannot filter by document attributes or metadata
- **Poor relevance ranking**: Similarity scores don't always reflect semantic relevance

This module implements four advanced optimization techniques that address these challenges:

1. **Contextual Retrieval** - Adds context to chunks before embedding
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

**The Problem**: When documents are split into chunks for vector storage, important contextual information is often lost. A chunk might contain "The company's revenue grew by 3%" but lose the context of which company, which time period, and what the baseline was.

**The Solution**: [Contextual Retrieval, introduced by Anthropic](https://www.anthropic.com/news/contextual-retrieval), adds contextual information to each chunk before embedding. This preserves the document structure and relationships that would otherwise be lost during chunking.

**How it works**:
1. **Document Processing**: The full document is processed and chunked normally
2. **Context Generation**: For each chunk, an LLM generates a brief context that situates the chunk within the overall document
3. **Context Prepending**: The generated context is prepended to the original chunk content
4. **Embedding**: The contextualized chunk (context + original content) is embedded and stored

**Example Transformation**:
```
Original chunk:
"The company's revenue grew by 3% over the previous quarter."

With contextual retrieval:
"This chunk is from an SEC filing on ACME corp's Q2 2023 performance; 
the previous quarter's revenue was $314 million. 

The company's revenue grew by 3% over the previous quarter."
```

**Key Benefits**:
- **35% reduction in retrieval failure rate** (Anthropic's findings)
- Preserves document context that gets lost during chunking
- Improves semantic understanding of isolated chunks
- Better handling of pronouns, references, and implicit information

**When to use**: Essential for documents with complex structure, cross-references, or where chunks depend on broader document context (legal documents, research papers, technical manuals).

### 2. Query Expansion (`2-query-expansion.py`)

**The Problem**: Users often phrase queries differently than how information appears in documents. A user might ask "How does document parsing work?" while the document uses terms like "text extraction," "OCR," or "document processing."

**The Solution**: Query expansion generates multiple related queries from the original user query, searches with each variant, then combines and deduplicates the results. This dramatically increases recall by capturing documents that use different terminology.

**Detailed Process**:
1. **Query Generation**: Use an LLM to generate 3-4 alternative phrasings of the original query
2. **Parallel Search**: Search the vector database with each query variant independently
3. **Result Aggregation**: Collect all results from all queries
4. **Deduplication**: Remove duplicate documents using content hashing
5. **Ranking**: Sort combined results by relevance score
6. **Selection**: Return top-k documents for generation

**Example Expansion**:
```python
Original query: "How does document parsing work?"

Expanded queries:
1. "How does document parsing work?" (original)
2. "What is text extraction from documents?"
3. "Document processing and analysis methods"
4. "OCR and document digitization techniques"
5. "Converting documents to structured data"
```

**Key Benefits**:
- **Increases recall** by covering semantic variations
- **Handles vocabulary mismatch** between query and documents
- **Improves retrieval for ambiguous queries** by exploring multiple interpretations
- **Works especially well with keyword-based search** systems
- **Captures domain-specific terminology** that users might not know

**When to use**: Particularly effective for technical domains, when users are unfamiliar with domain terminology, or when documents use varied vocabulary for similar concepts.

### 3. Self-Query (`3-self-query.py`)

**The Problem**: Users often have implicit filtering requirements in their queries. When someone asks "Find research papers about machine learning methodology," they want documents that are both semantically about machine learning AND have specific metadata attributes (document type: research paper, section: methodology).

**The Solution**: Self-query uses an LLM to extract structured metadata filters from natural language queries, then combines semantic search with precise filtering. This improves precision by ensuring results match both content and structural criteria.

**Architecture**:
1. **Schema Definition**: Define available metadata fields using Pydantic models with strict typing
2. **Query Analysis**: LLM analyzes the natural language query to extract semantic intent and metadata filters
3. **Structured Output**: Use OpenAI's structured output to ensure reliable filter extraction
4. **Dual Search**: Perform semantic search on the cleaned query, then apply metadata filters
5. **Filtered Results**: Return documents that match both semantic and structural criteria

**Example Query Decomposition**:
```python
Natural language query:
"Find research papers about machine learning methodology"

Extracted components:
{
    "semantic_query": "machine learning methodology",
    "filters": {
        "document_type": "research_paper",
        "section": "methodology", 
        "topic": "machine_learning"
    }
}
```

**Metadata Schema Example**:
```python
class MetadataSchema(BaseModel):
    content_type: Optional[Literal["tutorial", "research", "documentation", "case_study"]]
    difficulty: Optional[Literal["beginner", "intermediate", "advanced"]]
    topic: Optional[Literal["rag", "embeddings", "llm", "vector_search"]]
    section: Optional[Literal["introduction", "methodology", "results", "conclusion"]]
```

**Key Benefits**:
- **Combines semantic and structured search** for higher precision
- **Extracts implicit filtering requirements** from natural language
- **Handles complex queries** with multiple constraints
- **Reduces irrelevant results** by filtering on document attributes
- **Enables faceted search** through natural language

**When to use**: Essential for large document collections with rich metadata, when users need to filter by document attributes, or when precision is more important than recall.

### 4. Cohere Reranking (`4-cohere-reranking.py`)

**The Problem**: Vector similarity (cosine similarity) doesn't always correlate with semantic relevance. A document might have a lower similarity score but be more relevant to the user's actual intent. Traditional retrieval methods can miss highly relevant content that doesn't score well on similarity metrics.

**The Solution**: Cohere's rerank models use advanced transformer architectures trained specifically for relevance ranking. They analyze the relationship between queries and documents more deeply than simple similarity metrics.

**How Cohere Reranking Works**:
1. **Initial Retrieval**: Retrieve a larger set of documents (e.g., top 20-50) using standard methods
2. **Rerank API Call**: Send query and documents to Cohere's rerank endpoint
3. **Deep Analysis**: Cohere's model analyzes query-document relationships using:
   - **Cross-attention mechanisms** between query and document tokens
   - **Contextual understanding** of semantic relationships
   - **Intent matching** beyond keyword overlap
4. **Relevance Scoring**: Returns documents ranked by true semantic relevance
5. **Top-k Selection**: Use the top-ranked documents for generation

**Available Cohere Models**:
- **rerank-english-v3.0**: Latest English rerank model with 4k context length, optimized for accuracy
- **rerank-multilingual-v3.0**: Supports 100+ languages with strong cross-lingual capabilities
- **rerank-english-v2.0**: Previous generation model, still highly effective

**Model Capabilities**:
- **Context Length**: Up to 4,096 tokens per document
- **Batch Processing**: Rerank up to 1,000 documents per request
- **Language Support**: English and 100+ languages (multilingual model)
- **Domain Adaptation**: Works across various domains without fine-tuning

**Example Reranking Impact**:
```
Query: "How does AI help with image recognition?"

Before reranking (by similarity):
1. Sim: 0.540 | "AI techniques for classification..."
2. Sim: 0.536 | "Computer vision recognition tasks..."
3. Sim: 0.534 | "Deep learning applications..."

After reranking (by relevance):
1. Rerank: 0.783 | "AI applications in image recognition..." (was #13)
2. Rerank: 0.616 | "Deep learning for image recognition..." (was #6)  
3. Rerank: 0.510 | "Machine perception and vision..." (was #12)
```

**Key Benefits**:
- **Improves precision** by reordering results by true relevance
- **Surfaces hidden gems** - relevant documents with lower similarity scores
- **Reduces retrieval failure rate** by up to 67% when combined with other techniques
- **Works with any initial retrieval method** (semantic, keyword, hybrid)
- **No training required** - works out of the box across domains

**When to use**: Highly recommended for production RAG systems, especially when dealing with diverse content types, complex queries, or when retrieval quality is critical.

**Learn More**: 
- [Cohere Rerank Documentation](https://docs.cohere.com/docs/rerank)
- [Rerank API Reference](https://docs.cohere.com/reference/rerank)
- [Rerank Best Practices](https://docs.cohere.com/docs/reranking-best-practices)

## Integration with Existing RAG Pipeline

These techniques are designed to work seamlessly with the existing RAG pipeline in `../rag-pipeline/`. They:

- **Reuse the same vector database setup** and connection handling
- **Leverage existing document processing** and chunking strategies  
- **Maintain compatibility** with the RAG system architecture
- **Can be integrated incrementally** - add one technique at a time
- **Share the same embedding service** and OpenAI client configuration

## Building a Production RAG System with All Techniques

In a real-world application, these techniques work synergistically to create a robust, high-performance RAG system. Here's how you would architect and implement such a system:

### System Architecture

**Phase 1: Enhanced Document Ingestion**
Start with contextual retrieval during the ingestion phase. When processing documents, generate contextual information for each chunk before embedding. This creates a foundation of well-contextualized chunks that preserve document structure and relationships. Store both the original and contextualized content, along with rich metadata about document type, section, difficulty level, and topic.

**Phase 2: Intelligent Query Processing**
Implement a query processing pipeline that first uses self-query to extract structured filters and clean the semantic query. Then apply query expansion to generate multiple query variants. This dual approach ensures you capture both explicit filtering requirements and handle vocabulary mismatches.

**Phase 3: Multi-Stage Retrieval**
Execute retrieval in stages: first, use the expanded queries to retrieve a larger candidate set (20-50 documents) from your contextualized chunks. Apply the extracted metadata filters to ensure structural relevance. Then use Cohere reranking to reorder the filtered results by true semantic relevance, selecting the top 5-10 documents for generation.

## Additional Resources

- [Anthropic's Contextual Retrieval Paper](https://www.anthropic.com/news/contextual-retrieval)
- [Cohere Rerank Documentation](https://docs.cohere.com/docs/rerank)
- [Cohere Rerank API Reference](https://docs.cohere.com/reference/rerank)
- [Query Expansion Techniques](https://en.wikipedia.org/wiki/Query_expansion)