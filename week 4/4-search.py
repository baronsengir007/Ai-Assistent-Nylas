import lancedb
from lancedb.embeddings import get_registry

# --------------------------------------------------------------
# Exercise 4: Implementing Vector Search Strategies
# --------------------------------------------------------------
"""
In this exercise, we'll explore different search strategies for retrieving information
from our vector database containing the Bitcoin whitepaper.

As discussed in Week 4, there are three primary search approaches:

1. Semantic Search (Vector Search): Finds documents with similar meaning using embedding similarity
2. Text Search (Keyword/BM25): Finds documents containing specific keywords
3. Hybrid Search: Combines both approaches for improved results

Each approach has different strengths and weaknesses:
- Semantic search excels at conceptual queries but may miss specific terms
- Text search is precise for keywords but misses related concepts
- Hybrid search combines strengths of both methods

We'll implement all three strategies using LanceDB's clean API.
"""

# --------------------------------------------------------------
# Connect to the database
# --------------------------------------------------------------
"""
First, we'll connect to our LanceDB database where we stored the
Bitcoin whitepaper chunks in the previous exercise.
"""

uri = "data/lancedb"
db = lancedb.connect(uri)

# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------
"""
Next, we'll load the table containing our Bitcoin whitepaper chunks.
"""

table = db.open_table("crypto")
print(f"Total chunks in database: {table.count_rows()}")

# --------------------------------------------------------------
# Create the index
# --------------------------------------------------------------

table.create_fts_index("text", replace=True)

# --------------------------------------------------------------
# Implement different search strategies
# --------------------------------------------------------------
"""
Now we'll implement and compare different search strategies.
LanceDB makes this easy with the 'query_type' parameter.
"""

# Get the OpenAI embedding function for consistency
func = get_registry().get("openai").create(name="text-embedding-3-large")

# 1. Semantic Search (Vector Search)
print("\n--- SEMANTIC SEARCH RESULTS ---")
print("Query: 'How does Bitcoin achieve trustless transactions?'")
vector_results = table.search(
    query="How does Bitcoin achieve trustless transactions?",
    query_type="vector",  # Explicitly use vector search
).limit(3)

print("\nTop 3 Results:")
for i, row in enumerate(vector_results.to_pandas().iterrows(), 1):
    data = row[1]
    print(f"{i}. {data['metadata'].get('title', 'Untitled Section')}")
    print(f"   Text snippet: {data['text'][:150]}...\n")

# 2. Text Search (Full-Text Search)
print("\n--- TEXT SEARCH RESULTS ---")
print("Query: 'double-spending problem'")
text_results = table.search(
    query="double-spending problem",
    query_type="fts",  # Use full-text search (similar to BM25)
).limit(3)

print("\nTop 3 Results:")
for i, row in enumerate(text_results.to_pandas().iterrows(), 1):
    data = row[1]
    print(f"{i}. {data['metadata'].get('title', 'Untitled Section')}")
    print(f"   Text snippet: {data['text'][:150]}...\n")

# 3. Hybrid Search
print("\n--- HYBRID SEARCH RESULTS ---")
print("Query: 'How does Bitcoin prevent double-spending?'")
hybrid_results = table.search(
    query="How does Bitcoin prevent double-spending?",
    query_type="hybrid",  # Combine vector and text search
).limit(3)

print("\nTop 3 Results:")
for i, row in enumerate(hybrid_results.to_pandas().iterrows(), 1):
    data = row[1]
    print(f"{i}. {data['metadata'].get('title', 'Untitled Section')}")
    print(f"   Text snippet: {data['text'][:150]}...\n")

# --------------------------------------------------------------
# Compare search strategies
# --------------------------------------------------------------
"""
Let's compare the different search strategies with a complex query
that benefits from both semantic understanding and keyword matching.
"""

query = "Explain how Bitcoin's blockchain prevents tampering with transaction history"

print("\n--- SEARCH STRATEGY COMPARISON ---")
print(f"Query: '{query}'")


# Define a helper function to get and display results
def display_search_results(query, query_type, alpha=None):
    search_args = {"query": query, "query_type": query_type}
    if alpha is not None and query_type == "hybrid":
        search_args["alpha"] = alpha

    results = table.search(**search_args).limit(3)
    df = results.to_pandas()

    print(f"\n{query_type.upper()} SEARCH RESULTS:")
    for i, row in enumerate(df.iterrows(), 1):
        data = row[1]
        print(f"{i}. {data['metadata'].get('title', 'Untitled Section')}")
        # Shorter snippet for comparison
        print(f"   Text snippet: {data['text'][:100]}...\n")

    return df


# Run all three search types
vector_df = display_search_results(query, "vector")
text_df = display_search_results(query, "fts")
hybrid_df = display_search_results(query, "hybrid")
