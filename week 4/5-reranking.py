import lancedb
from lancedb.embeddings import get_registry
import cohere
import os
from dotenv import load_dotenv

# --------------------------------------------------------------
# Exercise 5: Improving Search with Cohere Reranking
# --------------------------------------------------------------
"""
In this exercise, we'll enhance our search results using Cohere's reranking model.
The process will be:

1. Perform a hybrid search with more results (30)
2. Rerank these results using Cohere's reranker
3. Compare the results before and after reranking
"""

# --------------------------------------------------------------
# Setup and Configuration
# --------------------------------------------------------------

# Load environment variables
load_dotenv()

# Initialize Cohere client
cohere_client = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

# Connect to LanceDB
uri = "data/lancedb"
db = lancedb.connect(uri)
table = db.open_table("crypto")

# Get OpenAI embedding function
func = get_registry().get("openai").create(name="text-embedding-3-large")

# --------------------------------------------------------------
# Implement Reranking
# --------------------------------------------------------------


def search_and_rerank(query: str, num_results: int = 30):
    """
    Perform hybrid search and rerank the results.

    Args:
        query: Search query
        num_results: Number of initial results to fetch

    Returns:
        Original results and reranked results
    """
    # 1. Perform hybrid search
    print(f"\nPerforming hybrid search for: '{query}'")
    initial_results = table.search(
        query=query,
        query_type="hybrid",
    ).limit(num_results)

    # Convert to pandas for easier handling
    results_df = initial_results.to_pandas()

    # 2. Prepare documents for reranking
    documents = results_df["text"].tolist()

    # 3. Rerank with Cohere
    print("\nReranking results with Cohere...")
    rerank_results = cohere_client.rerank(
        query=query,
        documents=documents,
        model="rerank-v3.5",
        top_n=10,
    )

    # Sort original DataFrame based on reranking order
    reranked_indices = [result.index for result in rerank_results.results]
    reranked_df = results_df.iloc[reranked_indices].reset_index(drop=True)

    return results_df, reranked_df


def display_results_comparison(query: str, original_df, reranked_df):
    """Display and compare original and reranked results."""

    print("\n=== Original Top 3 Results (Hybrid Search) ===")
    for i, row in enumerate(original_df.head(3).iterrows(), 1):
        data = row[1]
        print(f"\n{i}. {data['metadata'].get('title', 'Untitled Section')}")
        print(f"Text snippet: {data['text'][:150]}...")

    print("\n=== Reranked Top 3 Results (Cohere) ===")
    for i, row in enumerate(reranked_df.head(3).iterrows(), 1):
        data = row[1]
        print(f"\n{i}. {data['metadata'].get('title', 'Untitled Section')}")
        print(f"Text snippet: {data['text'][:150]}...")


# --------------------------------------------------------------
# Test Different Queries
# --------------------------------------------------------------

test_queries = [
    "How does Bitcoin's proof of work prevent double spending?",
    "Explain the role of miners in transaction verification",
    "What makes Bitcoin transactions irreversible?",
]

for query in test_queries:
    print("\n" + "=" * 80)
    print(f"TESTING QUERY: {query}")

    # Get both original and reranked results
    original_df, reranked_df = search_and_rerank(query)

    # Display comparison
    display_results_comparison(query, original_df, reranked_df)
