"""
Creating Embeddings with OpenAI

This file demonstrates how to create embeddings for text chunks using OpenAI's
embedding models, which is a crucial step in preparing data for vector databases.
"""

import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Create an embedding for a single text chunk."""
    response = client.embeddings.create(
        model=model,
        input=text,
        dimensions=1536,  # Default for text-embedding-3-small
    )
    return response.data[0].embedding


def batch_create_embeddings(
    texts: List[str], model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """Create embeddings for multiple text chunks in a single API call."""
    response = client.embeddings.create(model=model, input=texts, dimensions=1536)
    return [item.embedding for item in response.data]


def main():
    # Example texts
    texts = [
        "Vector databases are specialized databases designed to store and query vector embeddings.",
        "Embeddings are numerical representations of text that capture semantic meaning.",
        "Similarity search finds the most relevant vectors based on mathematical distance.",
    ]

    # Create single embedding
    print("Creating single embedding...")
    single_embedding = create_embedding(texts[0])
    print(f"Embedding dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print()

    # Create batch embeddings
    print("Creating batch embeddings...")
    batch_embeddings = batch_create_embeddings(texts)
    print(f"Number of embeddings: {len(batch_embeddings)}")
    print(f"Dimensions per embedding: {len(batch_embeddings[0])}")
    print(f"First embedding (first 5 values): {batch_embeddings[0][:5]}")


if __name__ == "__main__":
    main()
