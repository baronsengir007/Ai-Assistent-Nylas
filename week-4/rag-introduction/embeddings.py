"""
Simple Vector Embeddings Example

This file demonstrates basic vector embedding concepts using OpenAI's embedding model.
It shows how to create embeddings and calculate similarity between different texts.
"""

import os
from typing import List
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_embedding(text: str) -> List[float]:
    """Create an embedding for a single text."""
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def main():
    # Example texts to compare
    texts = [
        "I love programming in Python",
        "Python is my favorite programming language",
        "The weather is beautiful today",
        "I enjoy coding in Python",
    ]

    # Create embeddings for all texts
    embeddings = [create_embedding(text) for text in texts]

    # Prove that OpenAI embeddings are normalized (magnitude = 1)
    print("\nProof that dot product = cosine similarity for normalized vectors:")
    print(f"Magnitude of first embedding: {np.linalg.norm(embeddings[0]):.6f}")
    print(f"Direct dot product: {np.dot(embeddings[0], embeddings[1]):.6f}")
    print(
        f"Cosine similarity: {calculate_similarity(embeddings[0], embeddings[1]):.6f}"
    )
    print("They are equal! ✓\n")

    # Compare each text with the first one
    print("Similarity scores with first text:")
    print("-" * 50)
    for i, text in enumerate(texts[1:], 1):
        similarity = calculate_similarity(embeddings[0], embeddings[i])
        print(f"Text 1: {texts[0]}")
        print(f"Text {i + 1}: {text}")
        print(f"Similarity: {similarity:.4f}\n")


if __name__ == "__main__":
    main()
