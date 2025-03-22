"""
Test script for FastAPI endpoints
Run this after starting the server to test the API
"""

import requests
from pprint import pprint


def test_api():
    base_url = "http://localhost:8000"

    # Test health check
    print("\nTesting health check endpoint...")
    response = requests.get(f"{base_url}/")
    pprint(response.json())

    # Test text analysis
    print("\nTesting analysis endpoint...")
    data = {
        "text": "I absolutely love this new AI course! The exercises are practical and well-structured.",
        "max_tokens": 1000,
    }
    response = requests.post(f"{base_url}/analyze", json=data)
    pprint(response.json())


if __name__ == "__main__":
    test_api()
