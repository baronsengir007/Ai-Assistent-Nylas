"""
RAG Metrics Tutorial - Step 2: Setup Search Pipelines

This module sets up three different search pipelines for comparison:
1. Semantic Search: Pure vector similarity search
2. Hybrid Search: Combines semantic + keyword search using RRF
3. Hybrid + Reranking: Hybrid search with Cohere reranking

The pipelines are designed to be modular and consistent, enabling fair
comparison of retrieval quality metrics.
"""

# Import the pipeline classes from our search_pipelines module
from search_pipelines import (
    SearchPipelineManager,
)


def demonstrate_pipelines():
    """Demonstrate the different search pipelines."""
    print("🔧 RAG Metrics Tutorial - Search Pipeline Setup")
    print("=" * 60)

    # Initialize pipeline manager
    manager = SearchPipelineManager()

    try:
        # Show pipeline information
        print("\n📋 Available Search Pipelines:")
        pipeline_info = manager.get_pipeline_info()
        for name, info in pipeline_info.items():
            print(f"\n{info['name']}:")
            print(f"  Description: {info['description']}")
            print(f"  Features: {', '.join(info['features'])}")

        # Test pipelines
        print(f"\n{'-' * 60}")
        manager.test_pipelines()

        print(f"\n{'-' * 60}")
        print("✅ Pipeline setup complete!")
        print("\n📋 Next steps:")
        print("1. Pipelines are ready for evaluation")
        print("2. Run 3-run-experiments.py to execute evaluation")
        print("3. Each pipeline will be tested against the evaluation dataset")

    except Exception as e:
        print(f"❌ Error demonstrating pipelines: {e}")
        raise
    finally:
        manager.close_all()


def main():
    """Main function to set up and demonstrate search pipelines."""
    demonstrate_pipelines()


if __name__ == "__main__":
    main()
