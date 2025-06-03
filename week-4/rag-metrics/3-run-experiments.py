"""
RAG Metrics Tutorial - Step 3: Run Evaluation Experiments

This module runs comprehensive evaluation experiments across all search pipelines
and calculates key retrieval metrics. It processes the evaluation dataset created
in step 1 and tests each search method systematically.

Key Metrics Calculated:
- Precision@K (K=1,3,5,10): Fraction of retrieved documents that are relevant
- Recall: Fraction of relevant documents that are retrieved
- MRR (Mean Reciprocal Rank): Average of reciprocal ranks of first relevant result
- Execution Time: Query latency measurement

Results are saved to CSV files for detailed analysis and visualization.
"""

import sys
from pathlib import Path

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

import os
import time
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Import our search pipelines
from search_pipelines import SearchPipelineManager


class RAGMetricsEvaluator:
    """Evaluates RAG retrieval performance across different search methods."""

    def __init__(self, data_dir: str = "data"):
        """Initialize the evaluator."""
        self.data_dir = data_dir
        self.pipeline_manager = SearchPipelineManager()
        self.k_values = [1, 3, 5, 10]

    def load_evaluation_dataset(self) -> pd.DataFrame:
        """Load the evaluation dataset created in step 1."""
        eval_path = os.path.join(self.data_dir, "eval_queries.csv")

        if not os.path.exists(eval_path):
            raise FileNotFoundError(
                f"Evaluation dataset not found at {eval_path}. "
                "Please run 1-create-data.py first."
            )

        df = pd.read_csv(eval_path)
        print(f"📊 Loaded evaluation dataset with {len(df)} queries")
        print(f"   - Real questions: {len(df[df['query_type'] == 'real'])}")
        print(f"   - Synthetic questions: {len(df[df['query_type'] == 'synthetic'])}")

        return df

    def calculate_precision_at_k(
        self, retrieved_doc_ids: List[int], relevant_doc_ids: List[int], k: int
    ) -> float:
        """Calculate Precision@K metric."""
        if k == 0 or not retrieved_doc_ids:
            return 0.0

        top_k_retrieved = retrieved_doc_ids[:k]
        relevant_in_top_k = len(
            [doc_id for doc_id in top_k_retrieved if doc_id in relevant_doc_ids]
        )

        return relevant_in_top_k / len(top_k_retrieved)

    def calculate_recall(
        self, retrieved_doc_ids: List[int], relevant_doc_ids: List[int]
    ) -> float:
        """Calculate Recall metric."""
        if not relevant_doc_ids:
            return 0.0

        relevant_retrieved = len(
            [doc_id for doc_id in retrieved_doc_ids if doc_id in relevant_doc_ids]
        )
        return relevant_retrieved / len(relevant_doc_ids)

    def calculate_mrr(
        self, retrieved_doc_ids: List[int], relevant_doc_ids: List[int]
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR) metric."""
        for rank, doc_id in enumerate(retrieved_doc_ids, 1):
            if doc_id in relevant_doc_ids:
                return 1.0 / rank
        return 0.0

    def evaluate_single_query(
        self, query_row: pd.Series, method_name: str, k: int = 10
    ) -> Dict[str, Any]:
        """Evaluate a single query against a specific search method."""
        query = query_row["question"]
        expected_doc_id = query_row["expected_doc_id"]
        relevant_doc_ids = [
            expected_doc_id
        ]  # For simplicity, we have one relevant doc per query

        # Get the search pipeline
        pipeline = self.pipeline_manager.get_pipeline(method_name)

        # Measure execution time
        start_time = time.time()

        try:
            # Perform search
            results = pipeline.search(query, k=k)
            execution_time_ms = (time.time() - start_time) * 1000

            # Extract document IDs from results
            retrieved_doc_ids = [result["id"] for result in results]

            # Calculate metrics
            metrics = {
                "query_id": query_row["query_id"],
                "method_name": method_name,
                "query_type": query_row["query_type"],
                "retrieved_doc_ids": ",".join(map(str, retrieved_doc_ids)),
                "expected_doc_id": expected_doc_id,
                "execution_time_ms": execution_time_ms,
                "num_results": len(results),
            }

            # Calculate precision@k for different k values
            for k_val in self.k_values:
                metrics[f"precision_at_{k_val}"] = self.calculate_precision_at_k(
                    retrieved_doc_ids, relevant_doc_ids, k_val
                )

            # Calculate recall and MRR
            metrics["recall"] = self.calculate_recall(
                retrieved_doc_ids, relevant_doc_ids
            )
            metrics["mrr"] = self.calculate_mrr(retrieved_doc_ids, relevant_doc_ids)

            # Add ranking information
            if expected_doc_id in retrieved_doc_ids:
                metrics["relevant_doc_rank"] = (
                    retrieved_doc_ids.index(expected_doc_id) + 1
                )
                metrics["found_relevant"] = True
            else:
                metrics["relevant_doc_rank"] = None
                metrics["found_relevant"] = False

            # Add top result information
            if results:
                top_result = results[0]
                metrics["top_result_id"] = top_result["id"]
                metrics["top_result_score"] = self._extract_primary_score(top_result)
            else:
                metrics["top_result_id"] = None
                metrics["top_result_score"] = 0.0

        except Exception as e:
            print(
                f"❌ Error evaluating query {query_row['query_id']} with {method_name}: {e}"
            )
            # Return default metrics for failed queries
            metrics = {
                "query_id": query_row["query_id"],
                "method_name": method_name,
                "query_type": query_row["query_type"],
                "retrieved_doc_ids": "",
                "expected_doc_id": expected_doc_id,
                "execution_time_ms": 0.0,
                "num_results": 0,
                "recall": 0.0,
                "mrr": 0.0,
                "relevant_doc_rank": None,
                "found_relevant": False,
                "top_result_id": None,
                "top_result_score": 0.0,
            }

            # Add zero precision for all k values
            for k_val in self.k_values:
                metrics[f"precision_at_{k_val}"] = 0.0

        return metrics

    def _extract_primary_score(self, result: Dict[str, Any]) -> float:
        """Extract the primary score from a search result."""
        if "rerank_score" in result:
            return result["rerank_score"]
        elif "combined_score" in result:
            return result["combined_score"]
        elif "similarity" in result:
            return result["similarity"]
        else:
            return 0.0

    def run_evaluation(
        self, eval_df: pd.DataFrame, methods: List[str] = None, sample_size: int = None
    ) -> pd.DataFrame:
        """Run evaluation across all methods and queries."""
        if methods is None:
            methods = list(self.pipeline_manager.pipelines.keys())

        # Optionally sample queries for faster evaluation
        if sample_size and len(eval_df) > sample_size:
            print(f"🎯 Sampling {sample_size} queries from {len(eval_df)} total")
            eval_df = eval_df.sample(n=sample_size, random_state=42)

        print(
            f"🚀 Running evaluation on {len(eval_df)} queries with {len(methods)} methods"
        )
        print(f"📊 Methods: {', '.join(methods)}")

        all_results = []
        total_queries = len(eval_df) * len(methods)
        completed = 0

        for method in methods:
            print(f"\n🔍 Evaluating method: {method}")
            print("-" * 40)

            for idx, query_row in eval_df.iterrows():
                if completed % 50 == 0:
                    progress = (completed / total_queries) * 100
                    print(f"  Progress: {completed}/{total_queries} ({progress:.1f}%)")

                # Evaluate single query
                metrics = self.evaluate_single_query(query_row, method, k=10)
                all_results.append(metrics)
                completed += 1

        results_df = pd.DataFrame(all_results)
        print(
            f"\n✅ Evaluation complete! Processed {len(results_df)} query-method combinations"
        )

        return results_df

    def calculate_aggregate_metrics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregate metrics for each method."""
        print("📈 Calculating aggregate metrics...")

        aggregate_metrics = []

        for method in results_df["method_name"].unique():
            method_results = results_df[results_df["method_name"] == method]

            # Calculate means for each metric
            metrics = {
                "method_name": method,
                "total_queries": len(method_results),
                "avg_execution_time_ms": method_results["execution_time_ms"].mean(),
                "avg_num_results": method_results["num_results"].mean(),
                "recall": method_results["recall"].mean(),
                "mrr": method_results["mrr"].mean(),
                "success_rate": method_results["found_relevant"].mean(),
                "avg_relevant_rank": method_results["relevant_doc_rank"].mean(),
            }

            # Add precision@k metrics
            for k in self.k_values:
                metrics[f"precision_at_{k}"] = method_results[
                    f"precision_at_{k}"
                ].mean()

            # Add query type breakdown
            for query_type in ["real", "synthetic"]:
                type_results = method_results[
                    method_results["query_type"] == query_type
                ]
                if len(type_results) > 0:
                    metrics[f"{query_type}_queries"] = len(type_results)
                    metrics[f"{query_type}_success_rate"] = type_results[
                        "found_relevant"
                    ].mean()
                    metrics[f"{query_type}_mrr"] = type_results["mrr"].mean()
                else:
                    metrics[f"{query_type}_queries"] = 0
                    metrics[f"{query_type}_success_rate"] = 0.0
                    metrics[f"{query_type}_mrr"] = 0.0

            aggregate_metrics.append(metrics)

        aggregate_df = pd.DataFrame(aggregate_metrics)

        # Sort by MRR (higher is better)
        aggregate_df = aggregate_df.sort_values("mrr", ascending=False)

        return aggregate_df

    def save_results(
        self, results_df: pd.DataFrame, aggregate_df: pd.DataFrame
    ) -> Tuple[str, str]:
        """Save evaluation results to CSV files."""
        os.makedirs(self.data_dir, exist_ok=True)

        # Add timestamp to filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        detailed_path = os.path.join(
            self.data_dir, f"retrieval_results_{timestamp}.csv"
        )
        results_df.to_csv(detailed_path, index=False)

        # Save aggregate results
        aggregate_path = os.path.join(
            self.data_dir, f"aggregate_metrics_{timestamp}.csv"
        )
        aggregate_df.to_csv(aggregate_path, index=False)

        # Also save without timestamp for easy access
        results_df.to_csv(
            os.path.join(self.data_dir, "retrieval_results_latest.csv"), index=False
        )
        aggregate_df.to_csv(
            os.path.join(self.data_dir, "aggregate_metrics_latest.csv"), index=False
        )

        print(f"💾 Saved detailed results to: {detailed_path}")
        print(f"💾 Saved aggregate metrics to: {aggregate_path}")

        return detailed_path, aggregate_path

    def print_summary(self, aggregate_df: pd.DataFrame):
        """Print a summary of the evaluation results."""
        print("\n📊 EVALUATION SUMMARY")
        print("=" * 60)

        print("\n🏆 Method Rankings (by MRR):")
        for idx, row in aggregate_df.iterrows():
            print(f"{idx + 1}. {row['method_name']}: MRR = {row['mrr']:.4f}")

        print("\n📈 Detailed Metrics:")

        # Create a formatted display
        display_df = aggregate_df.copy()

        # Round numeric columns for better display
        numeric_cols = [
            "avg_execution_time_ms",
            "recall",
            "mrr",
            "success_rate",
            "precision_at_1",
            "precision_at_3",
            "precision_at_5",
            "precision_at_10",
        ]

        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)

        # Select key columns for display
        key_cols = [
            "method_name",
            "precision_at_1",
            "precision_at_3",
            "precision_at_5",
            "recall",
            "mrr",
            "success_rate",
            "avg_execution_time_ms",
        ]

        display_cols = [col for col in key_cols if col in display_df.columns]
        print(display_df[display_cols].to_string(index=False))

        print("\n🔍 Key Insights:")
        best_method = aggregate_df.iloc[0]
        print(
            f"• Best overall method: {best_method['method_name']} (MRR: {best_method['mrr']:.4f})"
        )
        print(
            f"• Best precision@1: {aggregate_df.loc[aggregate_df['precision_at_1'].idxmax(), 'method_name']} "
            f"({aggregate_df['precision_at_1'].max():.4f})"
        )
        print(
            f"• Fastest method: {aggregate_df.loc[aggregate_df['avg_execution_time_ms'].idxmin(), 'method_name']} "
            f"({aggregate_df['avg_execution_time_ms'].min():.1f}ms)"
        )

    def close(self):
        """Close all pipeline connections."""
        self.pipeline_manager.close_all()


def main():
    """Main function to run the evaluation experiments."""
    print("🧪 RAG Metrics Tutorial - Running Evaluation Experiments")
    print("=" * 60)

    evaluator = RAGMetricsEvaluator()

    try:
        # Step 1: Load evaluation dataset
        print("\n📚 Step 1: Loading evaluation dataset...")
        eval_df = evaluator.load_evaluation_dataset()

        # Step 2: Run evaluation
        print("\n🚀 Step 2: Running evaluation experiments...")

        # You can limit the number of queries for faster testing
        # results_df = evaluator.run_evaluation(eval_df, sample_size=100)
        results_df = evaluator.run_evaluation(eval_df)

        # Step 3: Calculate aggregate metrics
        print("\n📊 Step 3: Calculating aggregate metrics...")
        aggregate_df = evaluator.calculate_aggregate_metrics(results_df)

        # Step 4: Save results
        print("\n💾 Step 4: Saving results...")
        detailed_path, aggregate_path = evaluator.save_results(results_df, aggregate_df)

        # Step 5: Print summary
        evaluator.print_summary(aggregate_df)

        print("\n🎉 Evaluation complete!")
        print("\n📋 Next steps:")
        print(f"1. Review detailed results: {detailed_path}")
        print(f"2. Review aggregate metrics: {aggregate_path}")
        print("3. Run 4-analyze-results.py to generate visualizations")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
