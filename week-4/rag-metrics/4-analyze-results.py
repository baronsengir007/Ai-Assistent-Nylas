"""
RAG Metrics Tutorial - Step 4: Analyze Results and Generate Visualizations

This module analyzes the evaluation results from step 3 and generates
comprehensive visualizations to understand the performance of different
search methods. It creates charts comparing metrics across methods
and provides insights into the effectiveness of each approach.

Key Visualizations:
- Precision@K comparison across methods
- MRR and Recall comparison
- Execution time vs Quality trade-offs
- Query type performance breakdown
- Method ranking dashboard
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from pathlib import Path

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")


class RAGResultsAnalyzer:
    """Analyzes RAG evaluation results and generates visualizations."""

    def __init__(self, data_dir: str = "data", output_dir: str = "plots"):
        """Initialize the analyzer."""
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Configure matplotlib for better plots
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 11
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["legend.fontsize"] = 11

    def load_results(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load evaluation results from CSV files."""
        # Try latest files first, then timestamped files
        aggregate_path = os.path.join(self.data_dir, "aggregate_metrics_latest.csv")
        detailed_path = os.path.join(self.data_dir, "retrieval_results_latest.csv")

        if not os.path.exists(aggregate_path):
            # Try to find timestamped files
            import glob

            aggregate_files = glob.glob(
                os.path.join(self.data_dir, "aggregate_metrics_*.csv")
            )
            detailed_files = glob.glob(
                os.path.join(self.data_dir, "retrieval_results_*.csv")
            )

            if aggregate_files and detailed_files:
                aggregate_path = max(aggregate_files)  # Get most recent
                detailed_path = max(detailed_files)
            else:
                raise FileNotFoundError(
                    f"No evaluation results found in {self.data_dir}. "
                    "Please run 3-run-experiments.py first."
                )

        aggregate_df = pd.read_csv(aggregate_path)
        detailed_df = pd.read_csv(detailed_path)

        print("📊 Loaded results:")
        print(f"   - Aggregate metrics: {len(aggregate_df)} methods")
        print(f"   - Detailed results: {len(detailed_df)} query-method combinations")

        return aggregate_df, detailed_df

    def create_precision_at_k_comparison(self, aggregate_df: pd.DataFrame) -> str:
        """Create Precision@K comparison chart."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data for plotting
        k_values = [1, 3, 5, 10]
        x = np.arange(len(k_values))
        width = 0.25

        methods = aggregate_df["method_name"].tolist()
        colors = sns.color_palette("husl", len(methods))

        for i, method in enumerate(methods):
            row = aggregate_df[aggregate_df["method_name"] == method].iloc[0]
            values = [row[f"precision_at_{k}"] for k in k_values]

            ax.bar(
                x + i * width, values, width, label=method, color=colors[i], alpha=0.8
            )

        ax.set_xlabel("K (Number of Retrieved Documents)")
        ax.set_ylabel("Precision@K")
        ax.set_title("Precision@K Comparison Across Search Methods")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"@{k}" for k in k_values])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", rotation=90, padding=3)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "precision_at_k_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📈 Created Precision@K comparison: {output_path}")
        return output_path

    def create_overall_metrics_comparison(self, aggregate_df: pd.DataFrame) -> str:
        """Create overall metrics comparison chart."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        methods = aggregate_df["method_name"].tolist()
        colors = sns.color_palette("husl", len(methods))

        # MRR Comparison
        mrr_values = aggregate_df["mrr"].tolist()
        bars1 = ax1.bar(methods, mrr_values, color=colors, alpha=0.8)
        ax1.set_title("Mean Reciprocal Rank (MRR)")
        ax1.set_ylabel("MRR")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)
        for bar, value in zip(bars1, mrr_values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # Recall Comparison
        recall_values = aggregate_df["recall"].tolist()
        bars2 = ax2.bar(methods, recall_values, color=colors, alpha=0.8)
        ax2.set_title("Recall")
        ax2.set_ylabel("Recall")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)
        for bar, value in zip(bars2, recall_values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # Success Rate Comparison
        success_values = aggregate_df["success_rate"].tolist()
        bars3 = ax3.bar(methods, success_values, color=colors, alpha=0.8)
        ax3.set_title("Success Rate (% queries with relevant result found)")
        ax3.set_ylabel("Success Rate")
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(True, alpha=0.3)
        for bar, value in zip(bars3, success_values):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # Execution Time Comparison
        time_values = aggregate_df["avg_execution_time_ms"].tolist()
        bars4 = ax4.bar(methods, time_values, color=colors, alpha=0.8)
        ax4.set_title("Average Execution Time")
        ax4.set_ylabel("Time (ms)")
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(True, alpha=0.3)
        for bar, value in zip(bars4, time_values):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.1f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "overall_metrics_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📈 Created overall metrics comparison: {output_path}")
        return output_path

    def create_performance_vs_speed_scatter(self, aggregate_df: pd.DataFrame) -> str:
        """Create performance vs speed scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8))

        x = aggregate_df["avg_execution_time_ms"]
        y = aggregate_df["mrr"]
        colors = sns.color_palette("husl", len(aggregate_df))

        scatter = ax.scatter(
            x, y, c=colors, s=100, alpha=0.7, edgecolors="black", linewidth=1
        )

        # Add method labels
        for i, method in enumerate(aggregate_df["method_name"]):
            ax.annotate(
                method,
                (x.iloc[i], y.iloc[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                ha="left",
            )

        ax.set_xlabel("Average Execution Time (ms)")
        ax.set_ylabel("Mean Reciprocal Rank (MRR)")
        ax.set_title("Performance vs Speed Trade-off")
        ax.grid(True, alpha=0.3)

        # Add diagonal lines for guidance
        ax.axhline(y=y.mean(), color="red", linestyle="--", alpha=0.5, label="Avg MRR")
        ax.axvline(x=x.mean(), color="red", linestyle="--", alpha=0.5, label="Avg Time")

        ax.legend()
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "performance_vs_speed.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📈 Created performance vs speed plot: {output_path}")
        return output_path

    def create_query_type_breakdown(self, detailed_df: pd.DataFrame) -> str:
        """Create query type performance breakdown."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Calculate metrics by query type and method
        breakdown = (
            detailed_df.groupby(["method_name", "query_type"])
            .agg({"mrr": "mean", "found_relevant": "mean", "precision_at_1": "mean"})
            .reset_index()
        )

        # MRR by query type
        pivot_mrr = breakdown.pivot(
            index="method_name", columns="query_type", values="mrr"
        )
        pivot_mrr.plot(kind="bar", ax=ax1, color=["skyblue", "lightcoral"], alpha=0.8)
        ax1.set_title("MRR by Query Type")
        ax1.set_ylabel("Mean Reciprocal Rank")
        ax1.set_xlabel("Search Method")
        ax1.tick_params(axis="x", rotation=45)
        ax1.legend(title="Query Type")
        ax1.grid(True, alpha=0.3)

        # Success rate by query type
        pivot_success = breakdown.pivot(
            index="method_name", columns="query_type", values="found_relevant"
        )
        pivot_success.plot(
            kind="bar", ax=ax2, color=["skyblue", "lightcoral"], alpha=0.8
        )
        ax2.set_title("Success Rate by Query Type")
        ax2.set_ylabel("Success Rate")
        ax2.set_xlabel("Search Method")
        ax2.tick_params(axis="x", rotation=45)
        ax2.legend(title="Query Type")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "query_type_breakdown.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📈 Created query type breakdown: {output_path}")
        return output_path

    def create_ranking_distribution(self, detailed_df: pd.DataFrame) -> str:
        """Create ranking distribution chart."""
        fig, axes = plt.subplots(
            1, len(detailed_df["method_name"].unique()), figsize=(15, 5)
        )

        if len(detailed_df["method_name"].unique()) == 1:
            axes = [axes]

        methods = detailed_df["method_name"].unique()
        colors = sns.color_palette("husl", len(methods))

        for i, method in enumerate(methods):
            method_data = detailed_df[detailed_df["method_name"] == method]

            # Filter out None values and convert to numeric
            ranks = method_data["relevant_doc_rank"].dropna()

            if len(ranks) > 0:
                axes[i].hist(
                    ranks,
                    bins=range(1, 12),
                    alpha=0.7,
                    color=colors[i],
                    edgecolor="black",
                )
                axes[i].set_title(f"{method}\nRanking Distribution")
                axes[i].set_xlabel("Rank of Relevant Document")
                axes[i].set_ylabel("Frequency")
                axes[i].set_xticks(range(1, 11))
                axes[i].grid(True, alpha=0.3)

                # Add statistics
                mean_rank = ranks.mean()
                axes[i].axvline(
                    mean_rank,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Mean: {mean_rank:.1f}",
                )
                axes[i].legend()
            else:
                axes[i].text(
                    0.5,
                    0.5,
                    "No relevant\ndocuments found",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(f"{method}\nNo Results")

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "ranking_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📈 Created ranking distribution: {output_path}")
        return output_path

    def create_comprehensive_dashboard(self, aggregate_df: pd.DataFrame) -> str:
        """Create a comprehensive dashboard with all key metrics."""
        fig = plt.figure(figsize=(20, 16))

        # Create a grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        methods = aggregate_df["method_name"].tolist()
        colors = sns.color_palette("husl", len(methods))

        # 1. Precision@K Heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        k_values = [1, 3, 5, 10]
        precision_data = []
        for method in methods:
            row = aggregate_df[aggregate_df["method_name"] == method].iloc[0]
            precision_data.append([row[f"precision_at_{k}"] for k in k_values])

        sns.heatmap(
            precision_data,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            xticklabels=[f"P@{k}" for k in k_values],
            yticklabels=methods,
            ax=ax1,
        )
        ax1.set_title("Precision@K Heatmap")

        # 2. MRR Ranking
        ax2 = fig.add_subplot(gs[0, 2])
        mrr_values = aggregate_df["mrr"].tolist()
        bars = ax2.barh(methods, mrr_values, color=colors, alpha=0.8)
        ax2.set_title("MRR Ranking")
        ax2.set_xlabel("MRR")
        for i, (bar, value) in enumerate(zip(bars, mrr_values)):
            ax2.text(
                value + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
            )

        # 3. Success Rate vs Execution Time
        ax3 = fig.add_subplot(gs[1, 0])
        x = aggregate_df["avg_execution_time_ms"]
        y = aggregate_df["success_rate"]
        ax3.scatter(x, y, c=colors, s=100, alpha=0.7, edgecolors="black")
        for i, method in enumerate(methods):
            ax3.annotate(
                method,
                (x.iloc[i], y.iloc[i]),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=9,
            )
        ax3.set_xlabel("Avg Execution Time (ms)")
        ax3.set_ylabel("Success Rate")
        ax3.set_title("Success Rate vs Speed")
        ax3.grid(True, alpha=0.3)

        # 4. Recall Comparison
        ax4 = fig.add_subplot(gs[1, 1])
        recall_values = aggregate_df["recall"].tolist()
        bars = ax4.bar(methods, recall_values, color=colors, alpha=0.8)
        ax4.set_title("Recall Comparison")
        ax4.set_ylabel("Recall")
        ax4.tick_params(axis="x", rotation=45)
        for bar, value in zip(bars, recall_values):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # 5. Method Summary Table
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis("off")

        # Create summary table
        summary_data = []
        for _, row in aggregate_df.iterrows():
            summary_data.append(
                [
                    row["method_name"],
                    f"{row['mrr']:.3f}",
                    f"{row['precision_at_1']:.3f}",
                    f"{row['success_rate']:.3f}",
                    f"{row['avg_execution_time_ms']:.0f}ms",
                ]
            )

        table = ax5.table(
            cellText=summary_data,
            colLabels=["Method", "MRR", "P@1", "Success", "Time"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax5.set_title("Summary Table", pad=20)

        # 6. Performance Radar Chart
        ax6 = fig.add_subplot(gs[2, :], projection="polar")

        # Normalize metrics for radar chart
        metrics_for_radar = [
            "mrr",
            "precision_at_1",
            "precision_at_5",
            "recall",
            "success_rate",
        ]
        angles = np.linspace(
            0, 2 * np.pi, len(metrics_for_radar), endpoint=False
        ).tolist()
        angles += angles[:1]  # Complete the circle

        for i, method in enumerate(methods):
            row = aggregate_df[aggregate_df["method_name"] == method].iloc[0]
            values = [row[metric] for metric in metrics_for_radar]
            values += values[:1]  # Complete the circle

            ax6.plot(angles, values, "o-", linewidth=2, label=method, color=colors[i])
            ax6.fill(angles, values, alpha=0.25, color=colors[i])

        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(["MRR", "P@1", "P@5", "Recall", "Success"])
        ax6.set_ylim(0, 1)
        ax6.set_title("Performance Radar Chart", pad=20)
        ax6.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        # Add main title
        fig.suptitle(
            "RAG Search Methods Comprehensive Evaluation Dashboard",
            fontsize=20,
            fontweight="bold",
            y=0.98,
        )

        output_path = os.path.join(self.output_dir, "comprehensive_dashboard.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Created comprehensive dashboard: {output_path}")
        return output_path

    def generate_insights_report(
        self, aggregate_df: pd.DataFrame, detailed_df: pd.DataFrame
    ) -> str:
        """Generate a text report with insights."""
        report_lines = []
        report_lines.append("# RAG Evaluation Insights Report")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Overall best method
        best_method = aggregate_df.iloc[0]
        report_lines.append(f"## 🏆 Best Overall Method: {best_method['method_name']}")
        report_lines.append(f"- MRR: {best_method['mrr']:.4f}")
        report_lines.append(f"- Precision@1: {best_method['precision_at_1']:.4f}")
        report_lines.append(f"- Success Rate: {best_method['success_rate']:.4f}")
        report_lines.append("")

        # Method comparisons
        report_lines.append("## 📊 Method Comparisons")
        report_lines.append("")

        for metric in ["mrr", "precision_at_1", "recall", "avg_execution_time_ms"]:
            if metric == "avg_execution_time_ms":
                best_idx = aggregate_df[metric].idxmin()
                report_lines.append(
                    f"**Fastest Method ({metric}):** {aggregate_df.loc[best_idx, 'method_name']} "
                    f"({aggregate_df.loc[best_idx, metric]:.1f}ms)"
                )
            else:
                best_idx = aggregate_df[metric].idxmax()
                report_lines.append(
                    f"**Best {metric.upper()}:** {aggregate_df.loc[best_idx, 'method_name']} "
                    f"({aggregate_df.loc[best_idx, metric]:.4f})"
                )

        report_lines.append("")

        # Query type analysis
        report_lines.append("## 🔍 Query Type Analysis")
        report_lines.append("")

        for query_type in ["real", "synthetic"]:
            type_data = detailed_df[detailed_df["query_type"] == query_type]
            if len(type_data) > 0:
                avg_mrr = type_data.groupby("method_name")["mrr"].mean()
                best_method_for_type = avg_mrr.idxmax()
                report_lines.append(f"**{query_type.title()} Questions:**")
                report_lines.append(
                    f"- Best method: {best_method_for_type} (MRR: {avg_mrr.max():.4f})"
                )
                report_lines.append(
                    f"- Total queries: {len(type_data) // len(detailed_df['method_name'].unique())}"
                )
                report_lines.append("")

        # Performance insights
        report_lines.append("## 💡 Key Insights")
        report_lines.append("")

        # Speed vs quality trade-off
        fastest_method = aggregate_df.loc[
            aggregate_df["avg_execution_time_ms"].idxmin()
        ]
        best_quality_method = aggregate_df.loc[aggregate_df["mrr"].idxmax()]

        if fastest_method["method_name"] != best_quality_method["method_name"]:
            report_lines.append(
                f"- **Speed vs Quality Trade-off:** {fastest_method['method_name']} is fastest "
                f"({fastest_method['avg_execution_time_ms']:.1f}ms) but {best_quality_method['method_name']} "
                f"has best quality (MRR: {best_quality_method['mrr']:.4f})"
            )
        else:
            report_lines.append(
                f"- **Best of Both Worlds:** {fastest_method['method_name']} achieves both "
                f"best speed and quality!"
            )

        # Precision degradation
        report_lines.append("")
        for _, row in aggregate_df.iterrows():
            p1 = row["precision_at_1"]
            p10 = row["precision_at_10"]
            degradation = ((p1 - p10) / p1 * 100) if p1 > 0 else 0
            report_lines.append(
                f"- **{row['method_name']} Precision Degradation:** "
                f"{degradation:.1f}% from P@1 to P@10"
            )

        report_lines.append("")
        report_lines.append("## 📋 Recommendations")
        report_lines.append("")

        # Generate recommendations based on data
        if len(aggregate_df) > 1:
            semantic_row = aggregate_df[
                aggregate_df["method_name"].str.contains("semantic", case=False)
            ]
            hybrid_row = aggregate_df[
                aggregate_df["method_name"].str.contains("hybrid", case=False)
            ]

            if not semantic_row.empty and not hybrid_row.empty:
                semantic_mrr = semantic_row.iloc[0]["mrr"]
                hybrid_mrr = hybrid_row.iloc[0]["mrr"]

                if hybrid_mrr > semantic_mrr:
                    improvement = (hybrid_mrr - semantic_mrr) / semantic_mrr * 100
                    report_lines.append(
                        f"- **Hybrid Search Advantage:** Hybrid methods show {improvement:.1f}% "
                        f"improvement over pure semantic search"
                    )

        report_lines.append(
            "- Consider the speed vs quality trade-off based on your use case"
        )
        report_lines.append("- Monitor precision degradation at higher K values")
        report_lines.append("- Test with larger datasets to validate these findings")

        # Save report
        report_path = os.path.join(self.output_dir, "insights_report.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        print(f"📝 Generated insights report: {report_path}")
        return report_path

    def run_complete_analysis(self) -> Dict[str, str]:
        """Run complete analysis and generate all visualizations."""
        print("📊 RAG Metrics Tutorial - Analyzing Results")
        print("=" * 60)

        # Load results
        print("\n📚 Loading evaluation results...")
        aggregate_df, detailed_df = self.load_results()

        # Generate all visualizations
        print("\n🎨 Generating visualizations...")

        generated_files = {}

        generated_files["precision_k"] = self.create_precision_at_k_comparison(
            aggregate_df
        )
        generated_files["overall_metrics"] = self.create_overall_metrics_comparison(
            aggregate_df
        )
        generated_files["performance_speed"] = self.create_performance_vs_speed_scatter(
            aggregate_df
        )
        generated_files["query_breakdown"] = self.create_query_type_breakdown(
            detailed_df
        )
        generated_files["ranking_dist"] = self.create_ranking_distribution(detailed_df)
        generated_files["dashboard"] = self.create_comprehensive_dashboard(aggregate_df)
        generated_files["insights"] = self.generate_insights_report(
            aggregate_df, detailed_df
        )

        print("\n🎉 Analysis complete!")
        print(f"📁 All files saved to: {self.output_dir}")
        print("\n📋 Generated files:")
        for name, path in generated_files.items():
            print(f"   - {name}: {Path(path).name}")

        return generated_files


def main():
    """Main function to run the complete analysis."""
    analyzer = RAGResultsAnalyzer()

    try:
        generated_files = analyzer.run_complete_analysis()

        print("\n🚀 All visualizations and insights are ready!")
        print("👀 Open the files in the 'plots' directory to explore the results.")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
