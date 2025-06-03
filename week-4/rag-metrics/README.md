# RAG Metrics Tutorial: Comprehensive Retrieval Evaluation

This tutorial provides a complete framework for evaluating and comparing different RAG (Retrieval-Augmented Generation) retrieval methods using standardized metrics and real evaluation datasets. It demonstrates how to measure the effectiveness of semantic search, hybrid search, and hybrid search with reranking using the SQuAD 2.0 dataset.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Tutorial Structure](#tutorial-structure)
4. [Dataset and Methodology](#dataset-and-methodology)
5. [Metrics Explained](#metrics-explained)
6. [Search Methods Compared](#search-methods-compared)
7. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
8. [Understanding the Results](#understanding-the-results)
9. [Files and Outputs](#files-and-outputs)
10. [Customization and Extension](#customization-and-extension)
11. [Troubleshooting](#troubleshooting)

## 🎯 Overview

This tutorial creates a comprehensive evaluation framework for RAG retrieval systems by:

- **Loading real-world data**: Using SQuAD 2.0 dataset with known question-document pairs
- **Generating synthetic data**: Creating additional questions to expand the evaluation dataset
- **Implementing multiple search methods**: Semantic, hybrid, and hybrid+reranking approaches
- **Calculating standardized metrics**: Precision@K, Recall, MRR, and execution time
- **Generating visualizations**: Charts and dashboards for comparing method performance
- **Providing actionable insights**: Reports identifying strengths and weaknesses of each approach

### Key Benefits

- **Objective Comparison**: Compare different retrieval methods using standardized metrics
- **Known Ground Truth**: Use SQuAD 2.0 questions where we know the relevant documents
- **Production-Ready Code**: Modular, extensible implementation suitable for real-world use
- **Comprehensive Analysis**: Multiple metrics and visualizations for thorough evaluation
- **Educational Value**: Clear explanations of metrics and methodologies

## 🔧 Prerequisites

### System Requirements

- Python 3.9+
- PostgreSQL with pgvector extension (from week-4/pgvector-setup)
- Sufficient memory for embeddings and visualizations

### Required Dependencies

```bash
uv add install datasets pandas numpy matplotlib seaborn openai cohere psycopg[binary] pgvector python-dotenv
```

### API Keys Required

- **OpenAI API Key**: For generating embeddings and synthetic questions
- **Cohere API Key**: For reranking functionality

Add these to your `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
```

### Database Setup

Ensure your PostgreSQL database with pgvector is running and accessible at:
- Host: `localhost`
- Port: `5432`
- Database: `postgres`
- Username: `postgres`
- Password: `postgres`

The database should have the `documents` table created from the pgvector-setup tutorial.

## 📁 Tutorial Structure

This tutorial follows a sequential structure with four main steps:

```
week-4/rag-metrics/
├── 1-create-data.py          # Data loading and evaluation dataset creation
├── 2-setup-pipelines.py      # Search pipeline implementation and testing
├── 3-run-experiments.py      # Comprehensive evaluation execution
├── 4-analyze-results.py      # Visualization and analysis generation
├── search_pipelines.py       # Reusable search pipeline classes
├── data/                     # Generated datasets and results
│   ├── eval_queries.csv      # Evaluation dataset with known answers
│   ├── dataset_stats.json    # Dataset statistics
│   ├── retrieval_results_*.csv    # Detailed evaluation results
│   └── aggregate_metrics_*.csv    # Summary metrics by method
├── plots/                    # Generated visualizations
│   ├── precision_at_k_comparison.png
│   ├── overall_metrics_comparison.png
│   ├── performance_vs_speed.png
│   ├── query_type_breakdown.png
│   ├── ranking_distribution.png
│   ├── comprehensive_dashboard.png
│   └── insights_report.txt
└── README.md                 # This documentation
```

## 📊 Dataset and Methodology

### SQuAD 2.0 Dataset

We use the Stanford Question Answering Dataset (SQuAD) 2.0 because:

- **Known Relevance**: Each question has a known relevant context (document)
- **High Quality**: Human-created questions with verified answers
- **Diverse Content**: Covers multiple topics and question types
- **Realistic**: Represents real-world question-answering scenarios

### Data Processing Pipeline

1. **Document Chunking**: Break SQuAD contexts into overlapping chunks (500 words, 50-word overlap)
2. **Embedding Generation**: Create 1536-dimensional embeddings using OpenAI's text-embedding-3-small
3. **Storage**: Store chunks in PostgreSQL with pgvector for similarity search
4. **Ground Truth Mapping**: Maintain question → relevant document ID mappings

### Synthetic Data Generation

To expand our evaluation dataset, we:

- Generate 2 additional questions per document using GPT-3.5-turbo
- Ensure questions are answerable from the document content
- Maintain the question → document mapping for evaluation

### Evaluation Methodology

- **Binary Relevance**: Each question has one known relevant document
- **Multiple K Values**: Test precision at K=1, 3, 5, 10
- **Execution Time**: Measure query latency for performance comparison
- **Query Types**: Separate analysis for real vs synthetic questions

## 📈 Metrics Explained

### Precision@K
**Definition**: Fraction of retrieved documents (in top K) that are relevant.

```
Precision@K = (Relevant documents in top K) / K
```

**Interpretation**:
- Higher is better (max = 1.0)
- Shows how precise the method is at different retrieval depths
- Critical for user experience (users typically look at top few results)

**Example**: If we retrieve 5 documents and 2 are relevant, Precision@5 = 2/5 = 0.4

### Recall
**Definition**: Fraction of all relevant documents that were retrieved.

```
Recall = (Relevant documents retrieved) / (Total relevant documents)
```

**Interpretation**:
- Higher is better (max = 1.0)
- In our setup, each query has 1 relevant document, so Recall is binary (0 or 1)
- Shows completeness of retrieval

### Mean Reciprocal Rank (MRR)
**Definition**: Average of reciprocal ranks of the first relevant result.

```
MRR = (1/N) × Σ(1/rank_of_first_relevant_result)
```

**Interpretation**:
- Higher is better (max = 1.0)
- Emphasizes finding relevant results in top positions
- MRR = 1.0 means relevant document is always rank 1
- MRR = 0.5 means relevant document is on average at rank 2

**Example**: If relevant documents are at ranks [1, 3, 2], MRR = (1/1 + 1/3 + 1/2) / 3 = 0.61

### Success Rate
**Definition**: Percentage of queries where at least one relevant document was found.

```
Success Rate = (Queries with relevant results) / (Total queries)
```

### Execution Time
**Definition**: Average time (in milliseconds) to execute a search query.

**Trade-offs**: Often there's a trade-off between search quality and speed.

## 🔍 Search Methods Compared

### 1. Semantic Search
**Method**: Pure vector similarity using cosine distance on embeddings.

**Strengths**:
- Understands semantic meaning and context
- Handles synonyms and paraphrasing well
- Fast once embeddings are computed
- Language-agnostic

**Weaknesses**:
- May miss exact keyword matches
- Can be too broad for specific queries
- Sensitive to embedding quality

**Implementation**: Uses pgvector's cosine similarity (`<=>` operator) on OpenAI embeddings.

### 2. Hybrid Search
**Method**: Combines semantic and keyword search using Reciprocal Rank Fusion (RRF).

**Components**:
- **Semantic Branch**: Vector similarity search
- **Keyword Branch**: PostgreSQL full-text search with tsvector
- **Fusion**: RRF algorithm combines results with configurable weights

**Strengths**:
- Balances semantic understanding with keyword precision
- Better coverage (finds results that either method alone might miss)
- Tunable weights for different use cases
- Handles both conceptual and specific queries

**Implementation**: 
- Flexible keyword matching (OR logic for better recall)
- RRF with K=50 smoothing parameter
- Equal weights (1.0) for semantic and keyword components

### 3. Hybrid + Reranking
**Method**: Hybrid search followed by Cohere's neural reranking model.

**Pipeline**:
1. Retrieve 3x more documents using hybrid search
2. Rerank using Cohere's rerank-english-v3.0 model
3. Return top K results from reranked list

**Strengths**:
- Highest precision through neural reranking
- Combines retrieval efficiency with reranking accuracy
- State-of-the-art relevance modeling
- Handles complex query-document relationships

**Weaknesses**:
- Higher latency due to API calls
- Additional cost for reranking API
- Dependent on external service

## 🚀 Step-by-Step Execution Guide

### Step 1: Create Evaluation Dataset

```bash
cd week-4/rag-metrics
python 1-create-data.py
```

**What it does**:
- Clears existing documents from the database
- Loads 1,000 SQuAD 2.0 examples
- Chunks contexts and stores in vector database
- Generates 2 synthetic questions per document
- Creates `data/eval_queries.csv` with question-document mappings
- Saves dataset statistics

**Expected output**:
- ~2,000+ documents in the database
- ~3,000+ evaluation questions (1,000 real + 2,000+ synthetic)
- CSV file with ground truth mappings

**Time**: ~10-15 minutes (depending on API rate limits)

### Step 2: Setup and Test Search Pipelines

```bash
python 2-setup-pipelines.py
```

**What it does**:
- Initializes all three search pipelines
- Tests each pipeline with sample queries
- Validates database connections and API access
- Shows example results from each method

**Expected output**:
- Confirmation that all pipelines are working
- Sample results showing different scoring approaches
- Performance baseline measurements

**Time**: ~2-3 minutes

### Step 3: Run Comprehensive Evaluation

```bash
python 3-run-experiments.py
```

**What it does**:
- Loads the evaluation dataset
- Runs every query against all three search methods
- Calculates all metrics (Precision@K, Recall, MRR, etc.)
- Saves detailed and aggregate results to CSV files
- Provides summary statistics

**Expected output**:
- `data/retrieval_results_latest.csv`: Detailed results for every query-method combination
- `data/aggregate_metrics_latest.csv`: Summary metrics for each method
- Console summary showing method rankings

**Time**: ~15-30 minutes (depending on dataset size and API limits)

### Step 4: Generate Analysis and Visualizations

```bash
python 4-analyze-results.py
```

**What it does**:
- Loads evaluation results
- Generates comprehensive visualizations
- Creates performance comparison charts
- Produces insights report with recommendations

**Expected output**:
- 6 visualization files in `plots/` directory
- `plots/insights_report.txt` with detailed analysis
- Comprehensive dashboard showing all metrics

**Time**: ~2-3 minutes

## 📊 Understanding the Results

### Key Visualizations

#### 1. Precision@K Comparison
- **Bar chart** comparing all methods across K=1,3,5,10
- **Look for**: Which method maintains precision as K increases
- **Ideal pattern**: Minimal degradation from P@1 to P@10

#### 2. Overall Metrics Comparison
- **Four-panel** showing MRR, Recall, Success Rate, and Execution Time
- **Look for**: Best overall performer and speed vs quality trade-offs
- **Key insight**: Identify the best method for your specific priorities

#### 3. Performance vs Speed Scatter Plot
- **X-axis**: Execution time (lower is better)
- **Y-axis**: MRR (higher is better)
- **Ideal position**: Top-left (high quality, fast execution)

#### 4. Query Type Breakdown
- **Compares** real vs synthetic question performance
- **Look for**: Consistency across question types
- **Potential insight**: Some methods may work better on specific question styles

#### 5. Ranking Distribution
- **Histograms** showing where relevant documents are typically found
- **Look for**: More results at rank 1 (better method)
- **Red line**: Shows mean rank for each method

#### 6. Comprehensive Dashboard
- **All-in-one** view with heatmaps, rankings, and radar charts
- **Use for**: Executive summaries and method selection

### Interpreting Results

#### Expected Performance Ranking
Based on typical patterns, you might see:

1. **Hybrid + Reranking**: Highest precision and MRR, slowest execution
2. **Hybrid Search**: Good balance of precision and speed
3. **Semantic Search**: Fastest, but potentially lower precision

#### Key Questions to Ask

1. **Is the quality improvement worth the speed cost?**
   - Compare MRR gain vs execution time increase
   - Consider your application's latency requirements

2. **How much does precision degrade with K?**
   - Steep degradation suggests the method isn't finding relevant results consistently
   - Flat degradation indicates robust performance

3. **Are there differences between real and synthetic questions?**
   - Large differences might indicate dataset bias
   - Consistent performance suggests robust methods

4. **What's the success rate?**
   - Low success rates indicate the method often fails to find relevant documents
   - High success rates with low MRR suggest relevant documents are found but at low ranks

### Making Decisions Based on Results

#### Choose Semantic Search When:
- Speed is critical
- Queries are conceptual/semantic in nature
- You have high-quality embeddings
- Cost optimization is important

#### Choose Hybrid Search When:
- You need balanced performance
- Queries mix conceptual and specific elements
- You want good performance without external dependencies
- You need explainable results

#### Choose Hybrid + Reranking When:
- Precision is paramount
- Latency requirements are relaxed
- You have budget for reranking API calls
- You're building a premium search experience

## 📁 Files and Outputs

### Input Files Required
- **Environment variables**: OpenAI and Cohere API keys
- **Database**: PostgreSQL with pgvector and documents table

### Generated Data Files

#### `data/eval_queries.csv`
Contains the evaluation dataset with columns:
- `query_id`: Unique identifier for each question
- `question`: The actual question text
- `query_type`: 'real' (from SQuAD) or 'synthetic' (generated)
- `expected_doc_id`: ID of the relevant document
- `source`: 'squad_v2' or 'generated'
- `generated_by`: 'human' or 'gpt-3.5-turbo'
- `context_snippet`: Preview of the relevant document

#### `data/retrieval_results_latest.csv`
Detailed results for every query-method combination:
- Query information (id, type, expected document)
- Method information (name, execution time)
- Retrieved document IDs and ranks
- All calculated metrics (Precision@K, Recall, MRR)
- Success indicators and rankings

#### `data/aggregate_metrics_latest.csv`
Summary metrics aggregated by method:
- Overall performance metrics (MRR, Precision@K, Recall)
- Execution time statistics
- Success rates and rankings
- Query type breakdowns

### Generated Visualizations

All saved as high-resolution PNG files in `plots/`:

1. **`precision_at_k_comparison.png`**: Bar chart comparing Precision@K across methods
2. **`overall_metrics_comparison.png`**: Four-panel metric comparison
3. **`performance_vs_speed.png`**: Scatter plot of quality vs execution time
4. **`query_type_breakdown.png`**: Performance by real vs synthetic questions
5. **`ranking_distribution.png`**: Histograms of relevant document rankings
6. **`comprehensive_dashboard.png`**: All-in-one performance dashboard

### Analysis Report

#### `plots/insights_report.txt`
Text report containing:
- Best overall method identification
- Metric-specific winners (fastest, highest precision, etc.)
- Query type analysis
- Key insights and patterns
- Actionable recommendations

## 🔧 Customization and Extension

### Modifying the Dataset

#### Change Dataset Size
In `1-create-data.py`, modify:
```python
squad_examples = creator.load_squad_dataset(num_examples=2000)  # Increase from 1000
```

#### Adjust Synthetic Question Generation
```python
synthetic_question_mapping = creator.generate_synthetic_questions(num_synthetic=3)  # Increase from 2
```

#### Use Different Datasets
Replace SQuAD loading with your own dataset:
```python
def load_custom_dataset(self):
    # Load your dataset with 'context' and 'question' fields
    pass
```

### Modifying Search Methods

#### Add New Search Pipeline
Create a new class in `search_pipelines.py`:
```python
class CustomSearchPipeline(SearchPipeline):
    def __init__(self):
        super().__init__("custom_search")
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        # Implement your search logic
        pass
```

#### Adjust Hybrid Search Weights
In `HybridSearchPipeline.search()`:
```python
semantic_weight=2.0,    # Favor semantic results
keyword_weight=1.0
```

#### Change Reranking Model
In `HybridRerankingPipeline`:
```python
self.rerank_model = "rerank-multilingual-v2.0"  # Different Cohere model
```

### Modifying Metrics

#### Add Custom Metrics
In `3-run-experiments.py`, extend `evaluate_single_query()`:
```python
# Add NDCG calculation
metrics['ndcg'] = self.calculate_ndcg(retrieved_doc_ids, relevant_doc_ids)
```

#### Change K Values
```python
self.k_values = [1, 5, 10, 20]  # Different K values for Precision@K
```

### Visualization Customization

#### Modify Plot Styles
In `4-analyze-results.py`:
```python
plt.style.use('seaborn')  # Different style
sns.set_palette("Set2")   # Different color palette
```

#### Add New Visualizations
Create new methods in `RAGResultsAnalyzer`:
```python
def create_custom_analysis(self, data):
    # Your custom visualization
    pass
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### Database Connection Errors
```
Error: could not connect to PostgreSQL server
```
**Solution**: 
- Ensure PostgreSQL is running: `docker compose up -d` in pgvector-setup/docker
- Check connection parameters in `DATABASE_URL`
- Verify pgvector extension is installed

#### API Key Issues
```
Error: Invalid API key provided
```
**Solution**:
- Check `.env` file exists with correct keys
- Verify API keys are active and have sufficient credits
- Test keys independently

#### Memory Issues
```
Error: Out of memory
```
**Solution**:
- Reduce dataset size in `1-create-data.py`
- Use sampling in evaluation: `run_evaluation(eval_df, sample_size=500)`
- Close browser tabs and other memory-intensive applications

#### Slow Execution
**Solutions**:
- Reduce dataset size for testing
- Use fewer synthetic questions per document
- Sample queries for faster iteration:
```python
results_df = evaluator.run_evaluation(eval_df, sample_size=100)
```

#### Missing Dependencies
```
ImportError: No module named 'datasets'
```
**Solution**:
```bash
pip install datasets pandas numpy matplotlib seaborn openai cohere psycopg[binary] pgvector python-dotenv
```

#### Cohere Rate Limits
```
Error: Rate limit exceeded
```
**Solution**:
- Add delays between reranking calls
- Reduce the number of documents sent for reranking
- Use a higher-tier Cohere plan

### Data Quality Issues

#### Low Success Rates
If success rates are very low (<20%):
- Check that embeddings are being generated correctly
- Verify document chunking isn't too aggressive
- Ensure query-document mappings are correct

#### Identical Scores Across Methods
If all methods show similar performance:
- Check that different search methods are actually implemented differently
- Verify that the evaluation dataset has sufficient variety
- Ensure metrics are being calculated correctly

#### Unrealistic Execution Times
If execution times are extremely high or low:
- Check that database indexes are present
- Verify network connectivity for API calls
- Consider caching embeddings for repeated evaluations

### Debugging Tips

#### Enable Verbose Logging
Add debug prints to understand execution flow:
```python
print(f"Query: {query}")
print(f"Retrieved doc IDs: {retrieved_doc_ids}")
print(f"Expected doc ID: {expected_doc_id}")
```

#### Check Intermediate Results
Save intermediate outputs for analysis:
```python
# In run_evaluation()
if completed % 10 == 0:  # Save every 10 queries
    pd.DataFrame(all_results).to_csv(f"debug_results_{completed}.csv")
```

#### Validate Data Quality
Check that your evaluation dataset makes sense:
```python
# Check question-document mapping quality
eval_df.groupby('query_type')['question'].count()
eval_df['question'].str.len().describe()
```

## 🎯 Conclusion

This tutorial provides a comprehensive framework for evaluating RAG retrieval systems using real-world data and standardized metrics. The modular design allows for easy extension and customization while providing immediate insights into the relative performance of different search approaches.

### Key Takeaways

1. **Evaluation is Critical**: Systematic evaluation reveals performance differences that aren't apparent from casual testing
2. **Multiple Metrics Matter**: Different metrics highlight different aspects of search quality
3. **Context-Dependent Choices**: The best method depends on your specific use case and requirements
4. **Trade-offs Exist**: There are usually trade-offs between quality, speed, and cost
5. **Real Data is Essential**: Using datasets with known ground truth provides objective evaluation

### Next Steps

After completing this tutorial, consider:

1. **Testing with Your Own Data**: Apply this framework to your specific domain and use case
2. **Exploring Additional Methods**: Implement other retrieval techniques (e.g., BM25, dense passage retrieval)
3. **Optimizing Parameters**: Fine-tune weights, thresholds, and other parameters
4. **Scaling Up**: Test with larger datasets and more diverse queries
5. **Production Integration**: Adapt the evaluation framework for continuous monitoring

### Contributing

This tutorial is designed to be educational and extensible. Feel free to:
- Add new search methods
- Implement additional metrics
- Create new visualizations
- Improve documentation
- Share your findings and improvements

Remember: good evaluation leads to better RAG systems, which ultimately provide better user experiences.
