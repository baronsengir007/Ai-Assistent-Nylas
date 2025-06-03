"""
RAG Metrics Tutorial - Step 1: Data Creation

This module loads the SQuAD 2.0 dataset, processes documents into chunks,
and creates evaluation datasets with known relevant document mappings.
This is crucial for measuring retrieval quality since we know which documents
should be retrieved for each question.

Key Features:
- Loads SQuAD 2.0 dataset (10K examples)
- Chunks documents and stores in pgvector database
- Creates synthetic questions for each document
- Generates evaluation CSV files with known ground truth
- Uses existing pgvector infrastructure
"""

import sys
from pathlib import Path

# Add the pgvector-setup to the path so we can import from it
sys.path.append(str(Path(__file__).parent.parent) + "/pgvector-setup")

import json
import os
import pandas as pd
from typing import Dict, List
from datasets import load_dataset
import psycopg
from pgvector.psycopg import register_vector
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()

# Database and API configuration
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/postgres"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class RAGDatasetCreator:
    """Creates evaluation datasets for RAG metrics using SQuAD 2.0."""

    def __init__(self):
        """Initialize the dataset creator."""
        self.conn = psycopg.connect(DATABASE_URL)
        register_vector(self.conn)
        self.embedding_dimensions = 1536

    def clear_database(self):
        """Clear existing documents from the database."""
        print("🧹 Clearing existing documents...")
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM documents")
            self.conn.commit()
        print("✅ Database cleared")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI."""
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=self.embedding_dimensions,
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return [0.0] * self.embedding_dimensions

    def chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[str]:
        """Chunk text into smaller pieces with overlap."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            chunk = " ".join(chunk_words)

            # Only add chunks with meaningful content
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())

        return chunks

    def load_squad_dataset(self, num_examples: int = 1000) -> List[Dict]:
        """Load and process SQuAD 2.0 dataset."""
        print(f"📚 Loading SQuAD 2.0 dataset ({num_examples} examples)...")

        # Load dataset
        dataset = load_dataset("squad_v2", split="validation")

        # Process examples
        processed_examples = []
        for i, example in enumerate(dataset):
            if i >= num_examples:
                break

            # Skip examples without answers (SQuAD 2.0 has unanswerable questions)
            if not example["answers"]["text"]:
                continue

            processed_examples.append(
                {
                    "id": example["id"],
                    "context": example["context"],
                    "question": example["question"],
                    "answers": example["answers"]["text"],
                    "title": example.get("title", "Unknown"),
                }
            )

        print(f"✅ Loaded {len(processed_examples)} SQuAD examples with answers")
        return processed_examples

    def store_documents_with_mapping(
        self, squad_examples: List[Dict]
    ) -> Dict[str, int]:
        """Store SQuAD contexts as documents and return question->doc_id mapping."""
        print("💾 Storing documents in vector database...")

        question_to_doc_id = {}
        doc_id_to_questions = {}
        stored_contexts = set()  # Avoid duplicate contexts

        for i, example in enumerate(squad_examples):
            if i % 100 == 0:
                print(f"  Processing {i}/{len(squad_examples)} examples...")

            context = example["context"]
            question = example["question"]

            # Check if we've already stored this context
            context_hash = hash(context)
            if context_hash in stored_contexts:
                # Find the existing doc_id
                with self.conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM documents WHERE content = %s LIMIT 1",
                        (context,),
                    )
                    result = cur.fetchone()
                    if result:
                        doc_id = result[0]
                        question_to_doc_id[question] = doc_id
                        if doc_id not in doc_id_to_questions:
                            doc_id_to_questions[doc_id] = []
                        doc_id_to_questions[doc_id].append(question)
                continue

            # Chunk the context
            chunks = self.chunk_text(context)

            # Store each chunk as a separate document
            chunk_doc_ids = []
            for chunk_idx, chunk in enumerate(chunks):
                embedding = self.get_embedding(chunk)

                metadata = {
                    "source": "squad_v2",
                    "example_id": example["id"],
                    "title": example["title"],
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "original_context_length": len(context),
                    "question_count": 1,
                }

                with self.conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO documents (content, metadata, embedding)
                        VALUES (%s, %s, %s)
                        RETURNING id
                        """,
                        (chunk, psycopg.types.json.Json(metadata), embedding),
                    )
                    doc_id = cur.fetchone()[0]
                    chunk_doc_ids.append(doc_id)

            # For simplicity, map the question to the first chunk
            # In a more sophisticated setup, you might map to all relevant chunks
            if chunk_doc_ids:
                question_to_doc_id[question] = chunk_doc_ids[0]
                if chunk_doc_ids[0] not in doc_id_to_questions:
                    doc_id_to_questions[chunk_doc_ids[0]] = []
                doc_id_to_questions[chunk_doc_ids[0]].append(question)

            stored_contexts.add(context_hash)

        self.conn.commit()
        print(f"✅ Stored {len(stored_contexts)} unique contexts as chunked documents")

        # Get total document count
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            total_docs = cur.fetchone()[0]
            print(f"📊 Total documents in database: {total_docs}")

        return question_to_doc_id

    def generate_synthetic_questions(self, num_synthetic: int = 2) -> Dict[str, int]:
        """Generate synthetic questions for stored documents."""
        print(f"🤖 Generating {num_synthetic} synthetic questions per document...")

        # Get all documents
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, metadata 
                FROM documents 
                WHERE metadata->>'source' = 'squad_v2'
                ORDER BY id
                """
            )
            documents = cur.fetchall()

        print(f"📄 Found {len(documents)} documents to generate questions for")

        synthetic_question_to_doc_id = {}

        for i, (doc_id, content, metadata) in enumerate(documents):
            if i % 50 == 0:
                print(f"  Generating questions for document {i}/{len(documents)}...")

            # Generate synthetic questions using OpenAI
            questions = self._generate_questions_for_content(content, num_synthetic)

            for question in questions:
                synthetic_question_to_doc_id[question] = doc_id

            # Small delay to respect rate limits
            time.sleep(0.1)

        print(f"✅ Generated {len(synthetic_question_to_doc_id)} synthetic questions")
        return synthetic_question_to_doc_id

    def _generate_questions_for_content(
        self, content: str, num_questions: int = 2
    ) -> List[str]:
        """Generate questions for a given content using OpenAI."""
        try:
            prompt = f"""Based on the following text, generate {num_questions} diverse questions that can be answered using the information in this text. 

            The questions should:
            - Be specific and answerable from the text
            - Cover different aspects of the content
            - Be natural and realistic
            - Vary in complexity (some simple, some requiring inference)

            Text:
            {content[:1000]}  

            Return only the questions, one per line, without numbering or bullets."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
            )

            questions_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in questions_text.split("\n") if q.strip()]

            # Ensure we don't return more than requested
            return questions[:num_questions]

        except Exception as e:
            print(f"❌ Error generating questions: {e}")
            return []

    def create_evaluation_dataset(
        self, squad_questions: Dict[str, int], synthetic_questions: Dict[str, int]
    ) -> pd.DataFrame:
        """Create the final evaluation dataset."""
        print("📋 Creating evaluation dataset...")

        eval_data = []
        query_id = 1

        # Add SQuAD questions (real questions)
        for question, doc_id in squad_questions.items():
            eval_data.append(
                {
                    "query_id": query_id,
                    "question": question,
                    "query_type": "real",
                    "expected_doc_id": doc_id,
                    "source": "squad_v2",
                    "generated_by": "human",
                }
            )
            query_id += 1

        # Add synthetic questions
        for question, doc_id in synthetic_questions.items():
            eval_data.append(
                {
                    "query_id": query_id,
                    "question": question,
                    "query_type": "synthetic",
                    "expected_doc_id": doc_id,
                    "source": "generated",
                    "generated_by": "gpt-4o-mini",
                }
            )
            query_id += 1

        df = pd.DataFrame(eval_data)

        # Add some additional context info
        doc_contents = {}
        with self.conn.cursor() as cur:
            for doc_id in df["expected_doc_id"].unique():
                cur.execute("SELECT content FROM documents WHERE id = %s", (doc_id,))
                result = cur.fetchone()
                if result:
                    doc_contents[doc_id] = (
                        result[0][:200] + "..." if len(result[0]) > 200 else result[0]
                    )

        df["context_snippet"] = df["expected_doc_id"].map(doc_contents)

        print(f"✅ Created evaluation dataset with {len(df)} questions")
        print(f"   - Real questions: {len(df[df['query_type'] == 'real'])}")
        print(f"   - Synthetic questions: {len(df[df['query_type'] == 'synthetic'])}")

        return df

    def save_evaluation_dataset(self, df: pd.DataFrame, output_dir: str = "data"):
        """Save the evaluation dataset to CSV files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save main evaluation dataset
        eval_path = os.path.join(output_dir, "eval_queries.csv")
        df.to_csv(eval_path, index=False)
        print(f"💾 Saved evaluation dataset to {eval_path}")

        # Save summary statistics
        stats = {
            "total_questions": len(df),
            "real_questions": len(df[df["query_type"] == "real"]),
            "synthetic_questions": len(df[df["query_type"] == "synthetic"]),
            "unique_documents": df["expected_doc_id"].nunique(),
            "average_question_length": df["question"].str.len().mean(),
            "creation_date": pd.Timestamp.now().isoformat(),
        }

        stats_path = os.path.join(output_dir, "dataset_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"📊 Saved dataset statistics to {stats_path}")

        return eval_path

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Main function to create the RAG evaluation dataset."""
    print("🚀 RAG Metrics Tutorial - Creating Evaluation Dataset")
    print("=" * 60)

    # Initialize dataset creator
    creator = RAGDatasetCreator()

    try:
        # Step 1: Clear existing data
        creator.clear_database()

        # Step 2: Load SQuAD dataset
        squad_examples = creator.load_squad_dataset(num_examples=1000)

        # Step 3: Store documents and create question mappings
        squad_question_mapping = creator.store_documents_with_mapping(squad_examples)

        # Step 4: Generate synthetic questions
        synthetic_question_mapping = creator.generate_synthetic_questions(
            num_synthetic=2
        )

        # Step 5: Create evaluation dataset
        eval_df = creator.create_evaluation_dataset(
            squad_question_mapping, synthetic_question_mapping
        )

        # Step 6: Save to files
        eval_path = creator.save_evaluation_dataset(eval_df)

        print("\n🎉 Dataset creation complete!")
        print(f"📁 Evaluation dataset saved to: {eval_path}")
        print("\n📋 Next steps:")
        print("1. Run 2-setup-pipelines.py to configure search methods")
        print("2. Run 3-run-experiments.py to execute evaluation")
        print("3. Run 4-analyze-results.py to generate visualizations")

    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        raise
    finally:
        creator.close()


if __name__ == "__main__":
    main()
