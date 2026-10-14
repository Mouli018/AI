"""Index the document corpus into Qdrant collections.

Usage:
    python scripts/index_qdrant.py
    python scripts/index_qdrant.py --data data/hotpot_subset.json --n 300
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextrouter.data import build_corpus, load_questions
from contextrouter.qdrant_store import QdrantStore


def main():
    parser = argparse.ArgumentParser(description="Index corpus into Qdrant collection")
    parser.add_argument("--data", default="data/hotpot_subset.json", help="Path to QA dataset")
    parser.add_argument("--n", type=int, default=300, help="Number of questions to index")
    parser.add_argument("--collection", default="hotpotqa_corpus", help="Qdrant collection name")
    args = parser.parse_args()

    print(f"Loading questions from {args.data} (n={args.n})...")
    questions = load_questions(args.data, args.n)
    corpus = build_corpus(questions)
    print(f"Built corpus with {len(corpus)} unique Wikipedia paragraphs.")

    print(f"Initializing Qdrant collection '{args.collection}'...")
    store = QdrantStore(collection_name=args.collection)
    indexed_count = store.index_corpus(corpus)
    print(f"Successfully indexed {indexed_count} documents into Qdrant collection '{args.collection}'!")

    info = store.get_info()
    print("Qdrant Collection Status:", info)


if __name__ == "__main__":
    main()
