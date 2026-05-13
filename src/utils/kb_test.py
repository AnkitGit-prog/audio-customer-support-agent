"""
Knowledge Base Test Utility

Run this script to verify that ChromaDB is populated and returning
relevant results for sample customer-support queries.

Usage:
    python src/utils/kb_test.py
"""

import os
import sys
import logging

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv
load_dotenv()

import chromadb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "customer_support_kb"

SAMPLE_QUERIES = [
    "What is your return policy?",
    "How long does shipping take?",
    "What payment methods do you accept?",
]


def print_separator(char: str = "-", width: int = 70) -> None:
    """Print a visual separator line."""
    print(char * width)


def run_test() -> None:
    """Connect to ChromaDB and run sample queries, printing results."""
    print()
    print_separator("=")
    print("  KNOWLEDGE BASE TEST UTILITY")
    print(f"  ChromaDB path : {os.path.abspath(CHROMA_PERSIST_DIR)}")
    print(f"  Collection    : {COLLECTION_NAME}")
    print_separator("=")

    # Connect to ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        print(f"\n[ERROR] Could not connect to ChromaDB: {exc}")
        print(
            "\nMake sure you have started the FastAPI server at least once so "
            "the knowledge base is initialised, then re-run this script."
        )
        sys.exit(1)

    total_docs = collection.count()
    print(f"\n[OK] Connected successfully.")
    print(f"    Total documents in knowledge base: {total_docs}\n")

    if total_docs == 0:
        print("[!] Knowledge base is empty! Start the server to trigger ingestion.")
        sys.exit(1)

    # Run sample queries
    for i, query in enumerate(SAMPLE_QUERIES, start=1):
        print_separator()
        print(f"  Query {i}: \"{query}\"")
        print_separator()

        try:
            results = collection.query(
                query_texts=[query],
                n_results=3,
                include=["documents", "metadatas", "distances"],
            )

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for rank, (doc, meta, dist) in enumerate(
                zip(documents, metadatas, distances), start=1
            ):
                title = meta.get("title", "Unknown")
                similarity = round(1 - dist, 4)
                snippet = doc[:200].replace("\n", " ")
                if len(doc) > 200:
                    snippet += " …"

                print(f"\n  [{rank}] {title}")
                print(f"      Distance Score : {dist:.6f}")
                print(f"      Similarity     : {similarity:.4f}")
                print(f"      Snippet        : {snippet}")

        except Exception as exc:
            print(f"  [X] Query failed: {exc}")

        print()

    print_separator("=")
    print("  TEST COMPLETE")
    print_separator("=")
    print()


if __name__ == "__main__":
    run_test()
