import os
import sys
import csv
import html
from typing import List
from dotenv import load_dotenv

# Extend import path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from langchain.schema import Document
from src.ingestion import TextCleaner, TextSplitter
from src.vectorizer import EmbeddingModel, VectorStoreManager
from src.retrieval import Retriever


# ============================================================
# LOAD EARTHQUAKE CSV AS DOCUMENTS
# ============================================================

def load_earthquake_file_as_docs(file_path: str) -> List[Document]:
    """
    Load a CSV containing earthquake events and convert each row into a Document.
    Handles:
    - multiple fallback encodings
    - empty or malformed fields
    - HTML-escaped characters (&amp; -> &)
    """

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "iso-8859-1"]
    last_error = None
    file_handle = None

    # Try multiple encodings until one succeeds
    for enc in encodings_to_try:
        try:
            file_handle = open(file_path, "r", encoding=enc, newline="")
            file_handle.read(2048)
            file_handle.seek(0)
            break
        except UnicodeDecodeError as e:
            last_error = e
            if file_handle:
                file_handle.close()
            file_handle = None

    if not file_handle:
        raise UnicodeDecodeError(
            f"Unable to decode file with any encoding {encodings_to_try}. Last error: {last_error}"
        )

    documents: List[Document] = []

    with file_handle:
        reader = csv.DictReader(file_handle, delimiter=",")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Earthquake file not found: {file_path}")

        documents = []

        # INGV TXT uses "|" as delimiter
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="|")

            for row in reader:
                event_text = (
                    f"Event ID: {row.get('EventID', '')}\n"
                    f"Date/Time: {row.get('Time', '')}\n"
                    f"Latitude: {row.get('Latitude', '')}\n"
                    f"Longitude: {row.get('Longitude', '')}\n"
                    f"Depth (km): {row.get('Depth_Km', '')}\n"
                    f"Magnitude: {row.get('Magnitude', '')} ({row.get('MagType', '')})\n"
                    f"Location: {row.get('EventLocationName', '')}\n"
                    f"Event Type: {row.get('EventType', '')}\n"
                    f"Author: {row.get('Author', '')}\n"
                    f"Catalog: {row.get('Catalog', '')}\n"
                )

                documents.append(
                    Document(
                        page_content=event_text,
                        metadata={
                            "event_id": row.get("EventID", "")
                        }
                    )
                )

    return documents


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    load_dotenv()

    print("\n=== EARTHQUAKE RAG SYSTEM INITIALIZATION ===\n")

    # 1. Load earthquake dataset
    print("[1] Loading earthquake data...")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    earthquakes_file = os.path.join(data_dir, "query.txt")

    if not os.path.exists(earthquakes_file):
        print(f"[ERROR] File not found: {earthquakes_file}")
        return

    cleaner = TextCleaner()
    splitter = TextSplitter(chunk_size=400, chunk_overlap=40)

    print(f"→ Reading file: {earthquakes_file}")
    raw_events = load_earthquake_file_as_docs(earthquakes_file)
    print(f"✓ Loaded {len(raw_events)} earthquake events.")

    print("→ Cleaning text...")
    for doc in raw_events:
        doc.page_content = cleaner.clean(doc.page_content)

    print("→ Splitting into chunks...")
    event_chunks = splitter.split_documents(raw_events)
    print(f"✓ Generated {len(event_chunks)} chunks.\n")

    # 2. Build vector store
    print("[2] Building vector store index...")

    try:
        embedding_model = EmbeddingModel()
        vector_manager = VectorStoreManager(embedding_model)
        vector_manager.create_index(event_chunks)
        print("✓ Vector index successfully built.\n")
    except Exception as e:
        print(f"[ERROR] Failed to build vector index: {e}")
        return

    # 3. Test retrieval
    print("[3] Testing retrieval engine...\n")

    retriever = Retriever(vector_manager)

    sample_queries = [
        "Quali terremoti hanno magnitudo maggiore di 6?",
        "Ci sono eventi profondi nel Tirreno Meridionale?",
        "Terremoti recenti vicino alla Sicilia o Calabria",
    ]

    for query in sample_queries:
        print(f"🔍 Query: {query}")
        results = retriever.retrieve_with_logs(query, k=3)
        print("→ Retrieved results:")
        for log in results["logs"]:
            print(f"  [{log['rank']}] Score: {log['score']:.4f} ")
            print(f"      Snippet: {log['content_snippet']}\n")

    print("\n=== SYSTEM READY ===\n")


if __name__ == "__main__":
    main()