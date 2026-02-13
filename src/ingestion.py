import os
import csv
import re
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ============================================================
#   EARTHQUAKE DATA LOADER (TXT with pipe delimiter)
# ============================================================

class EarthquakeLoader:


    def load_txt(self, file_path: str) -> List[Document]:


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
#   TEXT CLEANER
# ============================================================

class TextCleaner:
    """
    Utility class for cleaning raw text:
    - removes excessive whitespace
    - collapses repeated line breaks
    """

    def clean(self, text: str) -> str:


        if not text:
            return ""

        text = re.sub(r"[ \t]+", " ", text)   # collapse tabs/spaces
        text = re.sub(r"\n+", "\n", text)     # collapse newlines
        return text.strip()


# ============================================================
#   TEXT SPLITTER
# ============================================================

class TextSplitter:


    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split_text(self, text: str) -> List[str]:
        """Split a raw string into smaller chunks."""
        return self.splitter.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split Document objects into multiple chunked Document objects."""
        return self.splitter.split_documents(documents)