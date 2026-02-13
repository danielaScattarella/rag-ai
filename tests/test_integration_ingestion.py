import pytest
from src.ingestion import EarthquakeLoader, TextCleaner, TextSplitter
from langchain_core.documents import Document


def test_full_earthquake_ingestion_pipeline(tmp_path):
    """
    End‑to‑end ingestion test for EarthquakeLoader → Cleaner → Splitter.
    Ensures that:
      - INGV pipe‑delimited TXT is parsed correctly
      - Text cleaning removes noise
      - Chunk splitting preserves metadata
    """

    # 1. Prepare a sample INGV earthquake TXT
    file_content = (
        "EventID|Time|Latitude|Longitude|Depth_Km|Magnitude|MagType|EventLocationName|EventType\n"
        "12345|2025-02-11T10:20:30.000000|40.123|15.456|12.5|3.2|ML|5 km SE TestCity|earthquake\n"
    )

    file_path = tmp_path / "query.txt"
    file_path.write_text(file_content, encoding="utf-8")

    # 2. Components
    loader = EarthquakeLoader()
    cleaner = TextCleaner()
    splitter = TextSplitter(chunk_size=80, chunk_overlap=10)

    # 3. Step A — Load
    docs = loader.load_txt(str(file_path))
    assert len(docs) == 1
    assert isinstance(docs[0], Document)

    # Verify correct parsing
    assert "Event ID: 12345" in docs[0].page_content
    assert "Latitude: 40.123" in docs[0].page_content
    assert docs[0].metadata["event_id"] == "12345"

    # 4. Step B — Clean text
    cleaned = cleaner.clean(docs[0].page_content)
    docs[0].page_content = cleaned

    # Cleaning validation
    assert "  " not in cleaned
    assert cleaned.strip() != ""

    # 5. Step C — Split into chunks
    chunks = splitter.split_documents(docs)

    assert len(chunks) >= 1

    # Check chunk size and metadata
    for chunk in chunks:
        assert len(chunk.page_content) <= 80
        assert chunk.metadata["event_id"] == "12345"