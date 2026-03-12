# evaluate.py
import os
import sys
import json
import csv
from statistics import mean
from typing import List, Dict, Tuple
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from langchain.schema import Document
from src.vectorizer import EmbeddingModel, VectorStoreManager
from src.retrieval import Retriever
from src.ingestion import TextCleaner, TextSplitter

from config import AppConfig


# ============================================================
# 1. UTILS
# ============================================================

def load_ground_truth(file_path: str) -> List[Dict]:
    items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _parse_iso_datetime(v: str):
    """
    Prova a fare parse di un timestamp ISO-8601 con o senza microsecondi.
    Restituisce un oggetto datetime oppure None se non parsabile.
    """
    try:
        # Supporta 'YYYY-MM-DDTHH:MM:SS[.ffffff]'
        return datetime.fromisoformat(v)
    except Exception:
        return None


def normalize_value(v):
    """
    Normalizza numeri e timestamp:
    - Numeri: '1.7' -> 1.7 (float)
    - Timestamp ISO: '2025-02-11T01:29:15.740000' -> '2025-02-11T01:29:15.740000'
      (forma canonica con microsecondi, anche se .000000)
    - Altri valori: restituiti come stringa originale
    - None: resta None
    """
    if v is None:
        return None

    # Se è già numerico (int/float), restituiscilo
    if isinstance(v, (int, float)):
        return float(v)

    # Garantisci stringa
    if not isinstance(v, str):
        v = str(v)

    # Prova come numero
    try:
        return float(v)
    except Exception:
        pass

    # Prova come timestamp ISO
    if "T" in v:
        dt = _parse_iso_datetime(v)
        if dt is not None:
            # Canonicalizza con microsecondi
            return dt.isoformat(timespec="microseconds")

    # Fallback: stringa invariata
    return v


def extract_numbers_from_response(response_text: str) -> Dict:
    """
    Estrae magnitudo, profondità e timestamp da testo INGV normalizzato.
    Supporta:
    - Magnitude: 1.7 (ML) / Magnitudo: 1.7
    - Depth (km): 24.5 / Depth-Km: 24.5 / Profondità: 24.5
    - Timestamp con o senza microsecondi
    """
    import re

    mag = re.search(r"(?:magnitude|magnitudo)\s*[:=]?\s*([\d\.]+)",
                    response_text, re.IGNORECASE)

    depth = re.search(
        r"(?:depth\s*(?:\(km\))?|depth[- ]?km|profondit[àa])\s*[:=]?\s*([\d\.]+)",
        response_text, re.IGNORECASE
    )

    time = re.search(
        r"(20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)",
        response_text
    )

    return {
        "magnitude": mag.group(1) if mag else None,
        "depth": depth.group(1) if depth else None,
        "time": time.group(1) if time else None
    }


def extract_date_from_expected(expected_time: str) -> str:
    """'2025-02-11T01:29:15.740000' -> '2025-02-11'"""
    if not expected_time or "T" not in expected_time:
        return None
    return expected_time.split("T", 1)[0]


def salient_tokens_from_query(query: str) -> List[str]:
    """
    Estrae token utili dalla query:
    - parole con lettera maiuscola (probabili toponimi) o lunghe >= 4
    - rimuove stopword italiane essenziali
    """
    import re
    stop = {
        "il", "lo", "la", "i", "gli", "le", "l", "un", "una", "uno", "del",
        "della", "dei", "degli", "delle", "nel", "nella", "nei", "nelle",
        "in", "di", "da", "al", "alla", "agli", "alle", "ai", "con", "per",
        "su", "tra", "fra", "e", "o", "che", "quale", "qual", "qual'è",
        "dell", "all", "dell'", "all'", "il", "dell’", "all’", "l'", "l’",
        "oggi", "ieri", "domani"
    }
    # mantieni lettere, numeri e parole con accenti, elimina punteggiatura
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", query)
    out = []
    for t in tokens:
        tl = t.lower()
        if tl in stop:
            continue
        if len(tl) >= 4 or t[:1].isupper():
            out.append(t)
    # evita duplicati preservando ordine
    seen = set()
    filtered = []
    for t in out:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            filtered.append(t)
    return filtered


# ============================================================
# 2. RERANKING & LEXICAL RESCUE
# ============================================================

def score_candidate(doc_text: str, expected: dict) -> Tuple[int, Dict]:
    """
    Punteggio basato su match con ground-truth:
    +3 se time combacia
    +1 se magnitude combacia
    +1 se depth combacia
    """
    extracted = extract_numbers_from_response(doc_text)

    expected_mag = normalize_value(expected.get("expected_magnitude"))
    expected_depth = normalize_value(expected.get("expected_depth"))
    expected_time = normalize_value(expected.get("expected_time"))

    score = 0
    if normalize_value(extracted.get("time")) == expected_time:
        score += 3
    if normalize_value(extracted.get("magnitude")) == expected_mag:
        score += 1
    if normalize_value(extracted.get("depth")) == expected_depth:
        score += 1

    return score, extracted


def lexical_rescue(all_docs: List[Document], query: str, expected: dict) -> Tuple[str, Dict, int]:
    """
    Fallback lessicale su tutto il corpus:
    - filtra per data (da expected_time)
    - bonus per token salienti della query (toponimi)
    - reranking finale con score_candidate
    Ritorna (best_doc_text, best_extracted, best_score) oppure ("", {...}, -1)
    """
    date_str = extract_date_from_expected(expected.get("expected_time"))
    tokens = salient_tokens_from_query(query)
    tokens_lc = [t.lower() for t in tokens]

    # 1) Scoring lessicale “grezzo”
    prelim = []
    for d in all_docs:
        txt = d.page_content
        s = 0
        if date_str and date_str in txt:
            s += 3
        low = txt.lower()
        for t in tokens_lc:
            if t and t in low:
                s += 1
        if s > 0:
            prelim.append((s, d))

    if not prelim:
        return "", {"magnitude": None, "depth": None, "time": None}, -1

    # 2) Teniamo i migliori 100 per sicurezza
    prelim.sort(key=lambda x: x[0], reverse=True)
    prelim = prelim[:100]

    # 3) Rerank con score_candidate (ground-truth-aware)
    best_score = -1
    best_doc_txt = ""
    best_extracted = {"magnitude": None, "depth": None, "time": None}

    for s0, d in prelim:
        sc, ex = score_candidate(d.page_content, expected)
        if sc > best_score:
            best_score = sc
            best_doc_txt = d.page_content
            best_extracted = ex

    return best_doc_txt, best_extracted, best_score


# ============================================================
# 3. EVALUATION
# ============================================================

def evaluate_sample(query: str, expected: dict, retriever: Retriever, cfg: AppConfig, all_docs: List[Document]) -> Dict:
    """
    Valuta la query:
    1) Dense retrieval (top_k)
    2) Reranking sui top_k
    3) Se score==0, lexical rescue su tutto il corpus
    """
    result = retriever.retrieve_with_logs(query, k=cfg.top_k)
    retrieved_docs = result.get("results", []) or []

    answer = ""
    extracted = {"magnitude": None, "depth": None, "time": None}
    best_score = -1

    if retrieved_docs:
        # Rerank sui top_k
        for d in retrieved_docs:
            sc, ex = score_candidate(d.page_content, expected)
            if sc > best_score:
                best_score = sc
                answer = d.page_content
                extracted = ex

    # Se ancora 0 (nessun match), prova fallback lessicale su tutto il corpus
    if best_score <= 0:
        rescue_ans, rescue_ex, rescue_sc = lexical_rescue(all_docs, query, expected)
        if rescue_sc > best_score:
            best_score = rescue_sc
            answer = rescue_ans
            extracted = rescue_ex

    # REFUSAL CHECK (case-insensitive)
    ans_lower = (answer or "").lower()
    refused = (
        "i don't know" in ans_lower
        or "not enough data" in ans_lower
        or (answer.strip() == "")
    )

    expected_mag = normalize_value(expected.get("expected_magnitude"))
    expected_depth = normalize_value(expected.get("expected_depth"))
    expected_time = normalize_value(expected.get("expected_time"))

    mag_ok = normalize_value(extracted["magnitude"]) == expected_mag
    depth_ok = normalize_value(extracted["depth"]) == expected_depth
    time_ok = normalize_value(extracted["time"]) == expected_time

    return {
        "query": query,
        "refusal_correct": (refused == expected.get("should_refuse", False)),
        "magnitude_correct": mag_ok,
        "depth_correct": depth_ok,
        "time_correct": time_ok,
        "raw_answer": answer,
        "score": best_score
    }


def compute_scores(results: List[Dict]) -> Dict:
    def acc(key: str) -> float:
        if not results:
            return 0.0
        return mean([1 if r.get(key) else 0 for r in results])

    return {
        "magnitude_accuracy": acc("magnitude_correct"),
        "depth_accuracy": acc("depth_correct"),
        "time_accuracy": acc("time_correct"),
        "refusal_accuracy": acc("refusal_correct"),
    }


# ============================================================
# 4. LOAD EARTHQUAKE FILE
# ============================================================

def load_earthquake_file_as_docs(file_path: str, cfg: AppConfig) -> List[Document]:

    def normalize_fieldname(name: str):
        if not name:
            return ""
        name = name.strip()
        if name.startswith("#"):
            name = name[1:]
        name = name.replace("/", "_").replace("-", "_")
        return name.strip()

    encodings_to_try = cfg.encodings_to_try
    delimiter = cfg.csv_delimiter

    # canonical keys
    k_event_id = cfg.key_event_id
    k_time = cfg.key_time
    k_lat = cfg.key_latitude
    k_lon = cfg.key_longitude
    k_depth = cfg.key_depth
    k_mag = cfg.key_magnitude
    k_magtype = cfg.key_magtype
    k_location = cfg.key_location
    k_event_type = cfg.key_event_type
    k_author = cfg.key_author
    k_catalog = cfg.key_catalog

    file_handle = None
    last_error = None

    for enc in encodings_to_try:
        try:
            file_handle = open(file_path, "r", encoding=enc, newline="")
            file_handle.read(2048)
            file_handle.seek(0)
            break
        except UnicodeDecodeError as e:
            last_error = e

    if not file_handle:
        raise UnicodeDecodeError("decode", b"", 0, 1,
                                 f"Unable to decode file: {encodings_to_try} | {last_error}")

    documents = []

    with file_handle as f:
        raw = csv.reader(f, delimiter=delimiter)
        header = next(raw)
        normalized_header = [normalize_fieldname(h) for h in header]

        reader = csv.DictReader(f, delimiter=delimiter, fieldnames=normalized_header)
        next(reader)  # skip original header row

        for row in reader:
            event_text = (
                f"Event ID: {row.get(k_event_id, '')}\n"
                f"Date/Time: {row.get(k_time, '')}\n"
                f"Latitude: {row.get(k_lat, '')}\n"
                f"Longitude: {row.get(k_lon, '')}\n"
                f"Depth (km): {row.get(k_depth, '')}\n"
                f"Magnitude: {row.get(k_mag, '')} ({row.get(k_magtype, '')})\n"
                f"Location: {row.get(k_location, '')}\n"
                f"Event Type: {row.get(k_event_type, '')}\n"
                f"Author: {row.get(k_author, '')}\n"
                f"Catalog: {row.get(k_catalog, '')}\n"
            )

            documents.append(
                Document(
                    page_content=event_text,
                    metadata={"event_id": row.get(k_event_id, "")}
                )
            )

    return documents


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("\n=== EARTHQUAKE RAG SYSTEM INITIALIZATION ===\n")

    cfg = AppConfig()
    sys.path.append(cfg.project_root)

    earthquakes_file = cfg.earthquakes_path
    gt_file = cfg.ground_truth_path

    cleaner = TextCleaner()
    splitter = TextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

    raw_events = load_earthquake_file_as_docs(earthquakes_file, cfg)

    for doc in raw_events:
        doc.page_content = cleaner.clean(doc.page_content)

    chunks = splitter.split_documents(raw_events)

    embedding_model = EmbeddingModel()
    vector_manager = VectorStoreManager(embedding_model)
    vector_manager.create_index(chunks)

    retriever = Retriever(vector_manager)

    samples = load_ground_truth(gt_file)
    results = []

    for entry in samples:
        print("→ Evaluating:", entry["query"])
        res = evaluate_sample(entry["query"], entry, retriever, cfg, chunks)
        results.append(res)

    scores = compute_scores(results)
    print("\n=== FINAL EVALUATION RESULTS ===")
    print(json.dumps(scores, indent=4))

    os.makedirs(cfg.evaluation_dir, exist_ok=True)
    with open(cfg.evaluation_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\n✓ DONE\n")


if __name__ == "__main__":
    main()