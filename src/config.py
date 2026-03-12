# config.py
import os
from configparser import ConfigParser
from typing import List


class AppConfig:
    """
    Carica e fornisce accesso tipizzato alle impostazioni da config.ini.
    Supporta override tramite variabili d'ambiente (opzionali).
    """
    def __init__(self, config_path: str = None):
        self._parser = ConfigParser()

        # Calcola la root del progetto di default: <repo>/<src|scripts>/.. => root
        default_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        if config_path is None:
            # Cerca config.ini nella root del progetto
            config_path = os.path.join(default_project_root, "config.ini")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"File di configurazione non trovato: {config_path}")

        self._parser.read(config_path, encoding="utf-8")

        # Assicura PATHS.project_root
        project_root_from_ini = self._parser.get("PATHS", "project_root", fallback="").strip()
        if not project_root_from_ini:
            # set nel parser per coerenza con i getter
            self._parser.set("PATHS", "project_root", default_project_root)

        self._project_root = self._env_or(
            "PATHS_PROJECT_ROOT",
            self._parser.get("PATHS", "project_root"),
        )

    # -----------------------
    # Getter di alto livello
    # -----------------------
    @property
    def project_root(self) -> str:
        return self._project_root

    @property
    def data_dir(self) -> str:
        rel = self._env_or("PATHS_DATA_DIR", self._parser.get("PATHS", "data_dir", fallback="data"))
        return os.path.join(self.project_root, rel)

    @property
    def earthquakes_path(self) -> str:
        fname = self._env_or("PATHS_EARTHQUAKES_FILENAME",
                             self._parser.get("PATHS", "earthquakes_filename", fallback="query.txt"))
        return os.path.join(self.data_dir, fname)

    @property
    def ground_truth_path(self) -> str:
        fname = self._env_or("PATHS_GROUND_TRUTH_FILENAME",
                             self._parser.get("PATHS", "ground_truth_filename", fallback="ground_truth.jsonl"))
        return os.path.join(self.data_dir, fname)

    @property
    def evaluation_dir(self) -> str:
        rel = self._env_or("PATHS_EVALUATION_DIR",
                           self._parser.get("PATHS", "evaluation_dir", fallback="evaluation"))
        return os.path.join(self.project_root, rel)

    @property
    def evaluation_results_path(self) -> str:
        fname = self._env_or("PATHS_EVALUATION_RESULTS_FILENAME",
                             self._parser.get("PATHS", "evaluation_results_filename", fallback="results.json"))
        return os.path.join(self.evaluation_dir, fname)

    # -----------------------
    # Ingestion / CSV
    # -----------------------
    @property
    def encodings_to_try(self) -> List[str]:
        raw = self._parser.get("INGESTION", "encodings_to_try",
                               fallback="utf-8, utf-8-sig, cp1252, iso-8859-1")
        # consentiamo override completo via env
        raw = self._env_or("INGESTION_ENCODINGS_TO_TRY", raw)
        return [e.strip() for e in raw.split(",") if e.strip()]

    @property
    def csv_delimiter(self) -> str:
        return self._env_or("INGESTION_CSV_DELIMITER",
                            self._parser.get("INGESTION", "csv_delimiter", fallback="|"))

    # -----------------------
    # Text processing
    # -----------------------
    @property
    def chunk_size(self) -> int:
        return int(self._env_or("TEXT_PROCESSING_CHUNK_SIZE",
                                str(self._parser.getint("TEXT_PROCESSING", "chunk_size", fallback=400))))

    @property
    def chunk_overlap(self) -> int:
        return int(self._env_or("TEXT_PROCESSING_CHUNK_OVERLAP",
                                str(self._parser.getint("TEXT_PROCESSING", "chunk_overlap", fallback=40))))

    # -----------------------
    # Retrieval
    # -----------------------
    @property
    def top_k(self) -> int:
        return int(self._env_or("RETRIEVAL_TOP_K",
                                str(self._parser.getint("RETRIEVAL", "top_k", fallback=3))))

    # -----------------------
    # Mappatura colonne CSV
    # -----------------------
    def _get_key(self, section: str, option: str, default: str) -> str:
        env_key = f"LOGIC_{option}".upper()
        return self._env_or(env_key, self._parser.get(section, option, fallback=default))

    @property
    def key_event_id(self) -> str:
        return self._get_key("LOGIC", "event_id_key", "EventID")

    @property
    def key_time(self) -> str:
        return self._get_key("LOGIC", "time_key", "Time")

    @property
    def key_latitude(self) -> str:
        return self._get_key("LOGIC", "latitude_key", "Latitude")

    @property
    def key_longitude(self) -> str:
        return self._get_key("LOGIC", "longitude_key", "Longitude")

    @property
    def key_depth(self) -> str:
        return self._get_key("LOGIC", "depth_key", "Depth_Km")

    @property
    def key_magnitude(self) -> str:
        return self._get_key("LOGIC", "magnitude_key", "Magnitude")

    @property
    def key_magtype(self) -> str:
        return self._get_key("LOGIC", "magtype_key", "MagType")

    @property
    def key_location(self) -> str:
        return self._get_key("LOGIC", "location_key", "EventLocationName")

    @property
    def key_event_type(self) -> str:
        return self._get_key("LOGIC", "event_type_key", "EventType")

    @property
    def key_author(self) -> str:
        return self._get_key("LOGIC", "author_key", "Author")

    @property
    def key_catalog(self) -> str:
        return self._get_key("LOGIC", "catalog_key", "Catalog")

    # -----------------------
    # Utils
    # -----------------------
    @staticmethod
    def _env_or(env_name: str, default_value: str) -> str:
        val = os.getenv(env_name)
        return val if val is not None and str(val).strip() != "" else default_value