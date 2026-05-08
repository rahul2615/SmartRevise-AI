"""
Model Cache - Pre-load all ML models at application startup.
This module provides a singleton ModelCache that loads ML models once,
so they're shared across all requests instead of being re-instantiated per call.

Models cached (ML only — no deep learning):
  - SpaCy NLP pipeline (en_core_web_sm)
  - NLTK corpora (wordnet, punkt, averaged_perceptron_tagger)
  - TF-IDF Vectorizer (scikit-learn)
"""

import time
import threading

_lock = threading.Lock()


class ModelCache:
    """Singleton that holds all ML models pre-loaded in memory."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.nlp = None
        self.nltk_ready = False
        self._load_times = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preload_all(self):
        """Call once at startup to warm up every model."""
        print("\n" + "=" * 55)
        print(">>  MODEL CACHE -- Pre-loading ML models at startup")
        print("=" * 55)
        total_start = time.time()

        self._load_nltk()
        self._load_spacy()

        elapsed = round(time.time() - total_start, 1)
        print(f"\n[OK]  All models loaded in {elapsed}s")
        for name, t in self._load_times.items():
            print(f"    * {name}: {t}s")
        print("=" * 55 + "\n")

    def get_spacy(self):
        """Return the cached spaCy Language model (or None)."""
        return self.nlp

    def is_nltk_ready(self):
        """Return True if NLTK corpora were downloaded successfully."""
        return self.nltk_ready

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_spacy(self):
        start = time.time()
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self._load_times["SpaCy (en_core_web_sm)"] = round(time.time() - start, 1)
            print(f"  [+] SpaCy loaded ({self._load_times['SpaCy (en_core_web_sm)']}s)")
        except Exception as e:
            print(f"  [X] SpaCy failed: {e}")
            self._load_times["SpaCy (en_core_web_sm)"] = "FAILED"

    def _load_nltk(self):
        start = time.time()
        try:
            import nltk
            for res in ["wordnet", "punkt", "averaged_perceptron_tagger"]:
                try:
                    if res == "wordnet":
                        nltk.data.find("corpora/wordnet")
                    elif res == "punkt":
                        nltk.data.find("tokenizers/punkt")
                    else:
                        nltk.data.find("taggers/averaged_perceptron_tagger")
                except LookupError:
                    nltk.download(res, quiet=True)
            self.nltk_ready = True
            self._load_times["NLTK Corpora"] = round(time.time() - start, 1)
            print(f"  [+] NLTK ready ({self._load_times['NLTK Corpora']}s)")
        except Exception as e:
            print(f"  [X] NLTK failed: {e}")
            self._load_times["NLTK Corpora"] = "FAILED"


# ---- Module-level convenience instance ----
cache = ModelCache()
