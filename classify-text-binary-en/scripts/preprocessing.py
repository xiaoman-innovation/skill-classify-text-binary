"""
Text preprocessing: cleaning, tokenization, vectorization, statistics,
and pretrained embedding loading (GloVe, fastText).
"""

from __future__ import annotations
import re
import os
import gzip
import string
import shutil
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------

def _is_missing(val) -> bool:
    """Check if a value is None, NaN, or pd.NA."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if hasattr(pd, 'isna') and pd.isna(val):
        return True
    return False


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_PUNCT_RE = re.compile(r"[%s]" % re.escape(string.punctuation))
_NUMBER_RE = re.compile(r"\b\d+(\.\d+)?\b")
_MULTISPACE_RE = re.compile(r"\s+")
_WORD_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*\b")


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(
    texts: list,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_html: bool = True,
    remove_punctuation: bool = True,
    remove_numbers: bool = False,
    remove_stopwords: bool = False,
    stopwords_lang: str = "english",
    min_length: int = 1,
) -> list:
    """
    Pipeline-based text cleaning. Returns list of cleaned strings.
    Empty results become empty string "" (not None).
    """
    cleaned = [str(t) if not _is_missing(t) else "" for t in texts]

    if lowercase:
        cleaned = [t.lower() for t in cleaned]

    if remove_urls:
        cleaned = [_URL_RE.sub(" ", t) for t in cleaned]

    if remove_html:
        cleaned = [_HTML_RE.sub(" ", t) for t in cleaned]

    if remove_punctuation:
        cleaned = [_PUNCT_RE.sub(" ", t) for t in cleaned]

    if remove_numbers:
        cleaned = [_NUMBER_RE.sub(" ", t) for t in cleaned]

    # Normalize whitespace
    cleaned = [_MULTISPACE_RE.sub(" ", t).strip() for t in cleaned]

    if remove_stopwords:
        stopwords = _get_stopwords(stopwords_lang)
        cleaned = [" ".join(w for w in t.split() if w not in stopwords)
                   for t in cleaned]

    if min_length > 1:
        cleaned = [t if len(t) >= min_length else "" for t in cleaned]

    return cleaned


def _get_stopwords(lang: str = "english") -> set:
    """Get stopwords set, with fallback to a minimal built-in list."""
    try:
        import nltk
        try:
            return set(nltk.corpus.stopwords.words(lang))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(nltk.corpus.stopwords.words(lang))
    except Exception:
        pass
    # Minimal built-in English stopword list
    return {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "shall", "can",
            "not", "no", "this", "that", "these", "those", "it", "its", "i",
            "me", "my", "we", "our", "you", "your", "he", "she", "they", "them"}


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_texts(
    texts: list,
    method: str = "word",
    max_features: int = None,
) -> list:
    """
    Tokenize a list of texts.
    method: 'word' | 'char' | 'word+char'
    Returns list of tokenized strings (space-joined tokens).
    """
    if method == "word":
        tokens_list = [_WORD_TOKEN_RE.findall(str(t).lower()) for t in texts]
    elif method == "char":
        tokens_list = [list(str(t).lower().replace(" ", "")) for t in texts]
    elif method == "word+char":
        tokens_list = []
        for t in texts:
            t_lower = str(t).lower()
            words = _WORD_TOKEN_RE.findall(t_lower)
            chars = list(t_lower.replace(" ", ""))
            tokens_list.append(words + chars)
    else:
        raise ValueError(f"Unknown tokenization method: {method}")

    if max_features is not None:
        all_tokens = [tok for tokens in tokens_list for tok in tokens]
        counter = Counter(all_tokens)
        top_tokens = {tok for tok, _ in counter.most_common(max_features)}
        tokens_list = [[t for t in tokens if t in top_tokens]
                       for tokens in tokens_list]

    return [" ".join(tokens) for tokens in tokens_list]


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------

def build_vectorizer(
    vectorizer_type: str = "tfidf",
    ngram_range: tuple = (1, 1),
    max_features: int = 10000,
    min_df: int = 2,
    max_df: float = 0.95,
    **kwargs,
):
    """
    Factory returning a scikit-learn vectorizer configured with sensible defaults.

    vectorizer_type: 'count' | 'tfidf' | 'onehot' | 'char' | 'char_wb'
    """
    if vectorizer_type in ("count", "onehot"):
        binary = (vectorizer_type == "onehot")
        return CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            binary=binary,
            lowercase=True,
            **kwargs,
        )
    elif vectorizer_type == "tfidf":
        return TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            sublinear_tf=True,
            **kwargs,
        )
    elif vectorizer_type in ("char", "char_wb"):
        analyzer = "char_wb" if vectorizer_type == "char_wb" else "char"
        return CountVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")


VECTORIZER_MAP = {
    "count": lambda: build_vectorizer("count", ngram_range=(1, 1)),
    "count_bigram": lambda: build_vectorizer("count", ngram_range=(1, 2)),
    "onehot": lambda: build_vectorizer("onehot", ngram_range=(1, 1)),
    "onehot_bigram": lambda: build_vectorizer("onehot", ngram_range=(1, 2)),
    "tfidf": lambda: build_vectorizer("tfidf", ngram_range=(1, 1)),
    "tfidf_bigram": lambda: build_vectorizer("tfidf", ngram_range=(1, 2)),
    "char_wb": lambda: build_vectorizer("char_wb", ngram_range=(3, 5)),
}

VECTORIZER_DISPLAY_NAMES = {
    "count": "Count (1-gram)",
    "count_bigram": "Count (1,2-gram)",
    "onehot": "OneHot (1-gram)",
    "onehot_bigram": "OneHot (1,2-gram)",
    "tfidf": "TF-IDF (1-gram)",
    "tfidf_bigram": "TF-IDF (1,2-gram)",
    "char_wb": "Char n-gram (3-5)",
}


def get_vectorizer(name: str):
    """Get a pre-configured vectorizer by shortcut name."""
    if name not in VECTORIZER_MAP:
        raise ValueError(f"Unknown vectorizer name '{name}'. "
                         f"Available: {list(VECTORIZER_MAP.keys())}")
    return VECTORIZER_MAP[name]()


# ---------------------------------------------------------------------------
# Embedding Vectorizer (dense embeddings for traditional ML)
# ---------------------------------------------------------------------------

class EmbeddingVectorizer:
    """
    sklearn-compatible vectorizer that converts texts to dense vectors by
    averaging pretrained word embeddings (GloVe / Word2Vec / fastText).

    Usage:
        vec = EmbeddingVectorizer(embeddings={'the': array([...]), ...},
                                  embedding_dim=300)
        X_dense = vec.fit_transform(texts)
    """
    def __init__(self, embeddings: dict, embedding_dim: int = 300):
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.is_fitted_ = True

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        result = np.zeros((len(X), self.embedding_dim), dtype=np.float32)
        for i, text in enumerate(X):
            words = str(text).split()
            vecs = [self.embeddings[w] for w in words if w in self.embeddings]
            if vecs:
                result[i] = np.mean(vecs, axis=0)
        return result

    def fit_transform(self, X, y=None):
        return self.transform(X)


def create_embedding_vectorizer(embeddings: dict, embedding_dim: int = 300):
    """Factory: return an EmbeddingVectorizer instance."""
    return EmbeddingVectorizer(embeddings, embedding_dim)


# ---------------------------------------------------------------------------
# Text Statistics
# ---------------------------------------------------------------------------

def compute_text_stats(texts: list) -> dict:
    """
    Compute comprehensive text statistics.

    Returns dict:
        count, mean_length, median_length, min_length, max_length,
        p25_length, p75_length, p95_length, vocab_size, missing_ratio,
        duplicate_ratio
    """
    if not texts:
        return {
            "count": 0, "mean_length": 0.0, "median_length": 0.0,
            "min_length": 0, "max_length": 0, "std_length": 0.0,
            "p25_length": 0.0, "p75_length": 0.0, "p90_length": 0.0,
            "p95_length": 0.0, "p99_length": 0.0,
            "vocab_size": 0, "missing_ratio": 0.0, "duplicate_ratio": 0.0,
        }
    n = len(texts)
    # Lengths: handle None / NaN as empty
    lengths = [len(str(t).split()) if not _is_missing(t) else 0
               for t in texts]

    # Missing
    missing = sum(1 for t in texts
                  if _is_missing(t)
                  or str(t).strip() == "")

    # Duplicates
    non_empty = [str(t).strip() for t in texts
                 if t is not None and not _is_missing(t)
                 and str(t).strip() != ""]
    dup_count = len(non_empty) - len(set(non_empty))

    # Vocabulary
    all_tokens = []
    for t in non_empty:
        all_tokens.extend(_WORD_TOKEN_RE.findall(t.lower()))
    vocab_size = len(set(all_tokens))

    arr = np.array(lengths)
    return {
        "count": n,
        "mean_length": round(float(np.mean(arr)), 1),
        "median_length": round(float(np.median(arr)), 1),
        "min_length": int(np.min(arr)),
        "max_length": int(np.max(arr)),
        "std_length": round(float(np.std(arr)), 1),
        "p25_length": round(float(np.percentile(arr, 25)), 1),
        "p75_length": round(float(np.percentile(arr, 75)), 1),
        "p90_length": round(float(np.percentile(arr, 90)), 1),
        "p95_length": round(float(np.percentile(arr, 95)), 1),
        "p99_length": round(float(np.percentile(arr, 99)), 1),
        "vocab_size": vocab_size,
        "missing_ratio": round(missing / max(n, 1), 4),
        "duplicate_ratio": round(dup_count / max(len(non_empty), 1), 4),
    }


# ---------------------------------------------------------------------------
# Vocabulary Richness
# ---------------------------------------------------------------------------

def compute_vocabulary_richness(texts: list) -> dict:
    """
    Compute vocabulary richness metrics.

    Returns dict:
        vocab_size, total_tokens, type_token_ratio (TTR),
        hapax_legomena_ratio, repeated_word_ratio
    """
    n = len(texts)
    if n == 0:
        return {
            "vocab_size": 0, "total_tokens": 0,
            "type_token_ratio": 0.0, "hapax_legomena_ratio": 0.0,
            "repeated_word_ratio": 0.0,
        }

    all_tokens = []
    for t in texts:
        t_clean = str(t).strip() if t is not None and not (
            isinstance(t, float) and np.isnan(t)) else ""
        if t_clean:
            all_tokens.extend(_WORD_TOKEN_RE.findall(t_clean.lower()))

    total_tokens = len(all_tokens)
    if total_tokens == 0:
        return {
            "vocab_size": 0, "total_tokens": 0,
            "type_token_ratio": 0.0, "hapax_legomena_ratio": 0.0,
            "repeated_word_ratio": 0.0,
        }

    counter = Counter(all_tokens)
    vocab_size = len(counter)
    hapax_count = sum(1 for v in counter.values() if v == 1)
    repeated_tokens = sum(v - 1 for v in counter.values())  # tokens beyond first occurrence

    return {
        "vocab_size": vocab_size,
        "total_tokens": total_tokens,
        "type_token_ratio": round(vocab_size / total_tokens, 4),
        "hapax_legomena_ratio": round(hapax_count / max(vocab_size, 1), 4),
        "repeated_word_ratio": round(repeated_tokens / total_tokens, 4),
    }


# ---------------------------------------------------------------------------
# Syntactic Complexity
# ---------------------------------------------------------------------------

# Clause-introducing words / subordinators for heuristic clause counting
_CLAUSE_MARKERS = re.compile(
    r'\b(that|which|who|whom|whose|because|although|though|while|'
    r'whereas|since|after|before|until|when|whenever|if|unless|'
    r'where|wherever|so\s+that|in\s+order\s+that|as|than|'
    r'whether|even\s+though|even\s+if|provided\s+that|'
    r'as\s+long\s+as|as\s+soon\s+as|as\s+if|as\s+though|'
    r'once|now\s+that|rather\s+than|so)\b',
    re.IGNORECASE,
)

# Sentence boundary detection (period, exclamation, question mark followed by space/capital or end)
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$')


def _count_clauses_heuristic(sentence: str) -> int:
    """Estimate clause count by counting subordinating conjunctions + 1."""
    markers = len(_CLAUSE_MARKERS.findall(sentence))
    return max(1, markers + 1)


def compute_syntactic_complexity(texts: list) -> dict:
    """
    Estimate syntactic complexity of English texts using heuristics.

    Returns dict:
        avg_sentence_length (words), avg_sentences_per_text,
        avg_clauses_per_sentence
    """
    n = len(texts)
    if n == 0:
        return {
            "avg_sentence_length": 0.0,
            "avg_sentences_per_text": 0.0,
            "avg_clauses_per_sentence": 0.0,
        }

    all_sentence_lengths = []
    all_sentence_counts = []
    all_clause_counts = []

    for t in texts:
        t_clean = str(t).strip() if t is not None and not (
            isinstance(t, float) and np.isnan(t)) else ""
        if not t_clean:
            all_sentence_counts.append(0)
            continue

        # Split into sentences
        # Simple heuristic: split on .!? followed by space+capital or end of string
        raw_sentences = _SENTENCE_SPLIT_RE.split(t_clean)
        # Also split on newline as fallback for texts without proper punctuation
        if len(raw_sentences) <= 1:
            raw_sentences = [s.strip() for s in t_clean.replace('\n', '.').split('.') if s.strip()]
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) >= 3]

        if not sentences:
            all_sentence_counts.append(1)
            words = len(_WORD_TOKEN_RE.findall(t_clean.lower()))
            all_sentence_lengths.append(words)
            all_clause_counts.append(1)
            continue

        all_sentence_counts.append(len(sentences))
        for sent in sentences:
            words = len(_WORD_TOKEN_RE.findall(sent.lower()))
            all_sentence_lengths.append(words)
            all_clause_counts.append(_count_clauses_heuristic(sent))

    avg_sent_len = round(float(np.mean(all_sentence_lengths)), 1) if all_sentence_lengths else 0.0
    avg_sent_per_text = round(float(np.mean(all_sentence_counts)), 1) if all_sentence_counts else 0.0
    avg_clauses = round(float(np.mean(all_clause_counts)), 1) if all_clause_counts else 0.0

    return {
        "avg_sentence_length": avg_sent_len,
        "avg_sentences_per_text": avg_sent_per_text,
        "avg_clauses_per_sentence": avg_clauses,
    }


# ---------------------------------------------------------------------------
# Embedding Loading (GloVe / fastText)
# ---------------------------------------------------------------------------

_EMBEDDING_CACHE = {}  # Cache for loaded embeddings to avoid re-reading large files per trial


def _warn_empty_embeddings(emb_type: str, path: str, embedding_dim: int):
    """Warn when an embedding file was loaded but no vectors matched the requested dim."""
    import os
    fsize_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
    print(f"  [WARN] {emb_type} file loaded but 0 vectors matched dim={embedding_dim}. "
          f"File: {path} ({fsize_mb:.0f} MB). Check that the file dimension matches "
          f"--embedding-dim (default 300).", flush=True)


def load_glove_embeddings(path: str, embedding_dim: int = 300) -> dict:
    """
    Load GloVe/fastText embeddings from a text file.
    Format per line: word f1 f2 ... fN
    Returns {word: np.ndarray} dict.  Cached per (path, dim).
    """
    import os
    cache_key = (os.path.abspath(path), embedding_dim)
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    embeddings = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            word = parts[0]
            try:
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                if len(vec) == embedding_dim:
                    embeddings[word] = vec
            except ValueError:
                continue
    if not embeddings:
        _warn_empty_embeddings("GloVe", path, embedding_dim)
    _EMBEDDING_CACHE[cache_key] = embeddings
    return embeddings


def _is_fasttext_binary_model(path: str) -> bool:
    """
    Detect fastText binary model (.bin) format.
    The fastText binary model header starts with model config int32 fields:
    dim, ws, epoch, minCount, neg, wordNgrams, loss, model, bucket, minn, maxn, ...
    We check: the file is large enough, starts with non-text bytes, and the
    third int32 (epoch/dim) is a plausible embedding dimension (50-4096).
    """
    import struct
    try:
        with open(path, "rb") as f:
            raw = f.read(48)
        if len(raw) < 48:
            return False
        # Parse first few int32 fields (little-endian, standard fastText format)
        dim = struct.unpack_from("<i", raw, 0)[0]
        ws = struct.unpack_from("<i", raw, 4)[0]
        epoch_or_vocab = struct.unpack_from("<i", raw, 8)[0]
        # The third field could be epoch or vocab_size.
        # Check: (1) dim is plausible (50-4096) or epoch_or_vocab is plausible
        # (2) the file doesn't start with a text header like "2000000 300\n"
        first_bytes = raw[:32]
        # If the first 4 bytes decode as ASCII digits, this is likely a text header
        try:
            first_word = raw.split(b"\n")[0].split(b" ")[0].decode("ascii")
            if first_word.isdigit():
                return False  # text header, not fastText binary model
        except (UnicodeDecodeError, IndexError):
            pass
        # Plausible check: dim or epoch_or_vocab in reasonable range
        plausible_dim = 50 <= dim <= 4096
        plausible_epoch = 50 <= epoch_or_vocab <= 4096
        return plausible_dim or plausible_epoch
    except Exception:
        return False


def _load_fasttext_binary_with_gensim(path: str, embedding_dim: int) -> dict:
    """
    Load a fastText binary .bin model using gensim and extract word vectors.
    Returns {word: np.ndarray} dict, or empty dict on failure.
    """
    try:
        import gensim.models.fasttext as ft
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ft.load_facebook_model(path)
        embeddings = {}
        for word in model.wv.key_to_index:
            vec = model.wv[word].astype(np.float32)
            if len(vec) == embedding_dim:
                embeddings[word] = vec
        if embeddings:
            print(f"  [ADAPT] loaded fastText binary model via gensim: "
                  f"{len(embeddings):,} vectors (dim={embedding_dim})", flush=True)
        return embeddings
    except Exception as e:
        print(f"  [ADAPT] gensim fastText fallback failed: {e}", flush=True)
        return {}


def load_fasttext_embeddings(path: str, embedding_dim: int = 300) -> dict:
    """
    Load fastText embeddings from .vec (text) or .bin (binary model) format.
    Auto-adapts to the actual file format:
      - Word2Vec binary (header: vocab_size dim) → word2vec loader
      - fastText binary model (.bin, no text header) → gensim fallback
      - Text .vec format → standard text parser
    Returns {word: np.ndarray} dict.  Cached per (path, dim).
    """
    import os
    cache_key = (os.path.abspath(path), embedding_dim)
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    # Step 1: try word2vec binary format (header: "vocab_size dim\\n")
    if _is_binary_word2vec(path):
        embeddings = load_word2vec_embeddings(path, embedding_dim)
        _EMBEDDING_CACHE[cache_key] = embeddings
        return embeddings

    # Step 2: try fastText model binary (.bin) format via gensim
    if _is_fasttext_binary_model(path):
        embeddings = _load_fasttext_binary_with_gensim(path, embedding_dim)
        if embeddings:
            _EMBEDDING_CACHE[cache_key] = embeddings
            return embeddings
        print(f"  [ADAPT] fastText binary model detected but gensim failed; "
              f"falling back to text parser", flush=True)

    # Step 3: text .vec format
    embeddings = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            word = parts[0]
            try:
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                if len(vec) == embedding_dim:
                    embeddings[word] = vec
            except ValueError:
                continue
    if not embeddings:
        _warn_empty_embeddings("fastText", path, embedding_dim)
    _EMBEDDING_CACHE[cache_key] = embeddings
    return embeddings


def _is_binary_word2vec(path: str) -> bool:
    """Detect Google word2vec binary format (first line is text header, rest binary).
    Text .vec files also start with a vocab_size/dim header, so we sample bytes
    after the header to distinguish: binary format contains packed floats (many
    non-printable bytes), text format is entirely printable."""
    try:
        with open(path, "rb") as f:
            header = f.readline().decode("utf-8", errors="replace").strip()
            parts = header.split()
            if not (len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()):
                return False
            # Sample 512 bytes past the header. Binary float data contains many
            # bytes outside the printable ASCII range, while text .vec files
            # are pure ASCII (digits, dots, minus signs, whitespace).
            chunk = f.read(512)
            if len(chunk) < 8:
                return False
            non_text = sum(1 for b in chunk
                          if b > 127 or (b < 0x20 and b not in (0x09, 0x0A, 0x0D)))
            return non_text > len(chunk) * 0.25
    except Exception:
        return False


def load_word2vec_embeddings(path: str, embedding_dim: int = 300) -> dict:
    """
    Load Word2Vec embeddings from text (.vec) or binary (.bin) format.
    - Text format: one word+vector per line (like GloVe)
    - Binary format: Google's original C binary format
    Returns {word: np.ndarray} dict.  Cached per (path, dim).
    """
    import os
    import struct
    cache_key = (os.path.abspath(path), embedding_dim)
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    embeddings = {}

    if _is_binary_word2vec(path):
        with open(path, "rb") as f:
            header = f.readline().decode("utf-8", errors="replace").strip()
            vocab_size, dim = map(int, header.split())
            # If file dimension differs from requested, use file dimension
            if dim != embedding_dim:
                if not hasattr(load_word2vec_embeddings, '_dim_warned'):
                    print(f"  [WARN] Word2Vec binary file has dim={dim}, "
                          f"requested dim={embedding_dim}. Using dim={dim}.", flush=True)
                    load_word2vec_embeddings._dim_warned = True
                embedding_dim = dim
            binary_len = np.dtype(np.float32).itemsize * embedding_dim
            for _ in range(vocab_size):
                word_bytes = bytearray()
                while True:
                    ch = f.read(1)
                    # Stop at any whitespace: space, tab, newline, or EOF
                    if ch in (b' ', b'\t', b'\n') or ch == b'':
                        break
                    word_bytes.extend(ch)
                word = word_bytes.decode("utf-8", errors="replace")
                vec_data = f.read(binary_len)
                if len(vec_data) == binary_len:
                    vec = np.frombuffer(vec_data, dtype=np.float32)
                    embeddings[word] = vec.copy()
    else:
        # Text format (like GloVe .vec)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                word = parts[0]
                try:
                    vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    if len(vec) == embedding_dim:
                        embeddings[word] = vec
                except ValueError:
                    continue

    if not embeddings:
        _warn_empty_embeddings("Word2Vec", path, embedding_dim)
    _EMBEDDING_CACHE[cache_key] = embeddings
    return embeddings


def load_embeddings(path: str, embedding_type: str = "glove",
                    embedding_dim: int = 300) -> dict:
    """
    Unified embedding loader dispatching by type.
    embedding_type: "glove" | "word2vec" | "fasttext"
    Returns {word: np.ndarray} dict.
    """
    if embedding_type == "word2vec":
        return load_word2vec_embeddings(path, embedding_dim)
    elif embedding_type == "fasttext":
        return load_fasttext_embeddings(path, embedding_dim)
    else:  # glove (default)
        return load_glove_embeddings(path, embedding_dim)


def build_embedding_matrix(
    word2idx: dict,
    embeddings: dict,
    embedding_dim: int,
    init_scale: float = 0.02,
) -> np.ndarray:
    """
    Build (vocab_size, embedding_dim) matrix from pretrained embeddings.
    Unknown words get random initialization scaled by init_scale.
    """
    vocab_size = len(word2idx)
    matrix = np.random.normal(scale=init_scale,
                              size=(vocab_size, embedding_dim)).astype(np.float32)
    found = 0
    for word, idx in word2idx.items():
        vec = embeddings.get(word)
        if vec is not None and len(vec) == embedding_dim:
            matrix[idx] = vec
            found += 1
    return matrix


def build_vocab(texts: list, max_vocab: int = 50000) -> dict:
    """Build word2idx vocabulary from list of tokenized texts."""
    counter = Counter()
    for t in texts:
        counter.update(str(t).split())
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        word2idx[word] = len(word2idx)
    return word2idx


def encode_texts_as_ids(texts: list, word2idx: dict, max_len: int) -> np.ndarray:
    """Convert tokenized texts to padded integer sequences."""
    result = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        tokens = str(t).split()[:max_len]
        for j, tok in enumerate(tokens):
            result[i, j] = word2idx.get(tok, 1)  # 1 = <UNK>
    return result


# ---------------------------------------------------------------------------
# Auto-download embeddings
# ---------------------------------------------------------------------------

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache",
                          "text-binary-classification", "embeddings")

# Per-session cache of embedding types whose download has definitively failed.
# Prevents repeated (slow) retry attempts across folds/models within one run.
_FAILED_EMBEDDING_TYPES: set = set()

_EMBEDDING_URLS = {
    "glove": {
        300: [
            # Stanford official (most reliable, CDN-backed)
            "https://nlp.stanford.edu/data/glove.6B.zip",
            # HuggingFace mirrors
            "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.300d.txt.gz",
            "https://hf-mirror.com/stanfordnlp/glove/resolve/main/glove.6B.300d.txt.gz",
        ],
    },
    "word2vec": {
        300: [
            # Official Google archive (most reliable)
            "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download",
            # HuggingFace mirrors
            "https://huggingface.co/fse/word2vec-google-news-300/resolve/main/GoogleNews-vectors-negative300.bin.gz",
            "https://hf-mirror.com/fse/word2vec-google-news-300/resolve/main/GoogleNews-vectors-negative300.bin.gz",
            # Alternative HuggingFace repo (slimmed version, smaller download)
            "https://huggingface.co/Word2vec/wikipedia2vec_enwiki_20180420_300d/resolve/main/wikipedia2vec_enwiki_20180420_300d.txt.gz",
            "https://hf-mirror.com/Word2vec/wikipedia2vec_enwiki_20180420_300d/resolve/main/wikipedia2vec_enwiki_20180420_300d.txt.gz",
        ],
    },
    "fasttext": {
        300: [
            # Facebook official CDN (most reliable, no auth required)
            "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz",
            "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300d.vec.gz",
            # HuggingFace mirrors (fallback)
            "https://huggingface.co/facebook/fasttext-wiki-news/resolve/main/cc.en.300d.bin.gz",
            "https://huggingface.co/facebook/fasttext-wiki-news/resolve/main/cc.en.300d.vec.gz",
        ],
    },
}

# Mapping of (emb_type, dim) to the filename that _load_* expects
_EMBEDDING_FILENAMES = {
    ("glove", 300): "glove.6B.300d.txt",
    ("word2vec", 300): "GoogleNews-vectors-negative300.bin",
    ("fasttext", 300): "cc.en.300d.vec",
}


def ensure_embeddings(emb_type: str, embedding_dim: int = 300) -> str | None:
    """
    Ensure pretrained embeddings exist in local cache, downloading if needed.

    Returns path to the embedding file on success, or None if the download
    failed or the embedding type/dimension combination is not supported.
    """
    if (emb_type, embedding_dim) in _FAILED_EMBEDDING_TYPES:
        return None  # Already failed this session — don't retry

    if emb_type not in _EMBEDDING_URLS:
        print(f"[WARN] Unknown embedding type: {emb_type}")
        return None
    if embedding_dim not in _EMBEDDING_URLS[emb_type]:
        print(f"[WARN] No download URL for {emb_type} {embedding_dim}d. "
              f"Available dims: {list(_EMBEDDING_URLS[emb_type].keys())}")
        return None

    filename = _EMBEDDING_FILENAMES.get((emb_type, embedding_dim))
    if filename is None:
        print(f"[WARN] No filename mapping for {emb_type} {embedding_dim}d")
        return None

    cached_path = os.path.join(_CACHE_DIR, filename)
    if os.path.exists(cached_path):
        return cached_path

    urls = _EMBEDDING_URLS[emb_type][embedding_dim]
    if isinstance(urls, str):
        urls = [urls]

    os.makedirs(_CACHE_DIR, exist_ok=True)

    for idx, url in enumerate(urls):
        is_gz = url.endswith(".gz")
        is_zip = url.endswith(".zip")
        compressed = is_gz or is_zip
        compressed_filename = os.path.basename(url)
        compressed_path = os.path.join(_CACHE_DIR, compressed_filename)

        # Download
        mirror_tag = f" (mirror {idx + 1}/{len(urls)})" if idx > 0 else ""
        print(f"  Downloading {emb_type} {embedding_dim}d embeddings{mirror_tag}...")
        print(f"  From: {url}")
        try:
            _download_with_progress(url, compressed_path)
        except Exception as e:
            print(f"  [WARN] Download failed: {e}")
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
            continue

        # Decompress
        if is_gz:
            try:
                _decompress_gz(compressed_path, cached_path)
            except Exception as e:
                print(f"  [WARN] Decompression failed: {e}")
                if os.path.exists(cached_path):
                    os.remove(cached_path)
                continue
        elif is_zip:
            try:
                _extract_zip(compressed_path, cached_path, filename)
            except Exception as e:
                print(f"  [WARN] Zip extraction failed: {e}")
                if os.path.exists(cached_path):
                    os.remove(cached_path)
                continue
        else:
            shutil.move(compressed_path, cached_path)

        # Clean up compressed file
        if os.path.exists(compressed_path):
            os.remove(compressed_path)

        return cached_path

    # All URLs failed — cache the failure so subsequent calls return immediately
    _FAILED_EMBEDDING_TYPES.add((emb_type, embedding_dim))
    print(f"\n  [WARN] All download URLs exhausted for {emb_type} {embedding_dim}d")
    print(f"  [TIP]  Manual fix: download the embedding file manually, then re-run with:")
    print(f"          --{emb_type.replace('word2vec','word2vec').replace('fasttext','fasttext').replace('glove','glove')}-path <path/to/file>")
    print(f"  [TIP]  Or choose a different embedding type that is reachable from your network.")
    return None


def _download_with_progress(url: str, dest: str):
    """Download a file with tqdm progress bar."""
    import urllib.request

    class _ProgressHook:
        def __init__(self):
            self.pbar = None

        def __call__(self, block_num, block_size, total_size):
            if self.pbar is None and total_size > 0:
                try:
                    from tqdm import tqdm
                except ImportError:
                    self.pbar = None
                    return
                self.pbar = tqdm(
                    total=total_size, unit="B", unit_scale=True,
                    unit_divisor=1024, desc="  Embeddings",
                )
            if self.pbar is not None:
                self.pbar.update(block_size)

    tmp_dest = dest + ".part"
    hook = _ProgressHook()
    try:
        urllib.request.urlretrieve(url, tmp_dest, reporthook=hook)
        os.replace(tmp_dest, dest)
    except Exception:
        if hook.pbar is not None:
            hook.pbar.close()
        if os.path.exists(tmp_dest):
            os.remove(tmp_dest)
        raise
    finally:
        if hook.pbar is not None:
            hook.pbar.close()


def _decompress_gz(gz_path: str, dest_path: str):
    """Decompress a .gz file to dest_path."""
    with gzip.open(gz_path, "rb") as f_in:
        with open(dest_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def _extract_zip(zip_path: str, dest_path: str, target_filename: str):
    """Extract a specific file from a .zip archive to dest_path."""
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the matching entry in the zip
        for name in zf.namelist():
            if name.endswith(target_filename) or name == target_filename:
                with zf.open(name) as f_in:
                    with open(dest_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"  Extracted: {name}")
                return
        raise FileNotFoundError(
            f"Could not find '{target_filename}' in zip archive"
        )
