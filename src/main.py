import csv
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack

try:
    import joblib
except ImportError as exc:
    raise ImportError("The 'joblib' package is required to load the trained model.") from exc

try:
    import spacy
except ImportError:
    spacy = None

# Add the OCR reader folder to the import path so we can reuse its helpers.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OCR_DIR = BASE_DIR / "ocr-reader"
if str(OCR_DIR) not in sys.path:
    sys.path.append(str(OCR_DIR))

from read_invoice import parse_invoice_text, process_invoice  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "vectorizer.pkl"
TRAINING_DATA_PATH = BASE_DIR / "model" / "data" / "training_set_compras.csv"

_ASSET_CACHE = {}


def _load_targets():
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {TRAINING_DATA_PATH}")

    with TRAINING_DATA_PATH.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)

    if not header or len(header) < 2:
        raise ValueError("Could not determine target labels from training dataset.")

    return header[1:]


def _load_spacy_model():
    if spacy is None:
        return None

    try:
        return spacy.load("pt_core_news_md")
    except Exception as exc:  # pragma: no cover - spaCy model may be unavailable locally.
        print(f"[WARNING] Unable to load spaCy model 'pt_core_news_md': {exc}")
        return None


def load_assets():
    """Load and cache heavy assets so we only pay the cost once."""
    if "assets" in _ASSET_CACHE:
        return _ASSET_CACHE["assets"]

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    targets = _load_targets()
    nlp = _load_spacy_model()

    assets = (model, vectorizer, targets, nlp)
    _ASSET_CACHE["assets"] = assets
    return assets


def _ensure_feature_dimensions(model, vectorizer):
    estimator = model.estimators_[0]
    total_features = estimator.coef_.shape[1]
    tfidf_features = vectorizer.transform([""]).shape[1]
    spacy_features = max(total_features - tfidf_features, 0)
    return tfidf_features, spacy_features


def _build_feature_vector(text, model, vectorizer, nlp):
    cleaned = text.lower().strip()
    tfidf_vector = vectorizer.transform([cleaned])
    _, spacy_features = _ensure_feature_dimensions(model, vectorizer)

    if spacy_features > 0:
        if nlp is not None:
            spacy_vector = nlp(cleaned).vector.reshape(1, -1)
            if spacy_vector.shape[1] != spacy_features:
                # Resize gracefully if the embedding dimension differs.
                spacy_vector = np.resize(spacy_vector, (1, spacy_features))
        else:
            # Keep feature size consistent even when spaCy isn't available.
            spacy_vector = np.zeros((1, spacy_features), dtype=np.float32)

        spacy_sparse = csr_matrix(spacy_vector)
        return hstack([tfidf_vector, spacy_sparse])

    return tfidf_vector


def predict_owners(item_name, assets, threshold=0.75):
    model, vectorizer, targets, nlp = assets
    features = _build_feature_vector(item_name, model, vectorizer, nlp)
    probabilities = model.predict_proba(features)

    predictions = []
    best_index = None
    best_probability = -1.0

    for index, proba in enumerate(probabilities):
        if proba.shape[1] == 2:
            probability = proba[0][1]
        else:
            probability = proba[0][0]

        if probability > threshold:
            predictions.append(targets[index])

        if probability > best_probability:
            best_probability = probability
            best_index = index

    if not predictions and best_index is not None:
        predictions.append(targets[best_index])

    return predictions


def process_invoice_items(image_path):
    ocr_text = process_invoice(str(image_path), lang="por", save_processed=False)
    items = parse_invoice_text(ocr_text)
    assets = load_assets()

    results = []
    for item in items:
        owners = predict_owners(item["item"], assets)
        results.append(
            {
                "item": item["item"],
                "quantidade": item["quantidade"],
                "preco_total": item["preco_total"],
                "donos": owners,
            }
        )

    return results


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv

    if len(arguments) != 1:
        print("Usage: python -m src.main <invoice_photo>", file=sys.stderr)
        return 1

    image_path = Path(arguments[0]).expanduser()
    if not image_path.is_file():
        print(f"[ERROR] Image file not found: {image_path}", file=sys.stderr)
        return 1

    try:
        results = process_invoice_items(image_path)
    except Exception as exc:
        print(f"[ERROR] Failed to process invoice: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("No items were detected in the invoice.")
        return 0

    print(f"Found {len(results)} items:")
    for idx, entry in enumerate(results, 1):
        owners = ", ".join(entry["donos"]) if entry["donos"] else "N/A"
        print(f"{idx}. {entry['item']} (x{entry['quantidade']}) - €{entry['preco_total']:.2f}")
        print(f"   -> Dono(s): {owners}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

