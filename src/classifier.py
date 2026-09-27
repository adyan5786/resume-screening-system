"""
Module Name: classifier.py
Aim: To classify resumes into designated job role/domain categories using supervised machine learning algorithms.
Input: Labeled resume feature vectors and target domain categories.
Output: Trained machine learning classifier, predicted job categories, and classification probabilities.

══════════════════════════════════════════════════════════════════════════════
  IMPORTANT — MODULE ROLE CLARIFICATION
══════════════════════════════════════════════════════════════════════════════

  This file has TWO distinct sections:

  [SECTION A]  CORE CLASSIFIER UTILITIES  (lines below this header)
  ─────────────────────────────────────────────────────────────────
  General-purpose functions (train_classifier, predict_category, etc.) used
  by the main pipeline.  The main pipeline is a SIMILARITY-RANKING system —
  cosine similarity between TF-IDF vectors — NOT a hard classifier.  These
  utilities exist for completeness and are imported by the Streamlit app.

  [SECTION B]  AUXILIARY CATEGORY-CLASSIFICATION EXPERIMENT  (see bottom)
  ─────────────────────────────────────────────────────────────────────────
  A clearly-labelled standalone experiment that trains a MultinomialNB model
  on TF-IDF features to PREDICT the resume's domain/category label.  This is
  an exploratory study distinct from the primary ranking system.
  ─ Entry point: run_classification_experiment(texts, labels)
  ─ Outputs: sklearn classification_report (precision, recall, F1 per class)

══════════════════════════════════════════════════════════════════════════════
"""

import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


# ═══════════════════════════════════════════════════════════════════════════
#  [SECTION A]  CORE CLASSIFIER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

_MODEL_REGISTRY = {
    "multinomial_nb": MultinomialNB,
    "linear_svc": LinearSVC,
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
}


def train_classifier(X_train, y_train, model_type: str = "multinomial_nb") -> object:
    """
    Train a machine learning classifier on resume feature matrices and category labels.

    Args:
        X_train (array-like): Training feature vectors.
        y_train (array-like): Ground truth job category labels.
        model_type (str, optional): Algorithm to use
            ('multinomial_nb', 'linear_svc', 'random_forest', 'logistic_regression').
            Defaults to 'multinomial_nb'.

    Returns:
        object: Trained classifier model artifact.
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from: {list(_MODEL_REGISTRY.keys())}"
        )

    clf = _MODEL_REGISTRY[model_type]()
    clf.fit(X_train, y_train)
    return clf


def predict_category(model: object, feature_vector) -> str:
    """
    Predict the job role category for a given resume feature vector.

    Args:
        model (object): Trained classification model.
        feature_vector (array-like): Numerical feature representation of the resume.
            Pass a 2-D array/sparse matrix with shape (1, n_features).

    Returns:
        str: Predicted job domain category label.
    """
    prediction = model.predict(feature_vector)
    return str(prediction[0])


def predict_probabilities(model: object, feature_vector) -> dict:
    """
    Compute category prediction probabilities for a resume feature vector.

    Args:
        model (object): Trained classification model (must support predict_proba or decision_function).
        feature_vector (array-like): Numerical feature representation of the resume.

    Returns:
        dict: Mapping of category names to probability scores.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feature_vector)[0]
        classes = model.classes_
    elif hasattr(model, "decision_function"):
        # For LinearSVC: convert raw decision scores to softmax-like probabilities
        raw_scores = model.decision_function(feature_vector)[0]
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probs = exp_scores / exp_scores.sum()
        classes = model.classes_
    else:
        raise AttributeError(
            "Model does not support predict_proba or decision_function."
        )

    return {str(cls): float(prob) for cls, prob in zip(classes, probs)}


def save_model(model: object, file_path: str) -> None:
    """
    Serialize and save a trained model artifact to disk.

    Args:
        model (object): Trained model instance.
        file_path (str): Destination file path (.pkl or .joblib).

    Returns:
        None
    """
    joblib.dump(model, file_path)


def load_model(file_path: str) -> object:
    """
    Load and deserialize a trained model artifact from disk.

    Args:
        file_path (str): Path to the saved model file.

    Returns:
        object: Loaded classifier model instance.
    """
    return joblib.load(file_path)


# ═══════════════════════════════════════════════════════════════════════════
#  [SECTION B]  AUXILIARY CATEGORY-CLASSIFICATION EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════
#
#  PURPOSE
#  ───────
#  This section is an *exploratory experiment* that asks a different question
#  from the main pipeline:
#
#    Main pipeline  →  "How similar is resume X to THIS job description?"
#                      (continuous similarity score, no labels required)
#
#    This experiment →  "Which domain/category does resume X belong to?"
#                       (discrete classification, requires labelled dataset)
#
#  The experiment uses MultinomialNB trained on TF-IDF features with an
#  80/20 stratified train/test split and reports per-class precision, recall,
#  and F1-score via sklearn's classification_report.
#
#  WHEN TO USE
#  ───────────
#  Run this experiment separately (not as part of the main screening pipeline)
#  to understand how well the vocabulary separates domain categories, or to
#  serve as a sanity-check label predictor for datasets where ground-truth
#  category labels are available (e.g. the Kaggle "Resume Dataset").
# ═══════════════════════════════════════════════════════════════════════════


def run_classification_experiment(
    texts: list,
    labels: list,
    test_size: float = 0.20,
    random_state: int = 42,
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    verbose: bool = True,
) -> dict:
    """
    [AUXILIARY EXPERIMENT] Train a MultinomialNB classifier on TF-IDF features to predict
    resume domain/category labels. Uses an 80/20 stratified train/test split.

    This function is a STANDALONE EXPLORATORY MODULE and is NOT part of the primary
    similarity-ranking pipeline.  It requires a labelled dataset.

    Args:
        texts (list): List of preprocessed resume strings.
        labels (list): Corresponding category labels (e.g. "Data Science", "HR", "Java Developer").
        test_size (float, optional): Fraction of data reserved for testing. Defaults to 0.20 (20%).
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        max_features (int, optional): TF-IDF vocabulary cap. Defaults to 5000.
        ngram_range (tuple, optional): N-gram range for TF-IDF. Defaults to (1, 2).
        verbose (bool, optional): If True, prints a formatted report to stdout. Defaults to True.

    Returns:
        dict: {
            "classification_report": str,    # Full sklearn classification_report string
            "accuracy": float,               # Overall test-set accuracy
            "vectorizer": TfidfVectorizer,   # Fitted vectorizer (for downstream use)
            "model": MultinomialNB,          # Trained classifier (for downstream use)
            "X_test": sparse matrix,         # Held-out feature matrix
            "y_test": list,                  # Held-out true labels
            "y_pred": list,                  # Model predictions on test set
        }
    """
    if len(texts) != len(labels):
        raise ValueError("'texts' and 'labels' must have the same length.")
    if len(texts) < 4:
        raise ValueError("At least 4 labelled samples are required for a train/test split.")

    # ── Vectorize using a corpus-level TF-IDF (fit on train split only to avoid leakage) ──
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        min_df=1,
    )
    X_train = vectorizer.fit_transform(X_train_texts)   # fit on training split only
    X_test = vectorizer.transform(X_test_texts)

    # ── Train MultinomialNB ──────────────────────────────────────────────
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────
    y_pred = model.predict(X_test).tolist()
    acc = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, zero_division=0)

    if verbose:
        separator = "-" * 70
        print(f"\n{separator}")
        print("  [AUXILIARY EXPERIMENT] Resume Category Classification - MultinomialNB")
        print(f"  Train size : {len(y_train)} samples")
        print(f"  Test  size : {len(y_test)} samples  ({int(test_size * 100)}% split)")
        print(f"  Vocabulary : {max_features} features  |  n-grams: {ngram_range}")
        print(separator)
        print(f"\n  Overall Accuracy : {acc:.4f}\n")
        print("  Per-class Classification Report:")
        print("  " + report_str.replace("\n", "\n  "))
        print(separator)
        print("  NOTE: This is an AUXILIARY EXPERIMENT -- not the primary ranking system.")
        print(f"{separator}\n")

    return {
        "classification_report": report_str,
        "accuracy": acc,
        "vectorizer": vectorizer,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }
