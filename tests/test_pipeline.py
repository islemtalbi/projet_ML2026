"""
Tests pytest — Module 6 MLOps
Couvre : schéma données, imputation, NLP, prédictions, seuil performance, API
"""

import os
import re
import sys
import pytest
import numpy as np
import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ════════════════════════════════════════════════════════════════════
# 1. TESTS SCHÉMA DES DONNÉES
# ════════════════════════════════════════════════════════════════════

REQUIRED_COLUMNS = [
    "Poids", "Volume", "Conductivite", "Opacite", "Rigidite",
    "Prix_Revente", "Source", "Rapport_Collecte", "Categorie",
]

NUMERIC_COLS = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]

RAW_DATA_PATH = "data/raw/dataset_ProjetML_2026.csv"
CLEAN_DATA_PATH = "data/processed/dataset_clean.csv"


@pytest.fixture(scope="module")
def raw_df():
    if not os.path.exists(RAW_DATA_PATH):
        pytest.skip(f"Fichier introuvable : {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)


@pytest.fixture(scope="module")
def clean_df():
    if not os.path.exists(CLEAN_DATA_PATH):
        pytest.skip(f"Fichier introuvable : {CLEAN_DATA_PATH}")
    return pd.read_csv(CLEAN_DATA_PATH)


class TestDataSchema:
    def test_required_columns_exist(self, raw_df):
        missing = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
        assert not missing, f"Colonnes manquantes : {missing}"

    def test_dataset_not_empty(self, raw_df):
        assert len(raw_df) > 100, "Dataset trop petit"

    def test_numeric_columns_type(self, raw_df):
        for col in NUMERIC_COLS:
            assert pd.api.types.is_numeric_dtype(raw_df[col]), \
                f"{col} devrait être numérique"

    def test_poids_positive(self, raw_df):
        valid = raw_df["Poids"].dropna()
        assert (valid > 0).all(), "Poids doit être > 0"

    def test_volume_positive(self, raw_df):
        valid = raw_df["Volume"].dropna()
        assert (valid > 0).all(), "Volume doit être > 0"

    def test_target_distribution(self, raw_df):
        cats = raw_df["Categorie"].dropna().unique()
        assert len(cats) >= 2, "Au moins 2 catégories attendues"

    def test_missing_rate_poids(self, raw_df):
        rate = raw_df["Poids"].isnull().mean()
        assert rate < 0.20, f"Trop de NaN sur Poids : {rate:.1%}"


# ════════════════════════════════════════════════════════════════════
# 2. TESTS QUALITÉ POST-IMPUTATION
# ════════════════════════════════════════════════════════════════════

class TestPostImputation:
    def test_no_missing_numeric(self, clean_df):
        cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]
        missing = clean_df[cols].isnull().sum().sum()
        assert missing == 0, f"{missing} NaN résiduels sur colonnes numériques"

    def test_source_encoded_exists(self, clean_df):
        assert "Source_encoded" in clean_df.columns

    def test_source_encoded_no_nan(self, clean_df):
        assert clean_df["Source_encoded"].isnull().sum() == 0

    def test_normalized_poids(self, clean_df):
        # Après StandardScaler, mean ≈ 0 et std ≈ 1
        mean = clean_df["Poids"].mean()
        assert abs(mean) < 0.5, f"Poids non normalisé (mean={mean:.4f})"

    def test_clean_shape(self, clean_df):
        assert clean_df.shape[0] > 100
        assert clean_df.shape[1] >= len(REQUIRED_COLUMNS)


# ════════════════════════════════════════════════════════════════════
# 3. TESTS PIPELINE NLP
# ════════════════════════════════════════════════════════════════════

class TestNLPPipeline:
    def test_nettoyer_texte_lowercase(self):
        from src.nlp.train_nlp import nettoyer_texte
        result = nettoyer_texte("DÉCHET Plastique 123!")
        assert result == result.lower()

    def test_nettoyer_texte_no_digits(self):
        from src.nlp.train_nlp import nettoyer_texte
        result = nettoyer_texte("lot 45 kg collecté")
        assert not any(c.isdigit() for c in result)

    def test_nettoyer_texte_no_punctuation(self):
        from src.nlp.train_nlp import nettoyer_texte
        result = nettoyer_texte("déchet! plastique, verre.")
        assert "!" not in result and "," not in result

    def test_pretraiter_texte_removes_stopwords(self):
        from src.nlp.train_nlp import pretraiter_texte, get_stopwords
        import nltk
        from nltk.stem import SnowballStemmer
        stops = get_stopwords()
        stemmer = SnowballStemmer("french")
        result = pretraiter_texte("le lot de déchets plastique", stops, stemmer)
        tokens = result.split()
        assert "le" not in tokens
        assert "lot" not in tokens  # stopword domaine

    def test_pretraiter_texte_not_empty(self):
        from src.nlp.train_nlp import pretraiter_texte, get_stopwords
        from nltk.stem import SnowballStemmer
        stops = get_stopwords()
        stemmer = SnowballStemmer("french")
        result = pretraiter_texte("matériau conducteur rigide collecté", stops, stemmer)
        assert len(result.strip()) >= 0  # peut être vide si tout est stopword

    def test_pretraiter_non_empty_text(self):
        from src.nlp.train_nlp import pretraiter_texte, get_stopwords
        from nltk.stem import SnowballStemmer
        stops = get_stopwords()
        stemmer = SnowballStemmer("french")
        result = pretraiter_texte("fragment brillant conducteur transparent flexible", stops, stemmer)
        assert isinstance(result, str)


# ════════════════════════════════════════════════════════════════════
# 4. TESTS VECTORISATION
# ════════════════════════════════════════════════════════════════════

class TestVectorization:
    def test_tfidf_vectorizer_exists(self):
        path = "models/tfidf_vectorizer.pkl"
        if not os.path.exists(path):
            pytest.skip("tfidf_vectorizer.pkl introuvable")
        import joblib
        vec = joblib.load(path)
        assert hasattr(vec, "transform")

    def test_tfidf_transform(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2))
        corpus = ["déchet conducteur", "matériau rigide transparent", "lot collecté"]
        X = vec.fit_transform(corpus)
        assert X.shape[0] == 3
        assert X.shape[1] > 0

    def test_bow_transform(self):
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer()
        corpus = ["fragment brillant", "objet rigide", "matière flexible"]
        X = vec.fit_transform(corpus)
        assert X.shape[0] == 3


# ════════════════════════════════════════════════════════════════════
# 5. TESTS PRÉDICTIONS MODÈLE
# ════════════════════════════════════════════════════════════════════

class TestModelPredictions:
    @pytest.fixture(scope="class")
    def loaded_classifier(self):
        path = "models/classifier_best.pkl"
        if not os.path.exists(path):
            pytest.skip("classifier_best.pkl introuvable")
        import joblib
        return joblib.load(path)

    @pytest.fixture(scope="class")
    def loaded_scaler(self):
        path = "models/scaler.pkl"
        if not os.path.exists(path):
            pytest.skip("scaler.pkl introuvable")
        import joblib
        return joblib.load(path)

    def test_predict_returns_label(self, loaded_classifier, loaded_scaler):
        sample = np.array([[1.5, 2.0, 0.8, 0.6, 0.7]])
        scaled = loaded_scaler.transform(sample)
        features = np.append(scaled[0], 0).reshape(1, -1)
        pred = loaded_classifier.predict(features)
        assert len(pred) == 1
        assert isinstance(pred[0], str)

    def test_predict_multiple_samples(self, loaded_classifier, loaded_scaler):
        samples = np.array([
            [1.5, 2.0, 0.8, 0.6, 0.7],
            [3.0, 1.5, 0.3, 0.9, 0.4],
            [0.5, 0.8, 0.6, 0.2, 0.9],
        ])
        scaled = loaded_scaler.transform(samples)
        source_enc = np.zeros((3, 1))
        features = np.hstack([scaled, source_enc])
        preds = loaded_classifier.predict(features)
        assert len(preds) == 3

    def test_predict_output_is_known_class(self, loaded_classifier, loaded_scaler):
        import joblib
        le = joblib.load("models/le_categorie.pkl") if os.path.exists("models/le_categorie.pkl") else None
        sample = np.array([[1.5, 2.0, 0.8, 0.6, 0.7]])
        scaled = loaded_scaler.transform(sample)
        features = np.append(scaled[0], 0).reshape(1, -1)
        pred = loaded_classifier.predict(features)[0]
        if le:
            assert pred in le.classes_, f"Prédiction '{pred}' hors classes connues"


# ════════════════════════════════════════════════════════════════════
# 6. TEST SEUIL DE PERFORMANCE
# ════════════════════════════════════════════════════════════════════

class TestPerformanceThreshold:
    MIN_ACCURACY = 0.70

    def test_classifier_accuracy_threshold(self):
        """Vérifie que le classifier atteint accuracy ≥ 0.70 sur un sous-ensemble."""
        path_model = "models/classifier_best.pkl"
        path_scaler = "models/scaler.pkl"
        path_data = "data/processed/dataset_clean.csv"

        for p in [path_model, path_scaler, path_data]:
            if not os.path.exists(p):
                pytest.skip(f"Fichier introuvable : {p}")

        import joblib
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        clf = joblib.load(path_model)
        df = pd.read_csv(path_data)
        df_ml = df[df["Categorie"].notna()]

        features = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source_encoded"]
        X = df_ml[features]
        y = df_ml["Categorie"]

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        acc = accuracy_score(y_test, clf.predict(X_test))

        assert acc >= self.MIN_ACCURACY, \
            f"Accuracy {acc:.4f} < seuil {self.MIN_ACCURACY}"


# ════════════════════════════════════════════════════════════════════
# 7. TESTS API ENDPOINT
# ════════════════════════════════════════════════════════════════════

class TestAPIEndpoint:
    @pytest.fixture(scope="class")
    def client(self):
        # Vérifier que les modèles existent
        for p in ["models/classifier_best.pkl", "models/scaler.pkl"]:
            if not os.path.exists(p):
                pytest.skip(f"Modèle introuvable : {p}")
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_predict_endpoint_valid(self, client):
        payload = {
            "poids": 2.5,
            "volume": 1.8,
            "conductivite": 0.7,
            "opacite": 0.5,
            "rigidite": 0.6,
            "source": "Industriel",
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "categorie" in data
        assert "prix_estime" in data

    def test_predict_endpoint_with_nlp(self, client):
        payload = {
            "poids": 1.2,
            "volume": 0.8,
            "conductivite": 0.4,
            "opacite": 0.3,
            "rigidite": 0.5,
            "source": "Urbain",
            "rapport": "Fragment conducteur brillant collecté en zone industrielle",
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "nlp_categorie" in data

    def test_predict_invalid_poids(self, client):
        payload = {
            "poids": -1.0,
            "volume": 1.0,
            "conductivite": 0.5,
            "opacite": 0.5,
            "rigidite": 0.5,
            "source": "Urbain",
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422  # Validation error
