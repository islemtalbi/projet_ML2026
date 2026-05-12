"""
Module 6 — API REST FastAPI
Endpoint /predict pour la classification de déchets
"""

import os
import re
import joblib
import numpy as np
import nltk
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# NLTK downloads
for pkg in ["stopwords", "punkt", "wordnet", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# ── Chemins modèles ────────────────────────────────────────────────────────────
MODELS_DIR = os.getenv("MODELS_DIR", "models")


def load_model(name: str):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path)


# ── App FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Eco-Smart Classifier API",
    description="Classification de déchets et estimation de valeur",
    version="1.0.0",
)

# Chargement paresseux des modèles
_models = {}


def get_models():
    if not _models:
        _models["classifier"]      = load_model("classifier_best.pkl")
        _models["regressor"]       = load_model("regressor_best.pkl")
        _models["scaler"]          = load_model("scaler.pkl")
        _models["le_source"]       = load_model("le_source.pkl")
        _models["le_categorie"]    = load_model("le_categorie.pkl")
        _models["tfidf"]           = load_model("tfidf_vectorizer.pkl")
        _models["nlp_classifier"]  = load_model("nlp_classifier.pkl")
    return _models


# ── Schémas Pydantic ──────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    poids:       float = Field(..., gt=0, description="Poids en kg")
    volume:      float = Field(..., gt=0, description="Volume en litres")
    conductivite: float = Field(..., description="Conductivité")
    opacite:     float = Field(..., ge=0, le=1, description="Opacité [0-1]")
    rigidite:    float = Field(..., description="Rigidité")
    source:      str   = Field(..., description="Source du déchet")
    rapport:     Optional[str] = Field(None, description="Rapport collecte (NLP)")


class PredictionResponse(BaseModel):
    categorie:       str
    prix_estime:     float
    confidence:      Optional[float] = None
    nlp_categorie:   Optional[str]   = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


# ── Preprocessing NLP ─────────────────────────────────────────────────────────
STOP_DOMAINE = {
    "lot", "déchet", "collecté", "volume", "poids", "kg",
    "litre", "usine", "site", "matériau", "aspect",
    "papier", "plastique", "metal", "métal", "verre",
    "organique", "carton", "aluminium", "ferreux", "ferraille",
}

_stop_fr  = set(stopwords.words("french")).union(STOP_DOMAINE)
_stemmer  = SnowballStemmer("french")


def preprocess_text(texte: str) -> str:
    texte = texte.lower()
    texte = re.sub(r"[^a-zàâäéèêëîïôùûüç\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    tokens = word_tokenize(texte, language="french")
    tokens = [_stemmer.stem(t) for t in tokens if t not in _stop_fr]
    return " ".join(tokens)


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", response_model=dict)
def root():
    return {"message": "Eco-Smart Classifier API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        get_models()
        loaded = True
    except Exception:
        loaded = False
    return {"status": "ok" if loaded else "degraded", "models_loaded": loaded}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        m = get_models()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Encoder Source
    try:
        source_enc = m["le_source"].transform([req.source])[0]
    except ValueError:
        source_enc = 0  # classe inconnue → 0

    # Normaliser features numériques
    raw = np.array([[req.poids, req.volume, req.conductivite,
                     req.opacite, req.rigidite]])
    scaled = m["scaler"].transform(raw)
    features = np.append(scaled[0], source_enc).reshape(1, -1)

    # Classification
    categorie = m["classifier"].predict(features)[0]

    # Probabilité (si dispo)
    confidence = None
    if hasattr(m["classifier"], "predict_proba"):
        proba = m["classifier"].predict_proba(features)[0]
        confidence = float(np.max(proba))

    # Régression (prix estimé)
    prix = float(m["regressor"].predict(features)[0])

    # NLP (optionnel)
    nlp_categorie = None
    if req.rapport:
        texte_clean = preprocess_text(req.rapport)
        vec = m["tfidf"].transform([texte_clean])
        nlp_categorie = m["nlp_classifier"].predict(vec)[0]

    return PredictionResponse(
        categorie=str(categorie),
        prix_estime=round(prix, 2),
        confidence=round(confidence, 4) if confidence else None,
        nlp_categorie=str(nlp_categorie) if nlp_categorie else None,
    )


@app.get("/categories")
def get_categories():
    try:
        m = get_models()
        return {"categories": list(m["le_categorie"].classes_)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
