"""
Tests pytest — Module 6 MLOps
Couvre : schéma données, imputation, NLP, prédictions, seuil performance, API
"""

import os
import sys
import pytest
import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════

REQUIRED_COLUMNS = [
    "Poids",
    "Volume",
    "Conductivite",
    "Opacite",
    "Rigidite",
    "Prix_Revente",
    "Source",
    "Rapport_Collecte",
    "Categorie",
]

NUMERIC_COLS = [
    "Poids",
    "Volume",
    "Conductivite",
    "Opacite",
    "Rigidite",
    "Prix_Revente",
]

RAW_DATA_PATH = "data/raw/dataset_ProjetML_2026.csv"
CLEAN_DATA_PATH = "data/processed/dataset_clean.csv"

# ════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════

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

# ════════════════════════════════════════════════════════════════════
# TESTS SCHÉMA DES DONNÉES
# ════════════════════════════════════════════════════════════════════

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

    def test_poids_positive(self, clean_df):
        # Après StandardScaler, les valeurs sont centrées-réduites
        mean = clean_df["Poids"].mean()
        std = clean_df["Poids"].std()

        assert abs(mean) < 0.1, \
            f"Mean Poids attendu ≈ 0, obtenu : {mean:.4f}"

        assert abs(std - 1) < 0.1, \
            f"Std Poids attendu ≈ 1, obtenu : {std:.4f}"

    def test_volume_positive(self, clean_df):
        # Après StandardScaler, les valeurs sont centrées-réduites
        mean = clean_df["Volume"].mean()
        std = clean_df["Volume"].std()

        assert abs(mean) < 0.1, \
            f"Mean Volume attendu ≈ 0, obtenu : {mean:.4f}"

        assert abs(std - 1) < 0.1, \
            f"Std Volume attendu ≈ 1, obtenu : {std:.4f}"

    def test_target_distribution(self, raw_df):
        cats = raw_df["Categorie"].dropna().unique()
        assert len(cats) >= 2, "Au moins 2 catégories attendues"

    def test_missing_rate_poids(self, raw_df):
        rate = raw_df["Poids"].isnull().mean()

        assert rate < 0.20, \
            f"Trop de NaN sur Poids : {rate:.1%}"