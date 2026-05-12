"""
Module 1 — Nettoyage et préparation des données
Extrait du notebook 01_EDA_Nettoyage.ipynb
"""

import os
import warnings
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] Shape : {df.shape}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"[dedup] {before - len(df)} doublons supprimés")
    return df


def remove_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["Poids"] > 0]
    df = df[df["Volume"] > 0]
    df = df[df["Prix_Revente"] >= 0]
    print(f"[clean] {before - len(df)} lignes avec valeurs impossibles supprimées")
    return df


def remove_outliers_iqr(df: pd.DataFrame, colonnes: list, factor: float = 3.0) -> pd.DataFrame:
    for col in colonnes:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        before = len(df)
        df = df[(df[col] >= Q1 - factor * IQR) & (df[col] <= Q3 + factor * IQR)]
        print(f"[outliers] {col} : {before - len(df)} supprimés")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # MAR → KNN (colonnes corrélées)
    cols_knn = ["Poids", "Volume", "Rigidite"]
    imputer_knn = KNNImputer(n_neighbors=5)
    df[cols_knn] = imputer_knn.fit_transform(df[cols_knn])

    # MCAR → Médiane
    cols_mediane = ["Conductivite", "Opacite", "Prix_Revente"]
    imputer_mediane = SimpleImputer(strategy="median")
    df[cols_mediane] = imputer_mediane.fit_transform(df[cols_mediane])

    # Source → Mode
    df["Source"].fillna(df["Source"].mode()[0], inplace=True)

    print(f"[impute] Valeurs manquantes restantes : {df.isnull().sum().sum()}")
    return df


def encode_features(df: pd.DataFrame):
    le_source = LabelEncoder()
    df["Source_encoded"] = le_source.fit_transform(df["Source"])

    le_categorie = LabelEncoder()
    mask = df["Categorie"].notna()
    df.loc[mask, "Categorie_encoded"] = le_categorie.fit_transform(df.loc[mask, "Categorie"])

    print(f"[encode] Source classes : {le_source.classes_}")
    print(f"[encode] Categorie classes : {le_categorie.classes_}")
    return df, le_source, le_categorie


def normalize_features(df: pd.DataFrame):
    cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    print(f"[normalize] Normalisation appliquée sur {cols}")
    return df, scaler


def preprocess(input_path: str, output_path: str, models_dir: str = "models"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = load_data(input_path)
    df = remove_duplicates(df)
    df = remove_impossible_values(df)
    df = remove_outliers_iqr(df, ["Poids", "Volume", "Prix_Revente"])
    df = impute_missing(df)
    df, le_source, le_categorie = encode_features(df)
    df, scaler = normalize_features(df)

    df.to_csv(output_path, index=False)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(le_source, os.path.join(models_dir, "le_source.pkl"))
    joblib.dump(le_categorie, os.path.join(models_dir, "le_categorie.pkl"))

    print(f"\n[done] Dataset nettoyé sauvegardé → {output_path}")
    print(f"[done] Shape finale : {df.shape}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw/dataset_ProjetML_2026.csv")
    parser.add_argument("--output", default="data/processed/dataset_clean.csv")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()
    preprocess(args.input, args.output, args.models)
