"""
Module 2 — Modélisation supervisée + MLflow tracking
Extrait du notebook 02_ML_Supervise.ipynb
"""

import os
import warnings
import argparse
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

FEATURES = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source_encoded"]
TARGET   = "Categorie"
MIN_ACCURACY = 0.70


# ══════════════════════════════════════════════════════════════════════
# CHARGEMENT & SPLIT
# ══════════════════════════════════════════════════════════════════════

def load_data(path: str):
    df = pd.read_csv(path)
    df_ml = df[df[TARGET].notna()].copy()
    X = df_ml[FEATURES]
    y = df_ml[TARGET]
    print(f"[data] {len(df_ml)} lignes labellisées | {X.shape[1]} features")
    return X, y, df_ml


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"[split] Train:{len(X_train)} | Val:{len(X_val)} | Test:{len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════
# CLASSIFICATION — BASELINE
# ══════════════════════════════════════════════════════════════════════

def train_baseline_models(X_train, X_val, y_train, y_val, experiment_name: str):
    """Entraîne plusieurs modèles baseline et les logue dans MLflow."""
    mlflow.set_experiment(experiment_name)

    modeles = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
        "SVM":                 SVC(random_state=42),
        "KNN":                 KNeighborsClassifier(),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
    }

    resultats = {}
    print("\n=== Baseline Models ===")
    for nom, modele in modeles.items():
        with mlflow.start_run(run_name=f"baseline_{nom.replace(' ', '_')}"):
            modele.fit(X_train, y_train)
            y_pred = modele.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1  = f1_score(y_val, y_pred, average="weighted")

            mlflow.log_param("model", nom)
            mlflow.log_param("features", str(FEATURES))
            mlflow.log_metric("accuracy_val", acc)
            mlflow.log_metric("f1_val", f1)
            mlflow.sklearn.log_model(modele, artifact_path="model")

            resultats[nom] = {"Accuracy": acc, "F1": f1, "model": modele}
            print(f"  {nom:25s} → Acc: {acc:.4f} | F1: {f1:.4f}")

    return resultats


# ══════════════════════════════════════════════════════════════════════
# CLASSIFICATION — TUNING RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════

def tune_random_forest(X_train, X_val, y_train, y_val, experiment_name: str):
    """GridSearchCV sur Random Forest avec MLflow."""
    mlflow.set_experiment(experiment_name)

    param_grid = {
        "n_estimators":      [100, 200],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf":  [1, 2],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best = grid_search.best_estimator_

    y_pred_val = best.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)
    f1_val  = f1_score(y_val, y_pred_val, average="weighted")
    cv_mean = grid_search.best_score_

    with mlflow.start_run(run_name="RF_GridSearchCV_tuned"):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy_val", acc_val)
        mlflow.log_metric("f1_val", f1_val)
        mlflow.log_metric("cv_accuracy_mean", cv_mean)
        mlflow.sklearn.log_model(
            best,
            artifact_path="model",
            registered_model_name="EcoSmartClassifier",
        )

    print(f"\n[tuning] Best params : {grid_search.best_params_}")
    print(f"[tuning] Val Acc: {acc_val:.4f} | CV: {cv_mean:.4f}")
    return best, acc_val


# ══════════════════════════════════════════════════════════════════════
# CLASSIFICATION — ÉVALUATION FINALE (TEST SET)
# ══════════════════════════════════════════════════════════════════════

def evaluate_final_classifier(model, X_test, y_test, experiment_name: str):
    mlflow.set_experiment(experiment_name)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    with mlflow.start_run(run_name="final_test_evaluation"):
        mlflow.log_metric("accuracy_test", acc)
        mlflow.log_metric("f1_test", f1)
        mlflow.log_param("split", "test")

    print(f"\n[test] Accuracy : {acc:.4f} | F1 : {f1:.4f}")
    print(classification_report(y_test, y_pred))

    if acc >= MIN_ACCURACY:
        print(f"[✓] Seuil minimum {MIN_ACCURACY} atteint ({acc:.4f})")
    else:
        print(f"[✗] Seuil minimum {MIN_ACCURACY} NON atteint ({acc:.4f}) !")

    # Sauvegarde métriques JSON
    os.makedirs("reports", exist_ok=True)
    import json
    with open("reports/metrics_classification.json", "w") as f:
        json.dump({"accuracy_test": round(acc, 4), "f1_test": round(f1, 4)}, f, indent=2)

    return acc, f1


# ══════════════════════════════════════════════════════════════════════
# RÉGRESSION — PRÉDICTION PRIX
# ══════════════════════════════════════════════════════════════════════

def train_regression(df_ml, experiment_name: str):
    """Entraîne plusieurs régresseurs pour prédire Prix_Revente."""
    mlflow.set_experiment(experiment_name)

    X_reg = df_ml[FEATURES]
    y_reg = df_ml["Prix_Revente"]

    X_train_r, X_temp_r, y_train_r, y_temp_r = train_test_split(
        X_reg, y_reg, test_size=0.30, random_state=42
    )
    X_val_r, X_test_r, y_val_r, y_test_r = train_test_split(
        X_temp_r, y_temp_r, test_size=0.50, random_state=42
    )

    modeles_reg = {
        "Linear Regression": LinearRegression(),
        "Ridge":             Ridge(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42) if False
                             else __import__("sklearn.ensemble", fromlist=["GradientBoostingRegressor"]).GradientBoostingRegressor(random_state=42),
        "SVR":               SVR(),
    }

    resultats_reg = {}
    print("\n=== Modèles Régression ===")
    for nom, modele in modeles_reg.items():
        with mlflow.start_run(run_name=f"reg_{nom.replace(' ', '_')}"):
            modele.fit(X_train_r, y_train_r)
            y_pred = modele.predict(X_test_r)
            rmse = float(np.sqrt(mean_squared_error(y_test_r, y_pred)))
            r2   = float(r2_score(y_test_r, y_pred))

            mlflow.log_param("model", nom)
            mlflow.log_metric("rmse_test", rmse)
            mlflow.log_metric("r2_test", r2)

            resultats_reg[nom] = {"RMSE": rmse, "R2": r2, "model": modele}
            print(f"  {nom:25s} → RMSE: {rmse:.4f} | R²: {r2:.4f}")

    # Meilleur régresseur
    best_reg_nom = max(resultats_reg, key=lambda k: resultats_reg[k]["R2"])
    best_reg = resultats_reg[best_reg_nom]["model"]
    print(f"\n[regression] Meilleur modèle : {best_reg_nom} (R²={resultats_reg[best_reg_nom]['R2']:.4f})")
    return best_reg


# ══════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def train(data_path: str, models_dir: str = "models",
          experiment: str = "EcoSmart_Classification"):
    os.makedirs(models_dir, exist_ok=True)

    X, y, df_ml = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Classification
    resultats = train_baseline_models(X_train, X_val, y_train, y_val, experiment)
    print("\n=== Tuning Random Forest ===")
    best_clf, acc_val = tune_random_forest(X_train, X_val, y_train, y_val, experiment)
    print("\n=== Évaluation Finale Classification ===")
    acc_test, f1_test = evaluate_final_classifier(best_clf, X_test, y_test, experiment)

    # Régression
    best_reg = train_regression(df_ml, experiment)

    # Sauvegarde
    joblib.dump(best_clf, os.path.join(models_dir, "classifier_best.pkl"))
    joblib.dump(best_reg, os.path.join(models_dir, "regressor_best.pkl"))
    print(f"\n[saved] classifier_best.pkl + regressor_best.pkl → {models_dir}/")

    return best_clf, best_reg, acc_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/processed/dataset_clean.csv")
    parser.add_argument("--models",     default="models")
    parser.add_argument("--experiment", default="EcoSmart_Classification")
    args = parser.parse_args()
    train(args.data, args.models, args.experiment)
