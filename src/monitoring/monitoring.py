"""
Module 6 -- Monitoring avec Evidently AI
Detection de data drift numerique + text drift (Jensen-Shannon)
"""

import os
import sys
import json
import warnings
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer

# Fix encodage Windows PowerShell
sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")

NUMERIC_FEATURES = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]
REPORTS_DIR = "reports"


# -- Drift numerique via Evidently --------------------------------------------

def run_numeric_drift_report(ref_path: str, cur_path: str, output_dir: str = REPORTS_DIR):
    """Genere un rapport Evidently de data drift numerique."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently.metrics import DatasetDriftMetric

        ref = pd.read_csv(ref_path)[NUMERIC_FEATURES].dropna()
        cur = pd.read_csv(cur_path)[NUMERIC_FEATURES].dropna()

        report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
        report.run(reference_data=ref, current_data=cur)

        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, "drift_report.html")
        json_path = os.path.join(output_dir, "drift_report.json")

        report.save_html(html_path)
        report.save_json(json_path)

        print(f"[evidently] Rapport sauvegarde -> {html_path}")
        print(f"[evidently] JSON sauvegarde    -> {json_path}")

        with open(json_path) as f:
            data = json.load(f)
        _print_drift_summary(data)
        return data

    except ImportError:
        print("[evidently] Evidently non installe -- utilisation du mode manuel")
        return run_manual_drift(ref_path, cur_path, output_dir)


def _print_drift_summary(data: dict):
    try:
        metrics = data.get("metrics", [])
        for m in metrics:
            if m.get("metric") == "DatasetDriftMetric":
                result = m["result"]
                n_drifted = result.get("number_of_drifted_columns", 0)
                n_total   = result.get("number_of_columns", 0)
                drift     = result.get("dataset_drift", False)
                print(f"\n[drift] Dataset drift detecte : {'OUI [!]' if drift else 'NON [OK]'}")
                print(f"[drift] Colonnes driftees : {n_drifted}/{n_total}")
    except Exception:
        pass


# -- Drift numerique manuel (sans Evidently) ----------------------------------

def run_manual_drift(ref_path: str, cur_path: str, output_dir: str = REPORTS_DIR) -> dict:
    """Calcule le drift colonne par colonne via Jensen-Shannon (distributions)."""
    ref = pd.read_csv(ref_path)[NUMERIC_FEATURES].dropna()
    cur = pd.read_csv(cur_path)[NUMERIC_FEATURES].dropna()

    results = {}
    DRIFT_THRESHOLD = 0.1

    print("\n=== RAPPORT DRIFT NUMERIQUE (Manuel) ===")
    print(f"{'Colonne':<20} {'JS Divergence':>15} {'Drift':>10}")
    print("-" * 50)

    for col in NUMERIC_FEATURES:
        bins = np.linspace(
            min(ref[col].min(), cur[col].min()),
            max(ref[col].max(), cur[col].max()),
            50
        )
        ref_hist, _ = np.histogram(ref[col], bins=bins, density=True)
        cur_hist, _ = np.histogram(cur[col], bins=bins, density=True)

        ref_hist = ref_hist + 1e-10
        cur_hist = cur_hist + 1e-10
        ref_hist /= ref_hist.sum()
        cur_hist /= cur_hist.sum()

        js = float(jensenshannon(ref_hist, cur_hist))
        drifted = js > DRIFT_THRESHOLD

        results[col] = {"js_divergence": round(js, 4), "drifted": drifted}
        flag = "[DRIFT]" if drifted else "[OK]"
        print(f"{col:<20} {js:>15.4f} {flag:>10}")

    n_drifted = sum(v["drifted"] for v in results.values())
    print(f"\n[resume] {n_drifted}/{len(NUMERIC_FEATURES)} colonnes avec drift")

    os.makedirs(output_dir, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "reference": ref_path,
        "current": cur_path,
        "threshold": DRIFT_THRESHOLD,
        "columns": results,
        "n_drifted": n_drifted,
        "dataset_drift": n_drifted > len(NUMERIC_FEATURES) // 2,
    }
    out_path = os.path.join(output_dir, "drift_manual.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[saved] {out_path}")
    return report


# -- Drift textuel (Jensen-Shannon sur TF-IDF) --------------------------------

def run_text_drift(ref_path: str, cur_path: str, output_dir: str = REPORTS_DIR) -> dict:
    """Calcule le text drift via Jensen-Shannon sur les distributions TF-IDF."""
    ref = pd.read_csv(ref_path)["Rapport_Collecte"].dropna().tolist()
    cur = pd.read_csv(cur_path)["Rapport_Collecte"].dropna().tolist()

    vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    vec.fit(ref + cur)

    ref_matrix = vec.transform(ref).toarray().mean(axis=0)
    cur_matrix = vec.transform(cur).toarray().mean(axis=0)

    ref_matrix = ref_matrix / (ref_matrix.sum() + 1e-10)
    cur_matrix = cur_matrix / (cur_matrix.sum() + 1e-10)

    js = float(jensenshannon(ref_matrix, cur_matrix))
    THRESHOLD = 0.15
    drifted = js > THRESHOLD

    print(f"\n[text drift] JS Divergence = {js:.4f}")
    print(f"[text drift] {'[DRIFT DETECTE]' if drifted else '[Pas de drift]'} (seuil={THRESHOLD})")

    result = {
        "timestamp": datetime.now().isoformat(),
        "js_divergence": round(js, 4),
        "threshold": THRESHOLD,
        "text_drift": drifted,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "text_drift.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")
    return result


# -- Alertes ------------------------------------------------------------------

def check_alerts(numeric_report: dict, text_report: dict = None):
    """Verifie les seuils et genere des alertes."""
    alerts = []

    n_drifted = numeric_report.get("n_drifted", 0)
    if n_drifted > 0:
        alerts.append(f"[!] {n_drifted} colonne(s) numerique(s) en drift")

    if text_report and text_report.get("text_drift"):
        alerts.append(f"[!] Text drift detecte (JS={text_report['js_divergence']:.4f})")

    if alerts:
        print("\n=== ALERTES ===")
        for a in alerts:
            print(f"  {a}")
    else:
        print("\n[OK] Aucune alerte -- donnees en production stables")

    log = {
        "timestamp": datetime.now().isoformat(),
        "alerts": alerts,
        "n_alerts": len(alerts),
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "alerts.json"), "w") as f:
        json.dump(log, f, indent=2)
    return alerts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitoring drift Eco-Smart")
    parser.add_argument("--reference", default="data/processed/dataset_clean.csv",
                        help="Dataset de reference (train)")
    parser.add_argument("--current",   default="data/processed/dataset_clean.csv",
                        help="Dataset courant (production)")
    parser.add_argument("--output",    default=REPORTS_DIR)
    parser.add_argument("--evidently", action="store_true",
                        help="Utiliser Evidently si disponible")
    args = parser.parse_args()

    print("=" * 60)
    print("  ECO-SMART CLASSIFIER -- MONITORING DRIFT")
    print("=" * 60)

    if args.evidently:
        numeric_report = run_numeric_drift_report(args.reference, args.current, args.output)
    else:
        numeric_report = run_manual_drift(args.reference, args.current, args.output)

    text_report = run_text_drift(args.reference, args.current, args.output)
    check_alerts(numeric_report, text_report)
