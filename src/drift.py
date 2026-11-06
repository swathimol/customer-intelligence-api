"""Drift report: PSI + KS per feature, plus predicted-vs-actual rate.

Usage:
    python src/drift.py --data data --models models --out reports
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(ref, new, bins=10):
    """Population Stability Index, bins fixed on the reference data."""
    ref, new = ref.dropna(), new.dropna()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                      # near-constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    n = np.clip(np.histogram(new, edges)[0] / len(new), 1e-4, None)
    return float(np.sum((n - r) * np.log(n / r)))


def flag(x):
    return "MAJOR" if x > 0.25 else ("moderate" if x > 0.10 else "stable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    data, models, out = Path(a.data), Path(a.models), Path(a.out)
    out.mkdir(exist_ok=True)

    features = json.load(open(models / "features.json"))
    model = joblib.load(models / "best_model.pkl")

    ref = pd.concat([pd.read_parquet(data / "feat_train_1.parquet"),
                     pd.read_parquet(data / "feat_train_2.parquet")], ignore_index=True)
    others = {n: pd.read_parquet(data / f"feat_{n}.parquet")
              for n in ["val", "test", "drift_late"]}

    # per-feature PSI and KS
    rows = []
    for f in features:
        row = {"feature": f}
        for name, d in others.items():
            row[f"psi_{name}"] = psi(ref[f], d[f])
            row[f"ks_{name}"] = ks_2samp(ref[f].dropna(), d[f].dropna()).statistic
        rows.append(row)
    rep = pd.DataFrame(rows).set_index("feature").sort_values("psi_test", ascending=False)
    rep["flag_test"] = rep["psi_test"].map(flag)
    rep.round(4).to_csv(out / "drift_report.csv")

    # score vs actual rate (the calibration drift)
    ref["score"] = model.predict_proba(ref[features])[:, 1]
    lines = [f"train       mean score={ref['score'].mean():.3f}  actual={ref['label'].mean():.3f}"]
    for name, d in others.items():
        d["score"] = model.predict_proba(d[features])[:, 1]
        gap = d["score"].mean() - d["label"].mean()
        lines.append(f"{name:11s} mean score={d['score'].mean():.3f}  "
                     f"actual={d['label'].mean():.3f}  gap={gap:+.3f}  "
                     f"score PSI={psi(ref['score'], d['score']):.3f}")

    print(rep[["psi_val", "psi_test", "psi_drift_late", "flag_test"]].round(3).head(10))
    print("\nFlag counts (test):\n", rep["flag_test"].value_counts().to_string())
    print("\n" + "\n".join(lines))

    worst_gap = max(float(d["score"].mean() - d["label"].mean()) for d in others.values())
    if worst_gap > 0.10:
        print("\nACTION: predicted-actual gap > 0.10 -> retrain on recent snapshots")
    elif worst_gap > 0.05:
        print("\nACTION: predicted-actual gap > 0.05 -> recalibrate (Platt on latest labels)")


if __name__ == "__main__":
    main()