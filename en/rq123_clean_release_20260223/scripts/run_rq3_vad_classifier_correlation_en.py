#!/usr/bin/env python3
"""
Compare VAD+classifier auto CAF against manual CAF (Gold-40).

Outputs:
  - rq3_vad_classifier_correlation_summary.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
ANNOT = BASE / "annotation"
SELECTED = ANNOT / "selected_files.json"

CAF_MEASURES = ["AR", "SR", "MLR", "MCPR", "ECPR", "PR", "MCPD", "ECPD", "MPD"]


def icc_2_1(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    k = 2
    data = np.column_stack([x, y])
    gm = data.mean()
    rm = data.mean(axis=1)
    cm = data.mean(axis=0)
    ssr = k * np.sum((rm - gm) ** 2)
    ssc = n * np.sum((cm - gm) ** 2)
    sst = np.sum((data - gm) ** 2)
    sse = sst - ssr - ssc
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    if abs(denom) < 1e-15:
        return 1.0
    return float((msr - mse) / denom)


def with_file_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["file_id"] = out["file"].str.replace(".TextGrid", "", regex=False)
    out["task"] = np.where(out["file_id"].str.contains("ST1"), "ST1", "ST2")
    return out


def compute_rows(merged: pd.DataFrame, track: str, subset: str) -> list[dict]:
    rows = []
    for m in CAF_MEASURES:
        a = merged[f"{m}_auto"].to_numpy(dtype=float)
        b = merged[f"{m}_manual"].to_numpy(dtype=float)
        r, rp = stats.pearsonr(a, b)
        rho, rhop = stats.spearmanr(a, b)
        diff = a - b
        rows.append(
            {
                "track": track,
                "subset": subset,
                "measure": m,
                "n": len(merged),
                "mean_auto": round(float(np.mean(a)), 4),
                "mean_manual": round(float(np.mean(b)), 4),
                "pearson_r": round(float(r), 3),
                "pearson_p": float(rp),
                "spearman_rho": round(float(rho), 3),
                "spearman_p": float(rhop),
                "icc_2_1": round(float(icc_2_1(a, b)), 3),
                "bias": round(float(np.mean(diff)), 4),
                "mae": round(float(np.mean(np.abs(diff))), 4),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto-clf-csv", required=True, help="Classifier-guided auto CAF CSV")
    ap.add_argument("--manual-csv", required=True, help="Manual CAF reference CSV")
    ap.add_argument("--exclude", nargs="*", default=[], help="File IDs to exclude from correlation")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    gold40 = set(selected["all_selected"]) - set(args.exclude)

    auto_clf = with_file_id(pd.read_csv(args.auto_clf_csv))
    manual = with_file_id(pd.read_csv(args.manual_csv))

    manual_gold = manual[manual["file_id"].isin(gold40)].copy()
    clf_gold = auto_clf[auto_clf["file_id"].isin(gold40)].copy()

    merged = pd.merge(clf_gold, manual_gold, on="file_id", suffixes=("_auto", "_manual"))
    rows = []
    rows.extend(compute_rows(merged, "vad_classifier_gold40", "Overall"))
    rows.extend(compute_rows(merged[merged["task_auto"] == "ST1"], "vad_classifier_gold40", "ST1"))
    rows.extend(compute_rows(merged[merged["task_auto"] == "ST2"], "vad_classifier_gold40", "ST2"))

    summary = pd.DataFrame(rows)
    summary_out = out_dir / "rq3_vad_classifier_correlation_summary.csv"
    summary.to_csv(summary_out, index=False)

    print("=" * 60)
    print("RQ3 CORRELATION (Gold-40): VAD+classifier vs manual")
    print("=" * 60)
    for subset in ["Overall", "ST1", "ST2"]:
        c = summary[(summary["subset"] == subset) & (summary["measure"] == "MCPD")].iloc[0]
        print(
            f"{subset}: MCPD r={c['pearson_r']:.3f}, "
            f"ICC={c['icc_2_1']:.3f}, "
            f"MAE={c['mae']:.3f}"
        )
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()
