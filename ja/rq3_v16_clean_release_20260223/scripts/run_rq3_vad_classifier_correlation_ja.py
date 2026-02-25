#!/usr/bin/env python
"""
JA RQ3 correlation:
Compare VAD+classifier auto CAF against manual CAF.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CAF_MEASURES = ["AR", "SR", "MLR", "MCPR", "ECPR", "PR", "MCPD", "ECPD", "MPD"]


def file_id_from_name(fname: str) -> str:
    return re.sub(r"\.TextGrid$", "", str(fname), flags=re.IGNORECASE)


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


def compute_rows(merged: pd.DataFrame, track: str, subset: str) -> list[dict]:
    rows: list[dict] = []
    for m in CAF_MEASURES:
        a = merged[f"{m}_auto"].astype(float).values
        b = merged[f"{m}_manual"].astype(float).values
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
    ap.add_argument("--clf-auto-csv", type=Path, required=True)
    ap.add_argument("--manual-csv", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    args = ap.parse_args()

    c = pd.read_csv(args.clf_auto_csv)
    m = pd.read_csv(args.manual_csv)

    c["file_id"] = c["file"].apply(file_id_from_name)
    m["file_id"] = m["file"].apply(file_id_from_name)

    merged = pd.merge(c, m, on="file_id", suffixes=("_auto", "_manual"))
    merged["task"] = merged["file_id"].apply(lambda x: "ST1" if "ST1" in x else ("ST2" if "ST2" in x else "OTHER"))

    rows: list[dict] = []
    rows.extend(compute_rows(merged, "vad_classifier", "Overall"))
    rows.extend(compute_rows(merged[merged["task"] == "ST1"], "vad_classifier", "ST1"))
    rows.extend(compute_rows(merged[merged["task"] == "ST2"], "vad_classifier", "ST2"))
    summary = pd.DataFrame(rows)

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)

    for subset in ["Overall", "ST1", "ST2"]:
        z = summary[(summary["subset"] == subset) & (summary["measure"] == "MCPD")].iloc[0]
        print(
            f"{subset} MCPD: r={z['pearson_r']:.3f}, "
            f"ICC={z['icc_2_1']:.3f}, "
            f"MAE={z['mae']:.3f}"
        )

    print("Saved:", args.out_summary)


if __name__ == "__main__":
    main()
