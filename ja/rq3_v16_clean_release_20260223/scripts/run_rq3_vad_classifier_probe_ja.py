#!/usr/bin/env python
"""
JA RQ3 probe:
Compare baseline auto CAF, prior VAD auto CAF, and new classifier-hybrid auto CAF
against manual CAF.
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


def build_track(auto_df: pd.DataFrame, manual_df: pd.DataFrame, track_name: str) -> tuple[pd.DataFrame, list[dict]]:
    merged = pd.merge(auto_df, manual_df, on="file_id", suffixes=("_auto", "_manual"))
    merged["task"] = merged["file_id"].apply(lambda x: "ST1" if "ST1" in x else ("ST2" if "ST2" in x else "OTHER"))
    rows: list[dict] = []
    rows.extend(compute_rows(merged, track_name, "Overall"))
    rows.extend(compute_rows(merged[merged["task"] == "ST1"], track_name, "ST1"))
    rows.extend(compute_rows(merged[merged["task"] == "ST2"], track_name, "ST2"))
    return merged, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-auto-csv", type=Path, required=True)
    ap.add_argument("--vad-auto-csv", type=Path, required=True)
    ap.add_argument("--clf-auto-csv", type=Path, required=True)
    ap.add_argument("--manual-csv", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    ap.add_argument("--out-mcpd-deltas", type=Path, required=True)
    args = ap.parse_args()

    b = pd.read_csv(args.baseline_auto_csv)
    v = pd.read_csv(args.vad_auto_csv)
    c = pd.read_csv(args.clf_auto_csv)
    m = pd.read_csv(args.manual_csv)

    for df in [b, v, c, m]:
        if "error" in df.columns:
            df = df[df["error"].isna()]

    b["file_id"] = b["file"].apply(file_id_from_name)
    v["file_id"] = v["file"].apply(file_id_from_name)
    c["file_id"] = c["file"].apply(file_id_from_name)
    m["file_id"] = m["file"].apply(file_id_from_name)

    mb, rb = build_track(b, m, "baseline")
    mv, rv = build_track(v, m, "vad")
    mc, rc = build_track(c, m, "vad_classifier")
    summary = pd.DataFrame(rb + rv + rc)

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_mcpd_deltas.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)

    mcpd = pd.merge(
        mb[["file_id", "task", "MCPD_auto", "MCPD_manual"]].rename(columns={"MCPD_auto": "MCPD_auto_baseline"}),
        mv[["file_id", "MCPD_auto"]].rename(columns={"MCPD_auto": "MCPD_auto_vad"}),
        on="file_id",
        how="inner",
    )
    mcpd = pd.merge(
        mcpd,
        mc[["file_id", "MCPD_auto"]].rename(columns={"MCPD_auto": "MCPD_auto_clf"}),
        on="file_id",
        how="inner",
    )
    mcpd["abs_err_baseline"] = (mcpd["MCPD_auto_baseline"] - mcpd["MCPD_manual"]).abs()
    mcpd["abs_err_vad"] = (mcpd["MCPD_auto_vad"] - mcpd["MCPD_manual"]).abs()
    mcpd["abs_err_clf"] = (mcpd["MCPD_auto_clf"] - mcpd["MCPD_manual"]).abs()
    mcpd["delta_vad_minus_base"] = mcpd["abs_err_vad"] - mcpd["abs_err_baseline"]
    mcpd["delta_clf_minus_base"] = mcpd["abs_err_clf"] - mcpd["abs_err_baseline"]
    mcpd["delta_clf_minus_vad"] = mcpd["abs_err_clf"] - mcpd["abs_err_vad"]
    mcpd.to_csv(args.out_mcpd_deltas, index=False)

    for subset in ["Overall", "ST1", "ST2"]:
        x = summary[(summary["track"] == "baseline") & (summary["subset"] == subset) & (summary["measure"] == "MCPD")].iloc[0]
        y = summary[(summary["track"] == "vad") & (summary["subset"] == subset) & (summary["measure"] == "MCPD")].iloc[0]
        z = summary[(summary["track"] == "vad_classifier") & (summary["subset"] == subset) & (summary["measure"] == "MCPD")].iloc[0]
        print(
            f"{subset} MCPD: r {x['pearson_r']:.3f}->{y['pearson_r']:.3f}->{z['pearson_r']:.3f}, "
            f"ICC {x['icc_2_1']:.3f}->{y['icc_2_1']:.3f}->{z['icc_2_1']:.3f}, "
            f"MAE {x['mae']:.3f}->{y['mae']:.3f}->{z['mae']:.3f}"
        )

    print("Saved:", args.out_summary)
    print("Saved:", args.out_mcpd_deltas)


if __name__ == "__main__":
    main()

