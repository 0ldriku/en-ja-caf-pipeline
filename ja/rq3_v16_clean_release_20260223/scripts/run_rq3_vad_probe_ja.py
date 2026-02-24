#!/usr/bin/env python
"""
JA VAD probe:
Compare baseline auto CAF vs VAD-refined auto CAF against manual CAF.
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
    return (msr - mse) / denom


def compute_stats(sub: pd.DataFrame, track: str, subset: str) -> list[dict]:
    rows = []
    for m in CAF_MEASURES:
        a = sub[f"{m}_auto"].astype(float).values
        b = sub[f"{m}_manual"].astype(float).values
        r, rp = stats.pearsonr(a, b)
        rho, rhop = stats.spearmanr(a, b)
        diff = a - b
        rows.append(
            {
                "track": track,
                "subset": subset,
                "measure": m,
                "n": len(sub),
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
    ap.add_argument("--baseline-auto-csv", type=Path, required=True)
    ap.add_argument("--vad-auto-csv", type=Path, required=True)
    ap.add_argument("--manual-csv", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    ap.add_argument("--out-mcpd-deltas", type=Path, required=True)
    args = ap.parse_args()

    b = pd.read_csv(args.baseline_auto_csv)
    v = pd.read_csv(args.vad_auto_csv)
    m = pd.read_csv(args.manual_csv)

    if "error" in b.columns:
        b = b[b["error"].isna()]
    if "error" in v.columns:
        v = v[v["error"].isna()]
    if "error" in m.columns:
        m = m[m["error"].isna()]

    b["file_id"] = b["file"].apply(file_id_from_name)
    v["file_id"] = v["file"].apply(file_id_from_name)
    m["file_id"] = m["file"].apply(file_id_from_name)

    mb = pd.merge(b, m, on="file_id", suffixes=("_auto", "_manual"))
    mv = pd.merge(v, m, on="file_id", suffixes=("_auto", "_manual"))
    mb["task"] = mb["file_id"].apply(lambda x: "ST1" if "ST1" in x else ("ST2" if "ST2" in x else "OTHER"))
    mv["task"] = mv["file_id"].apply(lambda x: "ST1" if "ST1" in x else ("ST2" if "ST2" in x else "OTHER"))

    rows = []
    rows.extend(compute_stats(mb, "baseline", "Overall"))
    rows.extend(compute_stats(mb[mb["task"] == "ST1"], "baseline", "ST1"))
    rows.extend(compute_stats(mb[mb["task"] == "ST2"], "baseline", "ST2"))
    rows.extend(compute_stats(mv, "vad", "Overall"))
    rows.extend(compute_stats(mv[mv["task"] == "ST1"], "vad", "ST1"))
    rows.extend(compute_stats(mv[mv["task"] == "ST2"], "vad", "ST2"))
    out = pd.DataFrame(rows)

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_summary, index=False)

    mcpd = pd.merge(
        mb[["file_id", "task", "MCPD_auto", "MCPD_manual"]],
        mv[["file_id", "MCPD_auto"]].rename(columns={"MCPD_auto": "MCPD_auto_vad"}),
        on="file_id",
        how="inner",
    ).rename(columns={"MCPD_auto": "MCPD_auto_baseline"})
    mcpd["abs_err_baseline"] = (mcpd["MCPD_auto_baseline"] - mcpd["MCPD_manual"]).abs()
    mcpd["abs_err_vad"] = (mcpd["MCPD_auto_vad"] - mcpd["MCPD_manual"]).abs()
    mcpd["abs_err_delta_vad_minus_base"] = mcpd["abs_err_vad"] - mcpd["abs_err_baseline"]
    args.out_mcpd_deltas.parent.mkdir(parents=True, exist_ok=True)
    mcpd.to_csv(args.out_mcpd_deltas, index=False)

    print("Matched files:", len(mb))
    for subset in ["Overall", "ST1", "ST2"]:
        rb = out[(out["track"] == "baseline") & (out["subset"] == subset) & (out["measure"] == "MCPD")].iloc[0]
        rv = out[(out["track"] == "vad") & (out["subset"] == subset) & (out["measure"] == "MCPD")].iloc[0]
        print(
            f"{subset} MCPD: r {rb['pearson_r']:.3f}->{rv['pearson_r']:.3f}, "
            f"ICC {rb['icc_2_1']:.3f}->{rv['icc_2_1']:.3f}, "
            f"MAE {rb['mae']:.3f}->{rv['mae']:.3f}"
        )
    print("Saved:", args.out_summary)
    print("Saved:", args.out_mcpd_deltas)


if __name__ == "__main__":
    main()
