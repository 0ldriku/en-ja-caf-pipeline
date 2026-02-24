#!/usr/bin/env python
"""
RQ3 JA correlation from precomputed CAF CSV files.

Direct workflow:
1) Build ASR clause TextGrids with ja/ja_clause_segmenter_v16.py
2) Compute CAF CSVs with ja/caf_calculator_ja.py for ASR and manual-gold
3) Run this script to compute correlation/agreement tables
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
    """ICC(2,1): two-way random, single measures, absolute agreement."""
    n = len(x)
    k = 2
    data = np.column_stack([x, y])

    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    ssr = k * np.sum((row_means - grand_mean) ** 2)  # between subjects
    ssc = n * np.sum((col_means - grand_mean) ** 2)  # between raters
    sst = np.sum((data - grand_mean) ** 2)
    sse = sst - ssr - ssc

    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))

    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    if abs(denom) < 1e-15:
        return 1.0
    return (msr - mse) / denom


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r):
        return (np.nan, np.nan)
    if abs(r) >= 1.0:
        return (float(r), float(r))
    r_clip = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r_clip)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1.0 - alpha / 2.0)
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return (float(lo), float(hi))


def bootstrap_ci_paired(
    a: np.ndarray,
    b: np.ndarray,
    stat_fn,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    n = len(a)
    if n < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            v = stat_fn(a[idx], b[idx])
            if np.isfinite(v):
                vals.append(float(v))
        except Exception:
            continue
    if len(vals) < max(100, n_boot // 10):
        return (np.nan, np.nan)
    lo = np.percentile(vals, 100.0 * (alpha / 2.0))
    hi = np.percentile(vals, 100.0 * (1.0 - alpha / 2.0))
    return (float(lo), float(hi))


def compute_stats(
    sub: pd.DataFrame,
    label: str,
    ci_alpha: float,
    ci_boot: int,
    seed_base: int = 42,
) -> list[dict]:
    rows = []
    for i_m, m in enumerate(CAF_MEASURES):
        a = sub[f"{m}_auto"].astype(float).values
        b = sub[f"{m}_manual"].astype(float).values
        n = len(a)
        if n < 2:
            rows.append(
                dict(
                    measure=m,
                    subset=label,
                    n=n,
                    mean_auto=np.nan,
                    mean_manual=np.nan,
                    pearson_r=np.nan,
                    pearson_p=np.nan,
                    pearson_ci_lo=np.nan,
                    pearson_ci_hi=np.nan,
                    spearman_rho=np.nan,
                    spearman_p=np.nan,
                    spearman_ci_lo=np.nan,
                    spearman_ci_hi=np.nan,
                    icc_2_1=np.nan,
                    icc_ci_lo=np.nan,
                    icc_ci_hi=np.nan,
                    bias=np.nan,
                    mae=np.nan,
                    loa_lo=np.nan,
                    loa_hi=np.nan,
                )
            )
            continue

        r_val, r_p = stats.pearsonr(a, b)
        rho_val, rho_p = stats.spearmanr(a, b)
        icc = icc_2_1(a, b)
        r_ci_lo, r_ci_hi = fisher_ci(float(r_val), n, alpha=ci_alpha)
        rho_ci_lo, rho_ci_hi = fisher_ci(float(rho_val), n, alpha=ci_alpha)
        icc_ci_lo, icc_ci_hi = bootstrap_ci_paired(
            a,
            b,
            icc_2_1,
            n_boot=ci_boot,
            alpha=ci_alpha,
            seed=seed_base + (i_m * 17) + (0 if label == "Overall" else (1000 if label == "ST1" else 2000)),
        )

        diff = a - b
        bias = float(np.mean(diff))
        mae = float(np.mean(np.abs(diff)))
        sd_diff = float(np.std(diff, ddof=1))
        loa_lo = bias - 1.96 * sd_diff
        loa_hi = bias + 1.96 * sd_diff

        rows.append(
            dict(
                measure=m,
                subset=label,
                n=n,
                mean_auto=round(float(np.mean(a)), 4),
                mean_manual=round(float(np.mean(b)), 4),
                pearson_r=round(float(r_val), 3),
                pearson_p=float(r_p),
                pearson_ci_lo=round(r_ci_lo, 3) if np.isfinite(r_ci_lo) else np.nan,
                pearson_ci_hi=round(r_ci_hi, 3) if np.isfinite(r_ci_hi) else np.nan,
                spearman_rho=round(float(rho_val), 3),
                spearman_p=float(rho_p),
                spearman_ci_lo=round(rho_ci_lo, 3) if np.isfinite(rho_ci_lo) else np.nan,
                spearman_ci_hi=round(rho_ci_hi, 3) if np.isfinite(rho_ci_hi) else np.nan,
                icc_2_1=round(float(icc), 3),
                icc_ci_lo=round(icc_ci_lo, 3) if np.isfinite(icc_ci_lo) else np.nan,
                icc_ci_hi=round(icc_ci_hi, 3) if np.isfinite(icc_ci_hi) else np.nan,
                bias=round(bias, 4),
                mae=round(mae, 4),
                loa_lo=round(loa_lo, 4),
                loa_hi=round(loa_hi, 4),
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto-csv", type=Path, required=True, help="CAF CSV from ASR clause TextGrids.")
    ap.add_argument("--manual-csv", type=Path, required=True, help="CAF CSV from manual-gold clause TextGrids.")
    ap.add_argument("--out-stats", type=Path, required=True, help="Output correlation/agreement CSV.")
    ap.add_argument("--out-file-level", type=Path, required=True, help="Output merged file-level CSV.")
    ap.add_argument("--ci-alpha", type=float, default=0.05, help="Alpha for confidence intervals.")
    ap.add_argument("--ci-bootstrap", type=int, default=3000, help="Bootstrap draws for ICC CI.")
    args = ap.parse_args()

    auto_df = pd.read_csv(args.auto_csv)
    manual_df = pd.read_csv(args.manual_csv)

    auto_df = auto_df[auto_df["error"].isna()] if "error" in auto_df.columns else auto_df
    manual_df = manual_df[manual_df["error"].isna()] if "error" in manual_df.columns else manual_df

    auto_df["file_id"] = auto_df["file"].apply(file_id_from_name)
    manual_df["file_id"] = manual_df["file"].apply(file_id_from_name)

    merged = pd.merge(auto_df, manual_df, on="file_id", suffixes=("_auto", "_manual"))
    merged["task"] = merged["file_id"].apply(lambda x: "ST1" if "ST1" in x else ("ST2" if "ST2" in x else "OTHER"))

    for m in CAF_MEASURES:
        merged = merged[pd.to_numeric(merged[f"{m}_auto"], errors="coerce").notna()]
        merged = merged[pd.to_numeric(merged[f"{m}_manual"], errors="coerce").notna()]

    rows = []
    rows.extend(compute_stats(merged, "Overall", ci_alpha=args.ci_alpha, ci_boot=args.ci_bootstrap, seed_base=42))
    rows.extend(
        compute_stats(merged[merged["task"] == "ST1"], "ST1", ci_alpha=args.ci_alpha, ci_boot=args.ci_bootstrap, seed_base=42)
    )
    rows.extend(
        compute_stats(merged[merged["task"] == "ST2"], "ST2", ci_alpha=args.ci_alpha, ci_boot=args.ci_bootstrap, seed_base=42)
    )
    res_df = pd.DataFrame(rows)

    args.out_stats.parent.mkdir(parents=True, exist_ok=True)
    args.out_file_level.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(args.out_stats, index=False)

    keep_cols = ["file_id", "task"]
    for m in CAF_MEASURES:
        keep_cols.extend([f"{m}_auto", f"{m}_manual"])
    merged[keep_cols].to_csv(args.out_file_level, index=False)

    print(f"Matched files: {len(merged)}")
    print(f"ST1: {(merged['task'] == 'ST1').sum()}  ST2: {(merged['task'] == 'ST2').sum()}")
    print(f"Saved stats: {args.out_stats}")
    print(f"Saved file-level: {args.out_file_level}")


if __name__ == "__main__":
    main()

