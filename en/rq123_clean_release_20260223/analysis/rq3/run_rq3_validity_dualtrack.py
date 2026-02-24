#!/usr/bin/env python3
"""
RQ3 dual-track validity summary:
1) Full-quality set (current release baseline): matched files minus quality exclusions
2) Gold-40 raw set (all gold files from selected_files.json)
3) Gold-40 quality-filtered set (gold files minus quality exclusions)

Outputs:
  - rq3_dualtrack_summary.csv
  - rq3_dualtrack_file_membership.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE / "results"
ANNOTATION_DIR = BASE / "annotation"

AUTO_CSV = RESULTS_DIR / "qwen3_filler_mfa_beam100" / "caf_results_beam100.csv"
MANUAL_CSV = RESULTS_DIR / "manual_260212" / "caf_results_manual.csv"
SELECTED_JSON = ANNOTATION_DIR / "selected_files.json"

CAF_MEASURES = ["AR", "SR", "MLR", "MCPR", "ECPR", "PR", "MCPD", "ECPD", "MPD"]

# Same exclusion list used in the release RQ3 script/report
EXCLUDED_FILES = {
    "ALL_142_M_RUS_ENG_ST1",
    "ALL_141_M_HEB_ENG_ST1",
    "ALL_139_M_PBR_ENG_ST2",
    "ALL_140_M_RUS_ENG_ST1",
    "ALL_144_M_RUS_ENG_ST1",
    "ALL_139_M_PBR_ENG_ST1",
    "ALL_143_M_FAR_ENG_ST2",
    "ALL_140_M_RUS_ENG_ST2",
    "ALL_145_F_VIE_ENG_ST1",
    "ALL_143_M_FAR_ENG_ST1",
    "ALL_142_M_RUS_ENG_ST2",
    "ALL_093_M_TUR_ENG_ST2",
    "ALL_086_F_CCT_ENG_ST2",
    "ALL_141_M_HEB_ENG_ST2",
    "ALL_110_M_KOR_ENG_ST2",
    "ALL_021_M_CMN_ENG_ST2",
}


def icc_2_1(x: np.ndarray, y: np.ndarray) -> float:
    """
    ICC(2,1): two-way random, single measures, absolute agreement.
    """
    n = len(x)
    k = 2
    data = np.column_stack([x, y])
    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    ssr = k * np.sum((row_means - grand_mean) ** 2)
    ssc = n * np.sum((col_means - grand_mean) ** 2)
    sst = np.sum((data - grand_mean) ** 2)
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
    return out


def compute_stats(df: pd.DataFrame, track: str, subset: str) -> list[dict]:
    rows = []
    for m in CAF_MEASURES:
        a = df[f"{m}_auto"].to_numpy(dtype=float)
        b = df[f"{m}_manual"].to_numpy(dtype=float)
        r_val, r_p = stats.pearsonr(a, b)
        rho_val, rho_p = stats.spearmanr(a, b)
        diff = a - b
        rows.append(
            {
                "track": track,
                "subset": subset,
                "n": len(df),
                "measure": m,
                "mean_auto": round(float(np.mean(a)), 4),
                "mean_manual": round(float(np.mean(b)), 4),
                "pearson_r": round(float(r_val), 3),
                "pearson_p": float(r_p),
                "spearman_rho": round(float(rho_val), 3),
                "spearman_p": float(rho_p),
                "icc_2_1": round(float(icc_2_1(a, b)), 3),
                "bias": round(float(np.mean(diff)), 4),
                "mae": round(float(np.mean(np.abs(diff))), 4),
            }
        )
    return rows


def main() -> None:
    auto_df = with_file_id(pd.read_csv(AUTO_CSV))
    manual_df = with_file_id(pd.read_csv(MANUAL_CSV))
    selected = json.loads(SELECTED_JSON.read_text(encoding="utf-8"))["all_selected"]
    selected_set = set(selected)

    merged = pd.merge(auto_df, manual_df, on="file_id", suffixes=("_auto", "_manual"))
    merged["task"] = np.where(merged["file_id"].str.contains("ST1"), "ST1", "ST2")

    # Ensure all CAF measure cells are numeric
    for m in CAF_MEASURES:
        merged = merged[
            pd.to_numeric(merged[f"{m}_auto"], errors="coerce").notna()
            & pd.to_numeric(merged[f"{m}_manual"], errors="coerce").notna()
        ]

    full_quality = merged[~merged["file_id"].isin(EXCLUDED_FILES)].copy()
    gold40_raw = merged[merged["file_id"].isin(selected_set)].copy()
    gold40_quality = gold40_raw[~gold40_raw["file_id"].isin(EXCLUDED_FILES)].copy()

    tracks = [
        ("full_quality_174", full_quality),
        ("gold40_raw", gold40_raw),
        ("gold40_quality_39", gold40_quality),
    ]

    all_rows: list[dict] = []
    for track_name, df_track in tracks:
        all_rows.extend(compute_stats(df_track, track_name, "Overall"))
        all_rows.extend(compute_stats(df_track[df_track["task"] == "ST1"], track_name, "ST1"))
        all_rows.extend(compute_stats(df_track[df_track["task"] == "ST2"], track_name, "ST2"))

    out_dir = Path(__file__).resolve().parent
    summary_out = out_dir / "rq3_dualtrack_summary.csv"
    pd.DataFrame(all_rows).to_csv(summary_out, index=False)

    membership = merged[["file_id", "task"]].copy()
    membership["in_full_quality_174"] = membership["file_id"].isin(full_quality["file_id"])
    membership["in_gold40_raw"] = membership["file_id"].isin(gold40_raw["file_id"])
    membership["in_gold40_quality_39"] = membership["file_id"].isin(gold40_quality["file_id"])
    membership["excluded_by_quality_filter"] = membership["file_id"].isin(EXCLUDED_FILES)
    membership_out = out_dir / "rq3_dualtrack_file_membership.csv"
    membership.to_csv(membership_out, index=False)

    print("=" * 80)
    print("RQ3 DUAL-TRACK SUMMARY")
    print("=" * 80)
    print(f"matched files            : {len(merged)}")
    print(f"full quality             : {len(full_quality)}")
    print(f"gold40 raw               : {len(gold40_raw)}")
    print(f"gold40 quality-filtered  : {len(gold40_quality)}")
    print(f"gold files in exclusions : {sum(fid in EXCLUDED_FILES for fid in selected)}")
    print("saved:")
    print(f"  - {summary_out}")
    print(f"  - {membership_out}")


if __name__ == "__main__":
    main()
