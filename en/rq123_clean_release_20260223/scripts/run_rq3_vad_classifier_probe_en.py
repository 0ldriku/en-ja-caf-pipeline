#!/usr/bin/env python3
"""
Compare three Gold-40 tracks against manual CAF:
  1) Baseline auto CAF
  2) Prior VAD-refined auto CAF (audio VAD)
  3) New classifier-guided pause refinement (VAD+classifier postprocess)

Outputs:
  - rq3_vad_classifier_probe_summary.csv
  - rq3_vad_classifier_probe_mcpd_file_deltas.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
ANNOT = BASE / "annotation"

BASELINE_AUTO = RESULTS / "qwen3_filler_mfa_beam100" / "caf_results_beam100.csv"
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


def build_track(track_name: str, auto_df: pd.DataFrame, manual_gold: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    merged = pd.merge(auto_df, manual_gold, on="file_id", suffixes=("_auto", "_manual"))
    rows: list[dict] = []
    rows.extend(compute_rows(merged, track_name, "Overall"))
    rows.extend(compute_rows(merged[merged["task_auto"] == "ST1"], track_name, "ST1"))
    rows.extend(compute_rows(merged[merged["task_auto"] == "ST2"], track_name, "ST2"))
    return merged, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto-vad-csv", required=True, help="Previous VAD-refined auto CAF CSV")
    ap.add_argument("--auto-clf-csv", required=True, help="New classifier-guided auto CAF CSV")
    ap.add_argument("--manual-csv", required=True, help="Manual CAF reference CSV")
    ap.add_argument("--exclude", nargs="*", default=[], help="File IDs to exclude from correlation")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    gold40 = set(selected["all_selected"]) - set(args.exclude)

    auto_base = with_file_id(pd.read_csv(BASELINE_AUTO))
    auto_vad = with_file_id(pd.read_csv(args.auto_vad_csv))
    auto_clf = with_file_id(pd.read_csv(args.auto_clf_csv))
    manual = with_file_id(pd.read_csv(args.manual_csv))

    manual_gold = manual[manual["file_id"].isin(gold40)].copy()
    base_gold = auto_base[auto_base["file_id"].isin(gold40)].copy()
    vad_gold = auto_vad[auto_vad["file_id"].isin(gold40)].copy()
    clf_gold = auto_clf[auto_clf["file_id"].isin(gold40)].copy()

    merged_base, rows_base = build_track("baseline_gold40", base_gold, manual_gold)
    merged_vad, rows_vad = build_track("vad_gold40", vad_gold, manual_gold)
    merged_clf, rows_clf = build_track("vad_classifier_gold40", clf_gold, manual_gold)

    summary = pd.DataFrame(rows_base + rows_vad + rows_clf)
    summary_out = out_dir / "rq3_vad_classifier_probe_summary.csv"
    summary.to_csv(summary_out, index=False)

    mcpd = pd.merge(
        merged_base[["file_id", "task_auto", "MCPD_auto", "MCPD_manual"]].rename(columns={"task_auto": "task"}),
        merged_vad[["file_id", "MCPD_auto"]].rename(columns={"MCPD_auto": "MCPD_auto_vad"}),
        on="file_id",
        how="inner",
    )
    mcpd = pd.merge(
        mcpd,
        merged_clf[["file_id", "MCPD_auto"]].rename(columns={"MCPD_auto": "MCPD_auto_clf"}),
        on="file_id",
        how="inner",
    )
    mcpd = mcpd.rename(columns={"MCPD_auto": "MCPD_auto_baseline"})
    mcpd["abs_err_baseline"] = (mcpd["MCPD_auto_baseline"] - mcpd["MCPD_manual"]).abs()
    mcpd["abs_err_vad"] = (mcpd["MCPD_auto_vad"] - mcpd["MCPD_manual"]).abs()
    mcpd["abs_err_clf"] = (mcpd["MCPD_auto_clf"] - mcpd["MCPD_manual"]).abs()
    mcpd["delta_vad_minus_base"] = mcpd["abs_err_vad"] - mcpd["abs_err_baseline"]
    mcpd["delta_clf_minus_base"] = mcpd["abs_err_clf"] - mcpd["abs_err_baseline"]
    mcpd["delta_clf_minus_vad"] = mcpd["abs_err_clf"] - mcpd["abs_err_vad"]
    mcpd_out = out_dir / "rq3_vad_classifier_probe_mcpd_file_deltas.csv"
    mcpd.to_csv(mcpd_out, index=False)

    print("=" * 90)
    print("RQ3 3-WAY PROBE (Gold-40): baseline vs prior VAD vs VAD+classifier")
    print("=" * 90)
    for subset in ["Overall", "ST1", "ST2"]:
        b = summary[(summary["track"] == "baseline_gold40") & (summary["subset"] == subset) & (summary["measure"] == "MCPD")].iloc[0]
        v = summary[(summary["track"] == "vad_gold40") & (summary["subset"] == subset) & (summary["measure"] == "MCPD")].iloc[0]
        c = summary[(summary["track"] == "vad_classifier_gold40") & (summary["subset"] == subset) & (summary["measure"] == "MCPD")].iloc[0]
        print(
            f"{subset}: MCPD r {b['pearson_r']:.3f}->{v['pearson_r']:.3f}->{c['pearson_r']:.3f}, "
            f"ICC {b['icc_2_1']:.3f}->{v['icc_2_1']:.3f}->{c['icc_2_1']:.3f}, "
            f"MAE {b['mae']:.3f}->{v['mae']:.3f}->{c['mae']:.3f}"
        )
    print(f"Saved: {summary_out}")
    print(f"Saved: {mcpd_out}")


if __name__ == "__main__":
    main()
