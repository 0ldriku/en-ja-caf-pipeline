#!/usr/bin/env python3
"""
RQ3: Concurrent Validity of CAF Measures — Auto vs Manual Pipeline

Compare 9 CAF measures computed from:
  Auto:   qwen3_filler_mfa_beam100 (Qwen3 ASR → MFA beam=100 → spaCy segmenter v3)
  Manual: manual_260212 (human transcripts → same spaCy segmenter v3)

Both pipelines use the same clause segmentation script (textgrid_caf_segmenter_v3.py)
and CAF calculator (caf_calculator.py). The only difference is the input: ASR transcript
+ forced alignment vs human transcript + manual TextGrid timing.

Statistics per CAF measure:
  - Pearson r (linear correlation)
  - Spearman ρ (rank correlation)
  - ICC(2,1) (two-way random, single measures, absolute agreement)
  - Mean bias (auto − manual)
  - Mean absolute error (MAE)
  - 95% limits of agreement (Bland-Altman: bias ± 1.96·SD)

Run with:  ../.venv/bin/python rq3/run_rq3_validity.py   (from en/analysis/)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent          # en/
RESULTS_DIR = BASE / 'results'
AUTO_CSV = RESULTS_DIR / 'qwen3_filler_mfa_beam100' / 'caf_results_beam100.csv'
MANUAL_CSV = RESULTS_DIR / 'manual_260212' / 'caf_results_manual.csv'

# ── Exclusions (from RUN_LOG exclusion analysis) ─────────────────────────────
# Category 1: Preamble mismatch — manual starts much later than ASR (11 files)
# Category 2: Incomplete manual transcription — mid-file gaps (5 files)
# Overlap: ALL_141_M_HEB_ENG_ST2 in both → 15 unique files excluded
EXCLUDED_FILES = {
    # Preamble mismatch (11)
    'ALL_142_M_RUS_ENG_ST1',
    'ALL_141_M_HEB_ENG_ST1',
    'ALL_139_M_PBR_ENG_ST2',
    'ALL_140_M_RUS_ENG_ST1',
    'ALL_144_M_RUS_ENG_ST1',
    'ALL_139_M_PBR_ENG_ST1',
    'ALL_143_M_FAR_ENG_ST2',
    'ALL_140_M_RUS_ENG_ST2',
    'ALL_145_F_VIE_ENG_ST1',
    'ALL_143_M_FAR_ENG_ST1',
    'ALL_142_M_RUS_ENG_ST2',
    # Incomplete manual (5, one overlaps with preamble)
    'ALL_093_M_TUR_ENG_ST2',
    'ALL_086_F_CCT_ENG_ST2',
    'ALL_141_M_HEB_ENG_ST2',
    'ALL_110_M_KOR_ENG_ST2',
    'ALL_021_M_CMN_ENG_ST2',
}

# ── CAF measures ─────────────────────────────────────────────────────────────
CAF_MEASURES = ['AR', 'SR', 'MLR', 'MCPR', 'ECPR', 'PR', 'MCPD', 'ECPD', 'MPD']

CAF_LABELS = {
    'AR':   'Articulation rate (syl/phon-time)',
    'SR':   'Speech rate (syl/total-time)',
    'MLR':  'Mean length of run (syl)',
    'MCPR': 'Mid-clause pause ratio',
    'ECPR': 'End-clause pause ratio',
    'PR':   'Pause ratio',
    'MCPD': 'Mid-clause pause duration (s)',
    'ECPD': 'End-clause pause duration (s)',
    'MPD':  'Mean pause duration (s)',
}


# ═════════════════════════════════════════════════════════════════════════════
#  ICC(2,1) — two-way random, single measures, absolute agreement
# ═════════════════════════════════════════════════════════════════════════════

def icc_2_1(x, y):
    """
    Compute ICC(2,1): two-way random-effects model, single measures,
    absolute agreement definition.

    x, y: paired arrays from two raters (auto, manual).

    Formula (Shrout & Fleiss, 1979; McGraw & Wong, 1996):
      ICC(2,1) = (MSR - MSE) / (MSR + (k-1)*MSE + k/n * (MSC - MSE))
    where k = 2 raters, n = number of subjects.
    """
    n = len(x)
    k = 2
    data = np.column_stack([x, y])  # n × k

    # Grand mean
    grand_mean = data.mean()
    # Row means and column means
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    # Sum of squares
    SSR = k * np.sum((row_means - grand_mean) ** 2)        # between subjects
    SSC = n * np.sum((col_means - grand_mean) ** 2)        # between raters
    SST = np.sum((data - grand_mean) ** 2)                 # total
    SSE = SST - SSR - SSC                                  # residual (error)

    # Mean squares
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))

    # ICC(2,1)
    denom = MSR + (k - 1) * MSE + (k / n) * (MSC - MSE)
    if abs(denom) < 1e-15:
        return 1.0
    return (MSR - MSE) / denom


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def file_id_from_name(fname):
    """Extract file ID from filename like ALL_001_F_GER_ENG_ST1.TextGrid"""
    return re.sub(r'\.TextGrid$', '', fname, flags=re.IGNORECASE)


def main():
    # ── Load data ────────────────────────────────────────────────────────
    auto_df = pd.read_csv(AUTO_CSV)
    manual_df = pd.read_csv(MANUAL_CSV)

    auto_df['file_id'] = auto_df['file'].apply(file_id_from_name)
    manual_df['file_id'] = manual_df['file'].apply(file_id_from_name)

    # Merge on file_id
    merged = pd.merge(auto_df, manual_df, on='file_id', suffixes=('_auto', '_manual'))

    # Add task column
    merged['task'] = merged['file_id'].apply(lambda x: 'ST1' if 'ST1' in x else 'ST2')

    # Filter out files with errors
    for m in CAF_MEASURES:
        for sfx in ['_auto', '_manual']:
            col = m + sfx
            if col in merged.columns:
                merged = merged[pd.to_numeric(merged[col], errors='coerce').notna()]

    n_before = len(merged)

    # Apply exclusions (preamble mismatch + incomplete manual)
    excluded_mask = merged['file_id'].isin(EXCLUDED_FILES)
    n_excluded = excluded_mask.sum()
    excluded_ids = sorted(merged.loc[excluded_mask, 'file_id'].tolist())
    merged = merged[~excluded_mask].reset_index(drop=True)

    n_total = len(merged)
    n_st1 = (merged['task'] == 'ST1').sum()
    n_st2 = (merged['task'] == 'ST2').sum()

    print('=' * 80)
    print('RQ3: CONCURRENT VALIDITY OF CAF MEASURES')
    print('    Auto (Qwen3 ASR + MFA beam=100) vs Manual (human transcripts)')
    print('=' * 80)
    print(f'\nAuto CSV  : {AUTO_CSV}')
    print(f'Manual CSV: {MANUAL_CSV}')
    print(f'Files matched: {n_before}')
    print(f'Excluded: {n_excluded}  (preamble mismatch: 11, incomplete manual: 5)')
    for eid in excluded_ids:
        print(f'  - {eid}')
    print(f'Analysed: {n_total}  (ST1: {n_st1}, ST2: {n_st2})')

    # ── Per-measure statistics ───────────────────────────────────────────
    def compute_stats(sub, label):
        rows = []
        for m in CAF_MEASURES:
            a = sub[m + '_auto'].values.astype(float)
            b = sub[m + '_manual'].values.astype(float)
            n = len(a)

            # Correlations
            r_val, r_p = stats.pearsonr(a, b)
            rho_val, rho_p = stats.spearmanr(a, b)

            # ICC
            icc = icc_2_1(a, b)

            # Agreement
            diff = a - b
            bias = np.mean(diff)
            mae = np.mean(np.abs(diff))
            sd_diff = np.std(diff, ddof=1)
            loa_lo = bias - 1.96 * sd_diff
            loa_hi = bias + 1.96 * sd_diff

            # Means
            mean_auto = np.mean(a)
            mean_manual = np.mean(b)

            rows.append(dict(
                measure=m,
                subset=label,
                n=n,
                mean_auto=round(mean_auto, 4),
                mean_manual=round(mean_manual, 4),
                pearson_r=round(r_val, 3),
                pearson_p=r_p,
                spearman_rho=round(rho_val, 3),
                spearman_p=rho_p,
                icc_2_1=round(icc, 3),
                bias=round(bias, 4),
                mae=round(mae, 4),
                loa_lo=round(loa_lo, 4),
                loa_hi=round(loa_hi, 4),
            ))
        return rows

    all_rows = []
    all_rows.extend(compute_stats(merged, 'Overall'))
    all_rows.extend(compute_stats(merged[merged['task'] == 'ST1'], 'ST1'))
    all_rows.extend(compute_stats(merged[merged['task'] == 'ST2'], 'ST2'))

    results_df = pd.DataFrame(all_rows)

    # ── Print summary ────────────────────────────────────────────────────
    for subset in ['Overall', 'ST1', 'ST2']:
        sub = results_df[results_df['subset'] == subset]
        print(f'\n── {subset} (n={sub.iloc[0]["n"]}) ──')
        print(f'{"Measure":<8} {"Mean(A)":>9} {"Mean(M)":>9} {"Pearson r":>10} '
              f'{"Spearman":>10} {"ICC(2,1)":>10} {"Bias":>9} {"MAE":>9}')
        print('-' * 85)
        for _, row in sub.iterrows():
            sig_r = '***' if row['pearson_p'] < .001 else ('**' if row['pearson_p'] < .01 else ('*' if row['pearson_p'] < .05 else ''))
            sig_rho = '***' if row['spearman_p'] < .001 else ('**' if row['spearman_p'] < .01 else ('*' if row['spearman_p'] < .05 else ''))
            print(f'{row["measure"]:<8} {row["mean_auto"]:>9.4f} {row["mean_manual"]:>9.4f} '
                  f'{row["pearson_r"]:>7.3f}{sig_r:<3} '
                  f'{row["spearman_rho"]:>7.3f}{sig_rho:<3} '
                  f'{row["icc_2_1"]:>10.3f} {row["bias"]:>9.4f} {row["mae"]:>9.4f}')

    # ── Interpretation guidelines ────────────────────────────────────────
    print('\n── Significance: * p<.05, ** p<.01, *** p<.001')
    print('── ICC interpretation (Koo & Li, 2016): <.50 poor, .50-.75 moderate, .75-.90 good, >.90 excellent')

    # ── Save CSV ─────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / 'rq3_concurrent_validity.csv'
    results_df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')

    # ── Per-file merged data (for potential plotting) ────────────────────
    file_out = Path(__file__).parent / 'rq3_file_level.csv'
    cols_keep = ['file_id', 'task']
    for m in CAF_MEASURES:
        cols_keep.extend([m + '_auto', m + '_manual'])
    merged[cols_keep].to_csv(file_out, index=False)
    print(f'Saved: {file_out}')

    # ── Summary ──────────────────────────────────────────────────────────
    ov = results_df[results_df['subset'] == 'Overall']
    min_r = ov['pearson_r'].min()
    max_r = ov['pearson_r'].max()
    min_icc = ov['icc_2_1'].min()
    max_icc = ov['icc_2_1'].max()
    print('\n' + '=' * 80)
    print(f'SUMMARY — RQ3 concurrent validity ({n_total} files):')
    print(f'  Pearson r range: {min_r:.3f} – {max_r:.3f}')
    print(f'  ICC(2,1) range:  {min_icc:.3f} – {max_icc:.3f}')
    all_sig = all(row['pearson_p'] < .001 for _, row in ov.iterrows())
    print(f'  All Pearson correlations p < .001: {all_sig}')
    print('=' * 80)


if __name__ == '__main__':
    main()
