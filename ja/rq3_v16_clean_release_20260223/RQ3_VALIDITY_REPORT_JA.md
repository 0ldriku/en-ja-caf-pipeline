# RQ3: Concurrent Validity of Japanese CAF Measures (ASR vs Manual)

## Overview
- Segmenter: `ja_clause_segmenter_v16.py` (v16-clean)
- Filler candidate step: `en/postprocess_vad_filler_classifier_en.py --gap-only --threshold 0.50`
- CAF calculator: `caf_calculator_ja_gap_classifier.py` (mora-based)
- Samples: 40 files (`ST1=20`, `ST2=20`)
- Manual gold standard: `manual_clauses_gold_v2` (final gold)
- Final run date: 2026-02-24 (fresh run)

## Final RQ3 Pipeline (fresh run)

| Step | Script | Output |
|:--|:--|:--|
| 0a. ASR + alignment | `scripts/asr/asr_qwen3_mfa_ja_v2_spanfix_b.py` | word-level ASR TextGrids + JSON |
| 0b. Manual-span blanking (JA dataset-specific) | `scripts/asr/make_beginning_removed_by_manual.py` | ASR labels outside manual span blanked |
| 1. Clause segmentation | `ja_clause_segmenter_v16.py` | clause TextGrids |
| 2. Gap-only filler scoring | `en/postprocess_vad_filler_classifier_en.py --gap-only --threshold 0.50` | candidate CSVs |
| 3. CAF computation | `caf_calculator_ja_gap_classifier.py` | `auto_caf_gaponly_neural_t050.csv` |
| 4. Correlation analysis | `run_rq3_vad_classifier_probe_ja.py` | exported as `analysis_final_taskwise_correlations_20260224/ja/ja_correlation_final_vad_classifier_all_subsets.csv` |

- Track reported below: `vad_classifier`

## Overall Results (N=40)

| Measure | Mean (ASR) | Mean (Manual) | Pearson *r* | Spearman *rho* | ICC(2,1) | MAE |
|:--|--:|--:|--:|--:|--:|--:|
| AR | 5.052 | 4.845 | .947 | .935 | .903 | 0.242 |
| SR | 2.563 | 2.669 | .992 | .988 | .982 | 0.116 |
| MLR | 7.360 | 7.645 | .991 | .987 | .975 | 0.460 |
| MCPR | 0.076 | 0.077 | .912 | .933 | .913 | 0.009 |
| ECPR | 0.077 | 0.078 | .903 | .883 | .898 | 0.008 |
| PR | 0.152 | 0.155 | .984 | .978 | .978 | 0.009 |
| MCPD | 0.893 | 0.752 | .955 | .902 | .878 | 0.151 |
| ECPD | 1.961 | 1.740 | .942 | .902 | .884 | 0.231 |
| MPD | 1.437 | 1.261 | .948 | .876 | .880 | 0.184 |

## By Task

### ST1 (N=20)

| Measure | Pearson *r* | Spearman *rho* | ICC(2,1) | MAE |
|:--|--:|--:|--:|--:|
| AR | .965 | .965 | .922 | 0.227 |
| SR | .993 | .989 | .986 | 0.101 |
| MLR | .997 | .987 | .988 | 0.348 |
| MCPR | .895 | .884 | .898 | 0.009 |
| ECPR | .872 | .838 | .861 | 0.009 |
| PR | .987 | .986 | .982 | 0.007 |
| MCPD | .959 | .863 | .877 | 0.130 |
| ECPD | .932 | .932 | .865 | 0.250 |
| MPD | .957 | .913 | .889 | 0.174 |

### ST2 (N=20)

| Measure | Pearson *r* | Spearman *rho* | ICC(2,1) | MAE |
|:--|--:|--:|--:|--:|
| AR | .929 | .911 | .886 | 0.257 |
| SR | .993 | .976 | .979 | 0.132 |
| MLR | .986 | .955 | .961 | 0.572 |
| MCPR | .925 | .925 | .928 | 0.009 |
| ECPR | .921 | .892 | .921 | 0.008 |
| PR | .981 | .927 | .977 | 0.010 |
| MCPD | .953 | .920 | .882 | 0.173 |
| ECPD | .947 | .880 | .896 | 0.213 |
| MPD | .940 | .872 | .875 | 0.195 |

## Interpretation

- All 9 CAF measures show strong concurrent validity on the final fresh run.
- Fluency metrics (SR, MLR, PR) remain strongest.
- Pause-duration metrics (MCPD, ECPD, MPD) are also strong, with expected ASR-over-manual bias in duration means.

## Reproducibility (final run artifacts)

- Run folder: `analysis/rq3_gaponly_neural_t050_freshrun_20260224/`
- Run log: `analysis/rq3_gaponly_neural_t050_freshrun_20260224/RUN_LOG.md`
- Summary CSV (source): `analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_summary.csv`
- Auto CAF CSV: `analysis/rq3_gaponly_neural_t050_freshrun_20260224/auto_caf_gaponly_neural_t050.csv`
- Final taskwise export (Overall/ST1/ST2): `analysis_final_taskwise_correlations_20260224/ja/ja_correlation_final_vad_classifier_all_subsets.csv`
- Per-subset exports: `analysis_final_taskwise_correlations_20260224/ja/ja_correlation_final_vad_classifier_overall.csv`, `analysis_final_taskwise_correlations_20260224/ja/ja_correlation_final_vad_classifier_st1.csv`, `analysis_final_taskwise_correlations_20260224/ja/ja_correlation_final_vad_classifier_st2.csv`
