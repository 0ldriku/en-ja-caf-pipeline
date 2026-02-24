# RQ3: Concurrent Validity of Japanese CAF Measures (ASR vs Manual)

## Overview
- **Segmenter**: `ja_clause_segmenter_v16.py` (v16-clean, filler/disfluency-free labels)
- **CAF calculator**: `caf_calculator_ja.py` (mora-based)
- **Samples**: 40 files (20 ST1, 20 ST2) from 20 L2 Japanese speakers
- **Manual gold standard**: gold_v2 (auto-segmented + 9 manual clause boundary fixes after comprehensive review against Vercellotti & Hall 2024 rules)
- **Date**: 2026-02-23

## Overall Results (n=40)

| Measure | Mean (ASR) | Mean (Manual) | Pearson r | Spearman rho | ICC(2,1) | MAE |
|---------|-----------|--------------|-----------|-------------|----------|------|
| AR | 5.122 | 4.845 | .952 | .942 | .877 | 0.296 |
| SR | 2.563 | 2.669 | .992 | .988 | .982 | 0.116 |
| MLR | 7.360 | 7.645 | .991 | .987 | .975 | 0.460 |
| MCPR | 0.078 | 0.077 | .915 | .929 | .915 | 0.009 |
| ECPR | 0.078 | 0.078 | .891 | .886 | .888 | 0.009 |
| PR | 0.156 | 0.155 | .990 | .986 | .984 | 0.007 |
| MCPD | 0.895 | 0.752 | .955 | .895 | .878 | 0.152 |
| ECPD | 1.951 | 1.740 | .944 | .905 | .890 | 0.223 |
| MPD | 1.428 | 1.261 | .953 | .886 | .890 | 0.175 |

## By Task

### ST1 (n=20)

| Measure | Pearson r | Spearman rho | ICC(2,1) | MAE |
|---------|-----------|-------------|----------|------|
| AR | .964 | .965 | .896 | 0.290 |
| SR | .993 | .989 | .986 | 0.101 |
| MLR | .997 | .987 | .988 | 0.348 |
| MCPR | .902 | .879 | .901 | 0.010 |
| ECPR | .850 | .829 | .844 | 0.009 |
| PR | .993 | .988 | .990 | 0.006 |
| MCPD | .960 | .853 | .881 | 0.130 |
| ECPD | .930 | .934 | .868 | 0.244 |
| MPD | .961 | .908 | .900 | 0.163 |

### ST2 (n=20)

| Measure | Pearson r | Spearman rho | ICC(2,1) | MAE |
|---------|-----------|-------------|----------|------|
| AR | .941 | .925 | .861 | 0.303 |
| SR | .993 | .976 | .979 | 0.132 |
| MLR | .986 | .955 | .961 | 0.572 |
| MCPR | .927 | .938 | .930 | 0.009 |
| ECPR | .916 | .908 | .915 | 0.008 |
| PR | .988 | .950 | .980 | 0.009 |
| MCPD | .953 | .916 | .881 | 0.174 |
| ECPD | .953 | .892 | .908 | 0.201 |
| MPD | .945 | .887 | .884 | 0.187 |

## Interpretation

All 9 CAF measures demonstrate strong concurrent validity between ASR-based and manual transcription-based calculations:

- **Fluency (SR, MLR, PR)**: Highest agreement (r = .990-.992, ICC = .975-.984). Speech rate and mean length of run are robust to transcription differences.
- **Complexity (AR, MCPR, ECPR)**: Strong agreement (r = .891-.952, ICC = .877-.915). Clause complexity ratios are well-preserved.
- **Accuracy (MCPD, ECPD, MPD)**: Strong agreement (r = .944-.955, ICC = .878-.890). Slight systematic bias (ASR > Manual by ~0.14-0.21) reflects ASR tendency to over-detect pauses within clauses.

### Gold standard quality
The manual gold standard (gold_v2) was created by:
1. Auto-segmenting 40 manual transcription TextGrids with v16-clean clause segmenter
2. Comprehensive manual review of all 40 files against Vercellotti & Hall (2024) clause coding rules
3. Applying 9 boundary fixes for: auxiliary splits (2), particle splits (2), verb inflection splits (3), compound verb splits (1), te-form splits (1)

### Comparison with previous run (v16-fixed)
Results are very similar to the previous run. The comprehensive gold_v2 review (9 fixes vs original 2) produced minimal changes to correlation statistics, confirming that the auto-segmenter's clause boundaries are highly consistent with linguistic gold standards.

## V16 Segmenter Patches

Following the gold_v2 review, the v16 clause segmenter was patched with two new word-index-level repair methods to auto-fix boundary violations:

1. **`_repair_inflection_splits`**: Fixes verb inflection morphology split across clauses (ました, ません, ましたが)
2. **`_repair_auxiliary_splits`**: Fixes te-form auxiliary constructions split across clauses (てしまう, ていく, てくる, ている, ておく)

### Patch validation
- **5 of 7** automatable gold_v2 issues now auto-fixed (the 2 のか/のかしら particle splits require syntactic context and remain manual-only)
- **2 edge cases** not auto-fixed: CCM35-ST2 (0.86s gap + filler), KKR40-ST2 (alignment-level issue)
- **ASR regression test**: 3 of 40 files changed, all improvements; correlation stats near-identical (max Δr = ±0.002, max Δρ = ±0.005, max ΔICC = ±0.001)
- **複合動詞 (compound verbs)**: V1連用形+V2 patterns (食べ始める, 走り出す, etc.) were reviewed — zero splits found across 2,510 clauses; GiNZA handles these correctly

### Impact on reported results
The patches do not change the RQ3 results reported above, which use gold_v2 manual fixes. The patches ensure future runs of v16 will automatically handle most of these boundary issues without manual intervention.

## VAD Sensitivity Probe (JA, 2026-02-23)

### Why this probe was run
We tested the same VAD-style pause refinement used in EN as a targeted check for MCPD drift (ASR missed fillers/non-lexical speech inside long blank spans).

### What was run
- Input clauses: `ja/rq3_v16_clean_release_20260223/auto_clauses` (40 files)
- Baseline auto CAF: `ja/rq3_v16_clean_release_20260223/auto_caf_results.csv`
- Manual CAF reference: `ja/rq3_v16_clean_release_20260223/manual_caf_results.csv`
- Audio source: `ja/data/dataset_l1_focus20/audio` (`.mp3`)
- VAD CAF script: `ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_vad.py`
- Probe comparison script: `ja/rq3_v16_clean_release_20260223/scripts/run_rq3_vad_probe_ja.py`
- Run log: `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/VAD_PROBE_RUN_LOG.md`

### VAD settings
- `vad_top_db = 30`
- `vad_min_occupancy = 0.20`
- `vad_min_voiced = 0.15 s`
- `vad_merge_gap = 0.10 s`

### Probe outputs
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/auto_caf_results_vad_tuned.csv`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/rq3_vad_probe_summary.csv`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/rq3_vad_probe_full_table.csv`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/rq3_vad_probe_changes_only.csv`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/rq3_vad_probe_changes_only_nonzero.csv`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_vad_probe/rq3_vad_probe_mcpd_file_deltas.csv`

### VAD refinement summary
- Matched files: 40
- `vad_pause_candidates`: 1860
- `vad_pauses_split`: 48
- `vad_speech_islands`: 164

### Main results (Overall, n=40): baseline vs VAD

| Measure | Pearson r (base -> VAD) | ICC(2,1) (base -> VAD) | MAE (base -> VAD) |
|---------|--------------------------|-------------------------|-------------------|
| AR | .952 -> .773 | .877 -> .772 | 0.296 -> 0.424 |
| SR | .992 -> .992 | .982 -> .982 | 0.116 -> 0.116 |
| MLR | .991 -> .991 | .975 -> .975 | 0.460 -> 0.460 |
| MCPR | .915 -> .913 | .915 -> .912 | 0.009 -> 0.010 |
| ECPR | .891 -> .832 | .888 -> .821 | 0.009 -> 0.013 |
| PR | .990 -> .972 | .984 -> .962 | 0.007 -> 0.012 |
| MCPD | .955 -> .952 | .878 -> .892 | 0.152 -> 0.136 |
| ECPD | .944 -> .760 | .890 -> .745 | 0.223 -> 0.351 |
| MPD | .953 -> .826 | .890 -> .818 | 0.175 -> 0.221 |

### MCPD by task: baseline vs VAD

| Subset | Pearson r (base -> VAD) | ICC(2,1) (base -> VAD) | MAE (base -> VAD) |
|--------|--------------------------|-------------------------|-------------------|
| Overall | .955 -> .952 | .878 -> .892 | 0.152 -> 0.136 |
| ST1 | .960 -> .957 | .881 -> .883 | 0.130 -> 0.125 |
| ST2 | .953 -> .950 | .881 -> .901 | 0.174 -> 0.148 |

### File-level MCPD effect
- Improved files: 8
- Worse files: 1
- Unchanged files: 31
- Largest improvement: `KKD19-ST2` (absolute MCPD error `0.319 -> 0.066`)
- Only worsened file: `CCH16-ST2` (absolute MCPD error `0.067 -> 0.093`)

### Interpretation
- VAD refinement helps MCPD specifically (lower MAE, higher ICC), especially in ST2.
- VAD refinement degrades several global duration/rate agreement metrics (AR, ECPD, MPD).
- Likely mechanism in this JA corpus: noisy learner recordings (laugh, clothing/paper handling, background noise) are energy-detected as voiced islands, causing over-splitting/reshaping of pause spans.
- SR and MLR were unchanged in this probe.

### Decision for reporting
- Keep the non-VAD baseline as the main RQ3 result.
- Report VAD as a targeted sensitivity analysis for MCPD pause-drift behavior, not as the default pipeline.

## Unified Gap-Only Candidate Re-run (2026-02-24)

To align EN and JA candidate extraction, we reran JA using the same shared candidate script as EN:

- Shared candidate script: `en/postprocess_vad_filler_classifier_en.py`
- Setting: `--gap-only` (no VAD island split inside gaps)
- Classifier: `shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt`
- Neural training code/reference:
  - `shared/filler_classifier/train_podcastfillers_neural_classifier.py`
  - `shared/filler_classifier/README.md` (training setup + paper reference)

JA gap-only pipeline:
1. Candidate extraction: `en/postprocess_vad_filler_classifier_en.py --gap-only ...`
2. CAF computation: `ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_gap_classifier.py`
3. Correlation probe: `ja/rq3_v16_clean_release_20260223/scripts/run_rq3_vad_classifier_probe_ja.py`

Run log and outputs:
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/RUN_LOG.md`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/auto_caf_gaponly_neural_t050.csv`
- `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/probe/rq3_gaponly_neural_t050_probe_summary.csv`

MCPD result (Overall, baseline -> prior VAD -> gap-only neural):
- Pearson r: `0.955 -> 0.952 -> 0.955`
- ICC(2,1): `0.878 -> 0.892 -> 0.878`
- MAE: `0.152 -> 0.137 -> 0.151`

Interpretation:
- Gap-only neural is essentially baseline-equivalent for JA MCPD correlation and did not beat the prior VAD-tuned JA probe.
- Therefore, the main RQ3 reporting remains the non-VAD baseline tables above, with this gap-only run treated as a reproducibility/sensitivity check.

Fresh rerun from filler detection (2026-02-24):
- A full fresh rerun (candidate extraction -> CAF -> probe) was executed in:
  - `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/`
  - `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/RUN_LOG.md`
  - `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_summary.csv`
- The fresh rerun reproduced the same MCPD summary pattern:
  - Overall: `r 0.955 -> 0.952 -> 0.955`, `ICC 0.878 -> 0.892 -> 0.878`, `MAE 0.152 -> 0.137 -> 0.151`
  - ST1: `r 0.960 -> 0.957 -> 0.959`, `ICC 0.881 -> 0.883 -> 0.877`, `MAE 0.130 -> 0.125 -> 0.130`
  - ST2: `r 0.953 -> 0.950 -> 0.953`, `ICC 0.881 -> 0.901 -> 0.882`, `MAE 0.174 -> 0.148 -> 0.172`
