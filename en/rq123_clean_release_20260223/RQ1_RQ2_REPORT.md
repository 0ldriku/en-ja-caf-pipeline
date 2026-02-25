# RQ1–RQ3 Analysis Report — English CAF Pipeline V2

**Date:** 2025-02-14  
**Pipeline:** `qwen3_filler_mfa_beam100` (Qwen3 ASR → filler-augmented MFA beam=100 → spaCy clause segmenter v3)  
**Gold standard:** 10 adjudicated blind files + 30 LLM production files (expert-reviewed)  
**Alignment method:** Edit-distance (Levenshtein), following Matsuura et al. (2025) / NIST SCTK  

---

## 1. Method

### 1.1 Gold standard construction

Clause boundary annotations were created using the LLM-assisted pipeline adapted from Morin & Marttinen Larsson (2025):

1. Two trained coders independently annotated 10 blind files (boundary-only task).
2. Disagreements were adjudicated to produce a gold reference for the 10 files.
3. An LLM (Claude) was trained on the adjudicated data and evaluated on 5 locked test files (micro F1 = .929, κ = .914).
4. The LLM annotated 30 production files; these outputs were then manually reviewed by the researcher and accepted as gold.

Total gold files: **40** (20 ST1, 20 ST2).

### 1.2 Evaluation unit (RQ1)

Following Matsuura et al. (2025), clause boundary agreement was evaluated as a **per-word binary classification task**: for each inter-word position in the aligned sequence, a boundary label (1 = clause boundary, 0 = no boundary) was assigned from both gold and auto segmentations.

- **Gold labels** were derived directly from the gold clause segments on the canonical (human-transcribed) word sequence.
- **Auto labels** were derived from the spaCy clause segmentation on the ASR (Qwen3) word sequence.
- The two word sequences were aligned via **minimum edit-distance alignment** (Levenshtein), replicating the alignment logic of NIST's SCTK tool (Chen & Yoon, 2012). At each alignment position:
  - **Correct (C):** same word → compare labels directly.
  - **Substitution (S):** different word, same aligned position → compare labels directly.
  - **Deletion (D):** canonical word absent from ASR → auto label set to 0 (pipeline penalized).
  - **Insertion (I):** ASR-only word → gold label set to 0.

### 1.3 Evaluation unit (RQ2)

For each silent pause ≥ 250 ms in the auto TextGrid words tier, two MCP/ECP classifications were obtained:

- **Auto label:** pause classified using auto clause intervals from the `clauses` tier.
- **Gold label:** pause classified using gold clause boundaries mapped to auto word timing via edit-distance alignment.

A pause was classified as ECP if its onset fell within 150 ms of any clause offset; otherwise MCP.

### 1.4 Metrics

- **Cohen's κ** (Landis & Koch, 1977): < .20 slight, .21–.40 fair, .41–.60 moderate, .61–.80 substantial, > .80 almost perfect.
- **Precision, Recall, F1** (boundary = positive class for RQ1; per-class for RQ2).
- Both **micro** (pooled across files) and **macro** (mean of per-file scores) reported.

---

## 2. Results

### 2.1 RQ1 — Clause boundary agreement

**To what extent do automatically generated clause boundaries agree with human-validated clause boundaries?**

| Metric | Overall | ST1 | ST2 |
|:--|--:|--:|--:|
| Gold boundaries (N) | 1,131 | 496 | 635 |
| Files | 40 | 20 | 20 |
| Precision | .848 | .882 | .822 |
| Recall | .842 | .857 | .830 |
| F1 (micro) | **.845** | **.869** | **.826** |
| κ (micro) | **.816** | **.845** | **.795** |
| F1 (macro) | .846 | — | — |
| κ (macro) | .819 | — | — |

**Alignment diagnostics:** Mean WER between canonical transcript and ASR clause-text = .121 (12.1%).

**Interpretation:** Overall κ = .816 indicates **almost perfect agreement** (Landis & Koch, 1977). ST1 (picture narrative) yields slightly higher agreement than ST2 (argumentative), consistent with lower ASR error rates on structured narratives. Precision and recall are balanced, indicating no systematic over- or under-segmentation.

#### Per-file distribution

| Statistic | F1 | κ |
|:--|--:|--:|
| Mean | .846 | .819 |
| SD | .093 | .102 |
| Min | .618 | .567 |
| Max | 1.000 | 1.000 |
| Median | .859 | .822 |

Files with lowest agreement (κ < .65): ALL_102_M_SHS_ENG_ST2 (κ = .578), ALL_013_M_JPN_ENG_ST2 (κ = .567). Both have high WER (> .19), suggesting ASR errors propagate to clause boundary detection.

### 2.2 RQ2 — Pause location agreement

**To what extent do automatically classified pause locations (MCP vs ECP) agree with gold-standard classifications?**

| Metric | Overall | ST1 | ST2 |
|:--|--:|--:|--:|
| Pauses (N) | 1,902 | 822 | 1,080 |
| Files | 40 | 20 | 20 |
| κ | **.840** | **.873** | **.815** |
| Accuracy | .921 | .937 | .909 |
| MCP Precision | .939 | .943 | .936 |
| MCP Recall | .919 | .938 | .906 |
| MCP F1 | **.929** | **.941** | **.920** |
| ECP Precision | .900 | .930 | .876 |
| ECP Recall | .924 | .935 | .914 |
| ECP F1 | **.912** | **.932** | **.894** |

**Interpretation:** Overall κ = .840 indicates **almost perfect agreement**. MCP classification (F1 = .929) slightly outperforms ECP (F1 = .912), as expected: mid-clause pauses are more clearly positioned within clause boundaries. ST1 again slightly outperforms ST2.

#### Per-file distribution

| Statistic | Accuracy |
|:--|--:|
| Mean | .921 |
| SD | .052 |
| Min | .789 |
| Max | 1.000 |
| Median | .927 |

Six files achieved perfect accuracy (1.000). The lowest-accuracy file (ALL_102_M_SHS_ENG_ST2, .789) also had low RQ1 agreement, confirming that clause boundary errors propagate to pause classification.

---

### 2.3 RQ3 - Concurrent validity of CAF measures (Final fresh run, quality-filtered Gold-39)

**To what extent do CAF measures from the automatic pipeline agree with those from the manual pipeline?**

#### Final RQ3 pipeline (fresh run)

| Step | Script | Output |
|:--|:--|:--|
| 0. ASR + alignment | `asr_qwen3_mfa_en.py` | word-level ASR TextGrids |
| 1. Clause segmentation | `scripts/textgrid_caf_segmenter_v3.py` | clause TextGrids |
| 2. Gap-only filler scoring | `en/postprocess_vad_filler_classifier_en.py --gap-only --threshold 0.50` | candidate CSVs + updated review tiers |
| 3. CAF computation | `scripts/caf_calculator_vad_classifier.py` | `auto_caf_gold40_gaponly_neural_t050.csv` |
| 4. Correlation analysis | `scripts/run_rq3_vad_classifier_probe_en.py` | exported as `analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_all_subsets.csv` |

- Cohort: **39 quality-filtered gold files** (`ST1=19`, `ST2=20`); excluded `ALL_139_M_PBR_ENG_ST1` (manual preamble mismatch)
- Track reported below: `vad_classifier_gold40` with quality exclusion (`N=39`).

#### Overall results (N=39)

| Measure | Mean (Auto) | Mean (Manual) | Pearson *r* | Spearman *rho* | ICC(2,1) | MAE |
|:--|--:|--:|--:|--:|--:|--:|
| AR | 3.708 | 3.691 | .956 | .911 | .953 | 0.157 |
| SR | 2.313 | 2.362 | .988 | .978 | .984 | 0.082 |
| MLR | 5.670 | 5.952 | .971 | .968 | .965 | 0.454 |
| MCPR | 0.108 | 0.103 | .966 | .981 | .962 | 0.011 |
| ECPR | 0.092 | 0.091 | .864 | .889 | .866 | 0.008 |
| PR | 0.200 | 0.194 | .961 | .965 | .958 | 0.014 |
| MCPD | 0.771 | 0.725 | .821 | .808 | .799 | 0.090 |
| ECPD | 1.090 | 1.048 | .938 | .912 | .935 | 0.115 |
| MPD | 0.919 | 0.887 | .957 | .908 | .944 | 0.074 |

#### Task summary

- **ST1 (N=19)**: weakest measure remains MCPD (`r=.713`, `ICC=.669`, `MAE=0.097`).
- **ST2 (N=20)**: weakest measure is ECPR (`r=.935`, `ICC=.935`, `MAE=0.006`); MCPD remains strong (`r=.910`, `ICC=.886`).

#### ST1 detailed correlations (N=19)

| Measure | Mean (Auto) | Mean (Manual) | Pearson *r* | Spearman *rho* | ICC(2,1) | MAE |
|:--|--:|--:|--:|--:|--:|--:|
| AR | 3.779 | 3.765 | .958 | .916 | .958 | 0.153 |
| SR | 2.292 | 2.344 | .991 | .981 | .987 | 0.083 |
| MLR | 5.453 | 5.711 | .969 | .965 | .961 | 0.434 |
| MCPR | 0.105 | 0.101 | .966 | .984 | .965 | 0.013 |
| ECPR | 0.097 | 0.098 | .774 | .781 | .775 | 0.010 |
| PR | 0.202 | 0.199 | .954 | .966 | .952 | 0.015 |
| MCPD | 0.786 | 0.732 | .713 | .741 | .669 | 0.097 |
| ECPD | 1.125 | 1.052 | .916 | .905 | .904 | 0.141 |
| MPD | 0.947 | 0.906 | .934 | .868 | .920 | 0.093 |

#### ST2 detailed correlations (N=20)

| Measure | Mean (Auto) | Mean (Manual) | Pearson *r* | Spearman *rho* | ICC(2,1) | MAE |
|:--|--:|--:|--:|--:|--:|--:|
| AR | 3.641 | 3.620 | .950 | .860 | .948 | 0.161 |
| SR | 2.334 | 2.380 | .985 | .985 | .983 | 0.082 |
| MLR | 5.876 | 6.182 | .972 | .962 | .968 | 0.472 |
| MCPR | 0.111 | 0.104 | .977 | .974 | .960 | 0.010 |
| ECPR | 0.086 | 0.084 | .935 | .956 | .935 | 0.006 |
| PR | 0.197 | 0.189 | .977 | .956 | .965 | 0.013 |
| MCPD | 0.756 | 0.718 | .910 | .865 | .886 | 0.084 |
| ECPD | 1.057 | 1.045 | .963 | .920 | .965 | 0.092 |
| MPD | 0.893 | 0.869 | .981 | .958 | .970 | 0.056 |
### 2.4 Fresh-run reproducibility

- Run folder: `analysis/rq3_gaponly_neural_t050_freshrun_20260224/`
- Run log: `analysis/rq3_gaponly_neural_t050_freshrun_20260224/RUN_LOG.md`
- Final summary CSV (source): `analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_summary_quality39.csv`
- Final auto CAF CSV: `analysis/rq3_gaponly_neural_t050_freshrun_20260224/auto_caf_gold40_gaponly_neural_t050.csv`
- Final taskwise export (Overall/ST1/ST2): `analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_all_subsets.csv`
- Per-subset exports: `analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_overall.csv`, `analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_st1.csv`, `analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_st2.csv`

## 3. Comparison with prior work

| System | Task | Pause location κ | Clause boundary F1 |
|:--|:--|--:|--:|
| Matsuura et al. (2025) | Monologue (L2 English) | .613–.749 | — |
| Matsuura et al. (2025) | Dialogue (L2 English) | .596–.672 | — |
| Chen & Yoon (2011) | Monologue (L2 English) | — | .690 |
| **Current pipeline** | **Monologue (L2 English)** | **.840** | **.845** |

The current pipeline substantially exceeds Matsuura et al.'s pause location agreement (κ = .840 vs .613–.749) and Chen & Yoon's clause boundary detection (F1 = .845 vs .690). Direct comparison should be interpreted with caution due to differences in corpora, L1 backgrounds, proficiency levels, and gold standard construction methods.

---

## 4. Methodological notes

1. **Alignment choice matters.** An earlier version using `difflib.SequenceMatcher` (longest common subsequence) yielded κ = .828 for RQ1 — inflated because LCS skips substituted words and can "heal" disagreements at ASR error positions. The edit-distance approach (κ = .816) is stricter and standard in speech processing evaluation.

2. **ASR error propagation.** Files with WER > .20 show noticeably lower boundary agreement (mean κ = .72 vs .84 for WER ≤ .20). This confirms Knill et al.'s (2018, 2019) finding that ASR errors propagate to downstream NLP tasks.

3. **Gold standard quality.** The LLM-assisted annotation achieved κ = .914 against human adjudication on locked test files, supporting the validity of the 30 production files as gold standard.

4. **RQ3 cohort control.** EN final reporting uses quality-filtered Gold-39 (ST1=19, ST2=20) after excluding `ALL_139_M_PBR_ENG_ST1`; JA uses 40 files (ST1=20, ST2=20).

---

## 5. Files

| File | Location |
|:--|:--|
| RQ1 script | `analysis/rq1/run_rq1_gold.py` |
| RQ1 per-file CSV | `analysis/rq1/rq1_clause_boundary_gold.csv` |
| RQ2 script | `analysis/rq2/run_rq2_gold.py` |
| RQ2 per-file CSV | `analysis/rq2/rq2_pause_location_gold.csv` |
| RQ3 final summary CSV (source) | `analysis/rq3_gaponly_neural_t050_freshrun_20260224/probe/rq3_gaponly_neural_t050_probe_summary_quality39.csv` |
| RQ3 final auto CAF CSV | `analysis/rq3_gaponly_neural_t050_freshrun_20260224/auto_caf_gold40_gaponly_neural_t050.csv` |
| RQ3 taskwise CSV (all subsets) | `analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_all_subsets.csv` |
| RQ3 taskwise CSV (Overall) | `analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_overall.csv` |
| RQ3 taskwise CSV (ST1) | `analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_st1.csv` |
| RQ3 taskwise CSV (ST2) | `analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_st2.csv` |
| RQ3 run log | `analysis/rq3_gaponly_neural_t050_freshrun_20260224/RUN_LOG.md` |

**Run commands** (from `en/rq123_clean_release_20260223/`):
```bash
python analysis/rq1/run_rq1_gold.py
python analysis/rq2/run_rq2_gold.py
python ../postprocess_vad_filler_classifier_en.py --help
python scripts/caf_calculator_vad_classifier.py --help
python scripts/run_rq3_vad_classifier_probe_en.py --help
```
