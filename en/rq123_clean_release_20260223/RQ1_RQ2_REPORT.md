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

### 2.3 RQ3 — Concurrent validity of CAF measures

**To what extent do CAF measures from the fully automatic pipeline agree with those from the manual-transcript pipeline?**

#### Data generation

**Auto pipeline** (`results/qwen3_filler_mfa_beam100/`, see `RUN_LOG.md`):

| Step | Script | Details |
|:--|:--|:--|
| 0. ASR + alignment | `asr_qwen3_mfa_en.py` | Qwen3-ASR-1.7B → filler injection → MFA `english_us_arpa` (beam=100, retry_beam=400). 190 files, 36,011 words, 99.1% MFA rate. |
| 1. Clause segmentation | `scripts/textgrid_caf_segmenter_v3.py` | spaCy `en_ud_L1L2e_combined_trf` + neural disfluency (xlm-roberta-base) |
| 2. CAF calculation | `scripts/caf_calculator.py` | → `caf_results_beam100.csv` |

**Manual pipeline** (`results/manual_260212/`, see `RUN_LOG.md`):

| Step | Script | Details |
|:--|:--|:--|
| Input | — | Human-transcribed TextGrids from `data/allsstar_full_manual/manual_textgrids/preprocessed/` |
| 1. Clause segmentation | `scripts/textgrid_caf_segmenter_v3.py` | Same spaCy model + neural disfluency as auto |
| 2. CAF calculation | `scripts/caf_calculator.py` | → `caf_results_manual.csv` |

Both pipelines use **identical** clause segmentation and CAF calculation scripts. The only difference is the input: ASR transcript + MFA forced alignment vs human transcript + manual TextGrid timing.

- **Files:** 190 matched, **16 excluded** → **174 analysed** (ST1: 88, ST2: 86)

#### Exclusions

16 files were excluded due to manual transcription quality issues (from `RUN_LOG.md` exclusion analysis):

| Category | Files | Reason |
|:--|--:|:--|
| Preamble mismatch | 11 | Manual starts much later than ASR (interviewer speech not transcribed) |
| Incomplete manual | 5 | Mid-file gaps in manual transcription |
| **Total excluded** | **16** | |

High-WER files due to L1 accent were **not** excluded — the WER comes from ASR mishearing, not manual quality.

#### Overall results (N = 174)

| Measure | Mean (Auto) | Mean (Manual) | Pearson *r* | Spearman *ρ* | ICC(2,1) | Bias | MAE |
|:--|--:|--:|--:|--:|--:|--:|--:|
| AR | 3.787 | 3.762 | .942*** | .932*** | .938 | +0.025 | 0.168 |
| SR | 2.358 | 2.415 | .985*** | .976*** | .980 | −0.057 | 0.089 |
| MLR | 5.728 | 6.144 | .962*** | .964*** | .939 | −0.416 | 0.538 |
| MCPR | 0.107 | 0.098 | .960*** | .965*** | .945 | +0.009 | 0.014 |
| ECPR | 0.092 | 0.090 | .893*** | .883*** | .887 | +0.003 | 0.008 |
| PR | 0.199 | 0.187 | .961*** | .963*** | .946 | +0.012 | 0.017 |
| MCPD | 0.760 | 0.710 | .888*** | .864*** | .863 | +0.050 | 0.084 |
| ECPD | 1.054 | 1.047 | .934*** | .910*** | .934 | +0.007 | 0.111 |
| MPD | 0.900 | 0.879 | .954*** | .924*** | .950 | +0.021 | 0.065 |

\* *p* < .05, \*\* *p* < .01, \*\*\* *p* < .001. ICC interpretation (Koo & Li, 2016): < .50 poor, .50–.75 moderate, .75–.90 good, > .90 excellent.

#### Summary by measure category

| Category | Measures | Pearson *r* | ICC(2,1) | Assessment |
|:--|:--|--:|--:|:--|
| **Speed** | AR, SR | .942–.985 | .938–.980 | Excellent |
| **Pause rate** | MCPR, ECPR, PR | .893–.961 | .887–.946 | Good–Excellent |
| **Composite** | MLR | .962 | .939 | Excellent |
| **Pause duration** | MCPD, ECPD, MPD | .888–.954 | .863–.950 | Good–Excellent |

**Interpretation:**

1. **All nine measures show good-to-excellent concurrent validity** (ICC .863–.980). After excluding files with manual transcription quality issues, the auto pipeline reliably reproduces all CAF measures.

2. **Pause duration measures (MCPD, ECPD, MPD) now show good-to-excellent ICC** (.863–.950). The previous poor results (ICC .307–.598 without exclusions) were driven by files where the manual transcription was incomplete or misaligned — not by genuine pipeline disagreement.

3. **Systematic biases are small:** < 0.06 syl/s for speed measures, < 0.01 pauses/syl for rate measures, and < 0.05 s for pause durations. The largest bias is MLR (−0.42 syl), meaning the auto pipeline slightly underestimates mean run length.

4. **All correlations are significant at *p* < .001.**

#### Sensitivity check: Gold-40 vs full-quality set

To mirror the Japanese RQ3 design, we additionally ran RQ3 on the 40-file gold set from `annotation/selected_files.json`.

- `gold40_raw`: all 40 files (20 ST1 + 20 ST2)
- `gold40_quality_39`: gold files after the same quality exclusions used in the full-set RQ3 (`ALL_139_M_PBR_ENG_ST1` removed due to preamble mismatch: manual transcript starts substantially later than the auto timeline)
- `full_quality_174`: current full-set baseline

| Track | N | Pearson r range | ICC(2,1) range | Lowest measure |
|:--|--:|--:|--:|:--|
| gold40_raw | 40 | .816-.987 | .788-.985 | MCPD |
| gold40_quality_39 | 39 | .820-.988 | .796-.984 | MCPD |
| full_quality_174 | 174 | .888-.985 | .863-.980 | MCPD |

Interpretation:

1. The gold-40 subset reaches the same overall conclusion as the full-quality set: strong concurrent validity across all nine CAF measures.
2. The weak point is stable across tracks: MCPD is the lowest-agreement measure in every run.
3. The small-N gold subset shows wider spread and lower stability for pause-duration agreement, which is expected for noisy duration measures with limited sample size.
4. For the paper, report both: gold-40 as the strict validation subset and full-quality-174 as the operational/generalization result.

#### Detailed dual-track correlation tables (Overall)

`full_quality_174` is reported above. Full `gold40_raw` and `gold40_quality_39` tables are below.

**Gold40 raw (N = 40)**

| Measure | Mean (Auto) | Mean (Manual) | Pearson *r* | Spearman *rho* | ICC(2,1) | Bias | MAE |
|:--|--:|--:|--:|--:|--:|--:|--:|
| AR | 3.754 | 3.695 | .965*** | .932*** | .959 | +0.059 | 0.144 |
| SR | 2.301 | 2.346 | .987*** | .973*** | .985 | -0.046 | 0.082 |
| MLR | 5.689 | 5.934 | .966*** | .963*** | .962 | -0.245 | 0.473 |
| MCPR | 0.110 | 0.103 | .966*** | .972*** | .958 | +0.007 | 0.012 |
| ECPR | 0.093 | 0.091 | .890*** | .907*** | .888 | +0.002 | 0.007 |
| PR | 0.203 | 0.194 | .961*** | .961*** | .951 | +0.009 | 0.016 |
| MCPD | 0.771 | 0.720 | .816*** | .806*** | .788 | +0.051 | 0.093 |
| ECPD | 1.150 | 1.108 | .971*** | .923*** | .968 | +0.042 | 0.113 |
| MPD | 0.942 | 0.906 | .958*** | .911*** | .951 | +0.036 | 0.076 |

**Gold40 quality-filtered (N = 39)**

| Measure | Mean (Auto) | Mean (Manual) | Pearson *r* | Spearman *rho* | ICC(2,1) | Bias | MAE |
|:--|--:|--:|--:|--:|--:|--:|--:|
| AR | 3.753 | 3.691 | .966*** | .928*** | .959 | +0.062 | 0.146 |
| SR | 2.313 | 2.362 | .988*** | .978*** | .984 | -0.049 | 0.082 |
| MLR | 5.670 | 5.952 | .971*** | .968*** | .965 | -0.282 | 0.454 |
| MCPR | 0.111 | 0.103 | .970*** | .981*** | .960 | +0.008 | 0.012 |
| ECPR | 0.094 | 0.091 | .895*** | .906*** | .892 | +0.002 | 0.007 |
| PR | 0.204 | 0.194 | .969*** | .968*** | .956 | +0.010 | 0.015 |
| MCPD | 0.772 | 0.725 | .820*** | .813*** | .796 | +0.048 | 0.091 |
| ECPD | 1.083 | 1.048 | .947*** | .917*** | .945 | +0.035 | 0.107 |
| MPD | 0.918 | 0.887 | .959*** | .905*** | .946 | +0.031 | 0.073 |

#### Task-level dual-track tables (ST1/ST2)

Task counts by track:

- `full_quality_174`: ST1 = 88, ST2 = 86
- `gold40_raw`: ST1 = 20, ST2 = 20
- `gold40_quality_39`: ST1 = 19, ST2 = 20

**Gold40 raw - ST1 (N = 20)**

| Measure | Mean (Auto) | Mean (Manual) | Pearson *r* | Spearman *rho* | ICC(2,1) | Bias | MAE |
|:--|--:|--:|--:|--:|--:|--:|--:|
| AR | 3.821 | 3.770 | .966*** | .959*** | .963 | +0.050 | 0.145 |
| SR | 2.268 | 2.313 | .989*** | .958*** | .986 | -0.045 | 0.084 |
| MLR | 5.503 | 5.687 | .954*** | .952*** | .951 | -0.184 | 0.473 |
| MCPR | 0.106 | 0.102 | .965*** | .962*** | .964 | +0.005 | 0.013 |
| ECPR | 0.098 | 0.097 | .830*** | .802*** | .830 | +0.001 | 0.009 |
| PR | 0.205 | 0.199 | .952*** | .956*** | .951 | +0.006 | 0.017 |
| MCPD | 0.781 | 0.722 | .707*** | .736*** | .662 | +0.059 | 0.100 |
| ECPD | 1.242 | 1.171 | .977*** | .926*** | .969 | +0.072 | 0.135 |
| MPD | 0.988 | 0.942 | .945*** | .860*** | .938 | +0.046 | 0.095 |

**Gold40 raw - ST2 (N = 20)**

| Measure | Mean (Auto) | Mean (Manual) | Pearson *r* | Spearman *rho* | ICC(2,1) | Bias | MAE |
|:--|--:|--:|--:|--:|--:|--:|--:|
| AR | 3.687 | 3.620 | .965*** | .889*** | .954 | +0.067 | 0.144 |
| SR | 2.334 | 2.380 | .985*** | .985*** | .983 | -0.047 | 0.082 |
| MLR | 5.876 | 6.182 | .972*** | .962*** | .968 | -0.306 | 0.473 |
| MCPR | 0.114 | 0.104 | .978*** | .971*** | .951 | +0.010 | 0.011 |
| ECPR | 0.087 | 0.084 | .938*** | .956*** | .933 | +0.003 | 0.006 |
| PR | 0.201 | 0.189 | .976*** | .970*** | .954 | +0.013 | 0.014 |
| MCPD | 0.761 | 0.718 | .908*** | .858*** | .878 | +0.043 | 0.087 |
| ECPD | 1.058 | 1.045 | .964*** | .920*** | .965 | +0.013 | 0.091 |
| MPD | 0.895 | 0.869 | .982*** | .964*** | .969 | +0.026 | 0.058 |

**Gold40 quality-filtered - ST1 (N = 19)**

| Measure | Mean (Auto) | Mean (Manual) | Pearson *r* | Spearman *rho* | ICC(2,1) | Bias | MAE |
|:--|--:|--:|--:|--:|--:|--:|--:|
| AR | 3.822 | 3.765 | .967*** | .953*** | .963 | +0.057 | 0.149 |
| SR | 2.292 | 2.344 | .991*** | .981*** | .987 | -0.052 | 0.083 |
| MLR | 5.453 | 5.711 | .969*** | .965*** | .961 | -0.258 | 0.434 |
| MCPR | 0.107 | 0.101 | .970*** | .984*** | .968 | +0.006 | 0.013 |
| ECPR | 0.100 | 0.098 | .822*** | .774*** | .823 | +0.002 | 0.008 |
| PR | 0.207 | 0.199 | .966*** | .969*** | .960 | +0.008 | 0.016 |
| MCPD | 0.785 | 0.732 | .716*** | .741*** | .675 | +0.053 | 0.095 |
| ECPD | 1.109 | 1.052 | .931*** | .914*** | .925 | +0.057 | 0.124 |
| MPD | 0.943 | 0.906 | .938*** | .839*** | .926 | +0.037 | 0.089 |

**Gold40 quality-filtered - ST2 (N = 20)**

| Measure | Mean (Auto) | Mean (Manual) | Pearson *r* | Spearman *rho* | ICC(2,1) | Bias | MAE |
|:--|--:|--:|--:|--:|--:|--:|--:|
| AR | 3.687 | 3.620 | .965*** | .889*** | .954 | +0.067 | 0.144 |
| SR | 2.334 | 2.380 | .985*** | .985*** | .983 | -0.047 | 0.082 |
| MLR | 5.876 | 6.182 | .972*** | .962*** | .968 | -0.306 | 0.473 |
| MCPR | 0.114 | 0.104 | .978*** | .971*** | .951 | +0.010 | 0.011 |
| ECPR | 0.087 | 0.084 | .938*** | .956*** | .933 | +0.003 | 0.006 |
| PR | 0.201 | 0.189 | .976*** | .970*** | .954 | +0.013 | 0.014 |
| MCPD | 0.761 | 0.718 | .908*** | .858*** | .878 | +0.043 | 0.087 |
| ECPD | 1.058 | 1.045 | .964*** | .920*** | .965 | +0.013 | 0.091 |
| MPD | 0.895 | 0.869 | .982*** | .964*** | .969 | +0.026 | 0.058 |

Why MCPD is lower in ST1:

1. ST1 MCPD has higher error variance than ST2 on the gold-40 set (SD of auto-manual difference: 0.125 s in ST1 vs 0.093 s in ST2).
2. ST1 includes one strong outlier with large positive MCPD error (`ALL_034_M_HIN_ENG_ST1`: auto 1.175 vs manual 0.799, +0.376 s), which pulls down agreement.
3. Sensitivity check confirms this: removing that one outlier raises ST1 MCPD from `r=.707, ICC=.662` to `r=.764, ICC=.743`; removing the top three ST1 MCPD outliers raises it further to `r=.821, ICC=.818`.
4. Root mechanism is pause-merge error from ASR word timing, not a broad clause-boundary failure: in `ALL_034_M_HIN_ENG_ST1`, both auto and manual place the same clause span (`and there were small birds on his head`), but ASR misses filler words (`uh`) and merges hesitation gaps into a single long mid-clause pause (auto 6.96 s vs manual 3.92 s + 1.48 s split by `uh`). This inflates the MCPD mean.
5. Across ST1, MCPD absolute error tracks this long-pause mismatch pattern (`Spearman rho = .777` between MCPD absolute error and max mid-pause gap), indicating that extreme merged in-clause pauses are the main driver.

### 2.4 VAD Sensitivity Probe (Gold-40)

To test whether the MCPD drift can be reduced, we ran an audio-based pause refinement probe on the same 40-file gold set.

Method:

1. Input pause candidates are empty-word intervals (`>= 0.25 s`) from the auto TextGrid `words` tier.
2. For each pause candidate, detect non-silent islands from the corresponding WAV segment (`librosa.effects.split`).
3. If speech occupancy is plausible (not too sparse / not fully dense), split the original pause into silent sub-intervals between detected speech islands.
4. Recompute CAF from the refined pause list and compare against manual CAF.

Tuned parameters used in the final probe:
- `vad_top_db = 30`
- `vad_min_occupancy = 0.20`
- `vad_min_voiced = 0.15 s`
- `vad_merge_gap = 0.10 s`

Scripts:
- `scripts/caf_calculator_vad.py`
- `analysis/rq3/run_rq3_vad_probe.py`

Probe outputs:
- `analysis/rq3/auto_caf_gold40_vad_tuned.csv`
- `analysis/rq3/rq3_vad_probe_summary.csv`
- `analysis/rq3/rq3_vad_probe_mcpd_file_deltas.csv`
- `analysis/rq3/VAD_PROBE_RUN_LOG.md`

#### Overall change (Gold-40): baseline vs VAD-tuned

| Measure | Pearson r (base -> VAD) | ICC(2,1) (base -> VAD) | MAE (base -> VAD) |
|:--|--:|--:|--:|
| AR | .965 -> .957 | .959 -> .930 | 0.1442 -> 0.1864 |
| SR | .987 -> .987 | .985 -> .985 | 0.0825 -> 0.0825 |
| MLR | .966 -> .966 | .962 -> .962 | 0.4727 -> 0.4727 |
| MCPR | .966 -> .974 | .958 -> .958 | 0.0120 -> 0.0118 |
| ECPR | .890 -> .880 | .888 -> .853 | 0.0074 -> 0.0085 |
| PR | .961 -> .969 | .951 -> .941 | 0.0157 -> 0.0179 |
| MCPD | .816 -> .895 | .788 -> .881 | 0.0931 -> 0.0698 |
| ECPD | .971 -> .967 | .968 -> .937 | 0.1132 -> 0.1347 |
| MPD | .958 -> .974 | .951 -> .940 | 0.0764 -> 0.0769 |

#### Task-specific delta (VAD - baseline)

| Measure | ST1 Delta r | ST1 Delta ICC | ST1 Delta MAE | ST2 Delta r | ST2 Delta ICC | ST2 Delta MAE |
|:--|--:|--:|--:|--:|--:|--:|
| AR | -0.006 | -0.027 | +0.055 | -0.013 | -0.032 | +0.029 |
| SR | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| MLR | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| MCPR | +0.012 | +0.006 | -0.002 | -0.004 | -0.010 | +0.001 |
| ECPR | -0.023 | -0.039 | +0.001 | -0.013 | -0.046 | +0.001 |
| PR | +0.020 | +0.004 | +0.000 | -0.004 | -0.024 | +0.004 |
| MCPD | +0.135 | +0.170 | -0.030 | +0.019 | +0.033 | -0.017 |
| ECPD | +0.001 | -0.008 | -0.011 | -0.020 | -0.083 | +0.054 |
| MPD | +0.032 | +0.016 | -0.016 | -0.009 | -0.047 | +0.017 |

Interpretation:

1. The probe confirms the diagnosed mechanism: splitting merged in-clause pauses improves MCPD agreement substantially, especially ST1.
2. Gains are targeted: MCPD improves in both tasks, strongest in ST1.
3. Tradeoff remains: some non-target measures (notably ECPD, and AR) lose agreement.
4. For reporting, baseline remains the main pipeline result, and VAD is best framed as a targeted sensitivity analysis for pause-duration drift.

---

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

4. **Exclusion impact on RQ3.** Without exclusions (N = 190), pause duration ICCs were poor (.307–.598). After removing 16 files with preamble/incomplete manual issues (N = 174), all ICCs rose to .863–.980. This confirms the low agreement was caused by manual transcription defects, not pipeline failure. Researchers should verify manual transcription quality before using it as a reference standard.

---

## 5. Files

| File | Location |
|:--|:--|
| RQ1 script | `analysis/rq1/run_rq1_gold.py` |
| RQ1 per-file CSV | `analysis/rq1/rq1_clause_boundary_gold.csv` |
| RQ2 script | `analysis/rq2/run_rq2_gold.py` |
| RQ2 per-file CSV | `analysis/rq2/rq2_pause_location_gold.csv` |
| RQ3 script | `analysis/rq3/run_rq3_validity.py` |
| RQ3 summary CSV | `analysis/rq3/rq3_concurrent_validity.csv` |
| RQ3 file-level CSV | `analysis/rq3/rq3_file_level.csv` |
| RQ3 dual-track script | `analysis/rq3/run_rq3_validity_dualtrack.py` |
| RQ3 dual-track summary CSV | `analysis/rq3/rq3_dualtrack_summary.csv` |
| RQ3 dual-track membership CSV | `analysis/rq3/rq3_dualtrack_file_membership.csv` |
| VAD CAF script | `scripts/caf_calculator_vad.py` |
| RQ3 VAD probe script | `analysis/rq3/run_rq3_vad_probe.py` |
| VAD-tuned auto CAF (gold40) | `analysis/rq3/auto_caf_gold40_vad_tuned.csv` |
| VAD probe summary CSV | `analysis/rq3/rq3_vad_probe_summary.csv` |
| VAD probe file deltas CSV | `analysis/rq3/rq3_vad_probe_mcpd_file_deltas.csv` |
| VAD probe run log | `analysis/rq3/VAD_PROBE_RUN_LOG.md` |

**Run commands** (from `en/rq123_clean_release_20260223/`):
```bash
python analysis/rq1/run_rq1_gold.py
python analysis/rq2/run_rq2_gold.py
python analysis/rq3/run_rq3_validity.py
python analysis/rq3/run_rq3_validity_dualtrack.py
python scripts/caf_calculator_vad.py results/qwen3_filler_mfa_beam100/clauses --use-audio-vad --audio-dir ../data/allsstar_full_manual/wav --file-list-json annotation/selected_files.json --file-list-key all_selected --vad-top-db 30 --vad-min-occupancy 0.2 --vad-min-voiced 0.15 --vad-merge-gap 0.1 --output analysis/rq3/auto_caf_gold40_vad_tuned.csv
python analysis/rq3/run_rq3_vad_probe.py --auto-vad-csv analysis/rq3/auto_caf_gold40_vad_tuned.csv --out-dir analysis/rq3
```

---

## 6. Reproducibility Update (2026-02-24, Unified Gap-Only Candidate Script)

To enforce cross-language consistency, candidate extraction was rerun using the same script for both EN and JA:

- Candidate script (shared): `en/postprocess_vad_filler_classifier_en.py`
- New setting: `--gap-only` (score each ASR gap as one full candidate span; no VAD island split)
- Classifier: `shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt`
- Neural training code/reference:
  - `shared/filler_classifier/train_podcastfillers_neural_classifier.py`
  - `shared/filler_classifier/README.md` (training setup + paper reference)

EN gap-only pipeline:
1. Candidate extraction: `en/postprocess_vad_filler_classifier_en.py --gap-only ...`
2. CAF computation: `en/rq123_clean_release_20260223/scripts/caf_calculator_vad_classifier.py`
3. Correlation probe: `en/rq123_clean_release_20260223/analysis/rq3/run_rq3_vad_classifier_probe.py`

Run log and outputs:
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/RUN_LOG.md`
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/auto_caf_gold40_gaponly_neural_t050.csv`
- `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/probe/rq3_gaponly_neural_t050_probe_summary.csv`

MCPD result (Overall, baseline -> prior VAD -> gap-only neural):
- Pearson r: `0.816 -> 0.895 -> 0.817`
- ICC(2,1): `0.788 -> 0.881 -> 0.790`
- MAE: `0.093 -> 0.070 -> 0.092`

Conclusion for EN:
- The unified gap-only neural candidate flow did not improve over the prior EN VAD-tuned probe and was only near baseline.
- For EN sensitivity analyses, the best-performing variant remains the pause-local VAD+classifier hybrid run documented under `analysis/rq3/probe_pausevadclf_hybrid_best_neural/`.

Fresh rerun from filler detection (2026-02-24):
- A full fresh rerun (candidate extraction -> CAF -> probe) was executed in:
  - `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/`
  - `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/RUN_LOG.md`
  - `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_20260224/probe/rq3_gaponly_neural_t050_probe_summary.csv`
- The fresh rerun reproduced the same MCPD summary pattern:
  - Overall: `r 0.816 -> 0.895 -> 0.817`, `ICC 0.788 -> 0.881 -> 0.790`, `MAE 0.093 -> 0.070 -> 0.092`
  - ST1: `r 0.707 -> 0.842 -> 0.704`, `ICC 0.662 -> 0.832 -> 0.656`, `MAE 0.100 -> 0.070 -> 0.101`
  - ST2: `r 0.908 -> 0.927 -> 0.910`, `ICC 0.878 -> 0.911 -> 0.886`, `MAE 0.086 -> 0.070 -> 0.084`
