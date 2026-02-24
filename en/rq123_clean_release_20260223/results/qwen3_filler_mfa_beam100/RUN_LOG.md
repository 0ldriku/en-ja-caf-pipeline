# Qwen3 + Filler-Augmented MFA (beam=100) CAF Pipeline Run Log

**Date:** 2026-02-12
**Purpose:** Full EN pipeline run with Qwen3-ASR + filler-augmented MFA (beam=100) + clause segmentation v3.

---

## Changes from qwen3_filler_mfa_260212 (default beam)

1. **MFA beam fix**: Added `--beam 100 --retry_beam 400` to MFA alignment
   - Default beam caused alignment drift on long files (10–28s trailing silence on ~20 files)
   - High beam prevents Kaldi from pruning good alignment paths for L2 accented speech
   - Result: 99.1% MFA alignment (35,690/36,011 words), 0 ASR gap outliers

2. **Pipeline simplification**: Removed segmentation logic (energy-based, word-gap, proportional)
   - Full-file MFA with high beam + filler injection is sufficient
   - No segmentation, no drift correction needed

---

## Input Data

| Item | Path |
|------|------|
| **Audio files** | `en/data/allsstar_full_manual/wav/` |
| **ASR + MFA output** | `en/results/qwen3_filler_mfa_beam100/` |
| **Input TextGrids (clean)** | `en/results/qwen3_filler_mfa_beam100/textgrids_clean/` |
| **Files processed** | 190 |

---

## Pipeline Steps

### Step 0: ASR + Filler-Augmented MFA (beam=100)

```bash
conda activate qwen3-asr
python en/asr_qwen3_mfa_en.py \
  -i en/data/allsstar_full_manual/wav \
  -o en/results/qwen3_filler_mfa_beam100
```

| Item | Value |
|------|-------|
| **Script** | `en/asr_qwen3_mfa_en.py` |
| **ASR model** | `Qwen/Qwen3-ASR-1.7B` |
| **Aligner model** | `Qwen/Qwen3-ForcedAligner-0.6B` |
| **MFA acoustic** | `english_us_arpa` |
| **MFA dictionary** | `english_us_arpa` |
| **MFA flags** | `--beam 100 --retry_beam 400` |
| **Conda env** | `qwen3-asr` (ASR) + `mfa` (alignment subprocess) |
| **Files** | 190 |
| **Total words** | 36,011 (35,690 MFA, 321 rough) |
| **MFA rate** | 99.1% |
| **Fillers injected** | 10,796 |
| **Output** | `textgrids/`, `textgrids_clean/`, `json/` |

---

### Step 1: Clause Segmentation (V3)

```bash
python en/scripts/textgrid_caf_segmenter_v3.py \
  -i en/results/qwen3_filler_mfa_beam100/textgrids_clean \
  -o en/results/qwen3_filler_mfa_beam100/clauses
```

| Item | Value |
|------|-------|
| **Script** | `scripts/textgrid_caf_segmenter_v3.py` |
| **spaCy model** | `en_ud_L1L2e_combined_trf` |
| **Disfluency model** | xlm-roberta-base (neural, in-project) |
| **Conda env** | `qwen3-asr` |
| **Files** | 190 |

---

### Step 2: CAF Calculation

```bash
python en/scripts/caf_calculator.py \
  en/results/qwen3_filler_mfa_beam100/clauses \
  --output en/results/qwen3_filler_mfa_beam100/caf_results_beam100.csv
```

| Item | Value |
|------|-------|
| **Script** | `scripts/caf_calculator.py` |
| **Files** | 190 |

---

## CAF Statistics

| Measure | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| AR | 3.797 | 0.612 | 2.399 | 5.694 |
| SR | 2.318 | 0.608 | 0.730 | 4.003 |
| MLR | 5.701 | 2.021 | 2.190 | 14.570 |
| MCPR | 0.108 | 0.053 | 0.014 | 0.308 |
| ECPR | 0.091 | 0.022 | 0.040 | 0.160 |
| PR | 0.199 | 0.065 | 0.069 | 0.468 |
| MCPD | 0.809 | 0.383 | 0.425 | 4.800 |
| ECPD | 1.131 | 0.611 | 0.426 | 4.661 |
| MPD | 0.953 | 0.353 | 0.434 | 2.568 |

---

## Comparison with Previous Runs

### CAF Mean Comparison

| Measure | manual_260207 | auto_260207 (CW) | qwen3_260212 | **qwen3_beam100** |
|---------|---------------|-------------------|--------------|-------------------|
| AR | 3.770 | 3.286 | 3.874 | **3.797** |
| SR | 2.345 | 2.361 | 2.318 | **2.318** |
| MLR | 6.082 | 6.969 | 5.660 | **5.701** |
| MCPR | 0.100 | 0.086 | 0.112 | **0.108** |
| ECPR | 0.089 | 0.076 | 0.088 | **0.091** |
| PR | 0.188 | 0.162 | 0.201 | **0.199** |
| MCPD | 0.722 | 0.727 | 0.817 | **0.809** |
| ECPD | 1.302 | 0.980 | 1.192 | **1.131** |
| MPD | 1.009 | 0.855 | 0.972 | **0.953** |

Notes:
- manual_260207: Manual TextGrids, segmenter v2, n=190
- auto_260207: CrisperWhisper ASR, segmenter v2, n=207
- qwen3_260212: Qwen3 ASR + filler MFA (default beam), segmenter v3, n=190
- **qwen3_beam100**: Qwen3 ASR + filler MFA (beam=100), segmenter v3, n=190

---

## Exclusion Analysis

Scripts: `analysis/show_excluded_files.py`, `analysis/_check_manual_gaps.py`

### Category 1: Preamble Mismatch — 11 files

Manual transcription starts much later than ASR (interviewer/preamble speech not transcribed in manual).
Detected by first-word timing difference >5s.

| File | L1 | Manual start | Auto start | Diff | ASR words in gap |
|------|----|-------------|-----------|------|-----------------|
| ALL_142_M_RUS_ENG_ST1 | RUS | 354.2s | 2.6s | 351.7s | 358 |
| ALL_141_M_HEB_ENG_ST1 | HEB | 107.6s | 0.2s | 107.4s | 86 |
| ALL_139_M_PBR_ENG_ST2 | PBR | 75.0s | 0.0s | 75.0s | 23 |
| ALL_140_M_RUS_ENG_ST1 | RUS | 60.4s | 0.6s | 59.8s | 73 |
| ALL_144_M_RUS_ENG_ST1 | RUS | 46.8s | 0.8s | 46.0s | — |
| ALL_139_M_PBR_ENG_ST1 | PBR | 42.5s | 0.0s | 42.5s | 23 |
| ALL_143_M_FAR_ENG_ST2 | FAR | 44.8s | 5.6s | 39.1s | 46 |
| ALL_140_M_RUS_ENG_ST2 | RUS | 40.4s | 1.4s | 39.0s | 68 |
| ALL_145_F_VIE_ENG_ST1 | VIE | 20.4s | 0.2s | 20.2s | — |
| ALL_143_M_FAR_ENG_ST1 | FAR | 17.2s | 0.1s | 17.2s | 30 |
| ALL_142_M_RUS_ENG_ST2 | RUS | 77.2s | 70.2s | 6.9s | — |

### Category 2: Incomplete Manual Transcription — 5 files

Manual transcription has mid-file gaps where spoken words were not transcribed.
Detected by finding manual silence gaps >0.5s containing >10 ASR words.
Verified in Praat: comparison TextGrids in `analysis/comparisons/incomplete/`.

| File | L1 | WER | Gap location | ASR words missed | Issue |
|------|----|-----|-------------|-----------------|-------|
| ALL_093_M_TUR_ENG_ST2 | TUR | 33% | 10–23s, 109–116s | 34 | Two mid-file spans untranscribed |
| ALL_086_F_CCT_ENG_ST2 | CCT | 35% | 131–163s | 31 | Final 30s of speech untranscribed |
| ALL_141_M_HEB_ENG_ST2 | HEB | 27% | 100–106s | 6 | Also preamble mismatch |
| ALL_110_M_KOR_ENG_ST2 | KOR | 26% | 97–103s | 8 | End-of-story missed |
| ALL_021_M_CMN_ENG_ST2 | CMN | 26% | 196–200s | 9 | Mid-file gap |

### Category 3: High WER (accent) — NOT excluded

These files have high WER (>25%) due to heavy L1 accent but the manual transcription is complete.
The WER comes from ASR mishearing accented words, not from alignment or manual quality issues.
Kept in the analysis. Comparison TextGrids in `analysis/comparisons/accent/`.

| File | L1 | WER | Match rate | Median err |
|------|----|-----|-----------|-----------|
| ALL_076_M_CCT_ENG_ST1 | CCT | 35% | 65% | 10ms |
| ALL_043_M_CMN_ENG_ST2 | CMN | 31% | 70% | 20ms |
| ALL_032_M_CMN_ENG_ST1 | CMN | 30% | 74% | 10ms |
| ALL_033_M_CMN_ENG_ST1 | CMN | 28% | 72% | 10ms |
| ALL_086_F_CCT_ENG_ST1 | CCT | 27% | 74% | 20ms |
| + 9 more files | CMN/KOR | 25–28% | 72–90% | 10–30ms |

**ASR Gaps (>5s, >10 words) — 0 files** ✅

beam=100 eliminated all ASR gap outliers (previously 5 files with default beam).

### Exclusion Summary

| Category | Files | Reason |
|----------|-------|--------|
| Preamble | 11 | Manual starts late (interviewer speech) |
| Incomplete manual | 5 | Mid-file gaps in manual transcription |
| **Overlap** | **-1** | ALL_141_M_HEB_ENG_ST2 in both |
| **Total excluded** | **15** | |
| **Remaining** | **174** | (ST1: 88, ST2: 86) |

---

## Correlation: Qwen3 beam=100 Auto vs Manual (Concurrent Validity)

Correlation script: `analysis/run_correlation.py`
Excluded: preamble (11) + incomplete manual (5) − overlap (1) = **15 files**

### Results (n=174, all p < .001)

| Measure | Overall | ST1 (n=88) | ST2 (n=86) |
|---------|---------|------------|------------|
| AR | 0.942*** | 0.938*** | 0.949*** |
| SR | 0.985*** | 0.986*** | 0.983*** |
| MLR | 0.962*** | 0.969*** | 0.955*** |
| MCPR | 0.960*** | 0.962*** | 0.960*** |
| ECPR | 0.893*** | 0.868*** | 0.909*** |
| PR | 0.961*** | 0.960*** | 0.965*** |
| MCPD | 0.888*** | 0.891*** | 0.889*** |
| ECPD | 0.934*** | 0.952*** | 0.920*** |
| MPD | 0.954*** | 0.954*** | 0.953*** |
| **Mean** | **0.942** | **0.942** | **0.943** |
| **Min** | **0.888** | **0.868** | **0.889** |

### Comparison vs Previous Runs

| | CrisperWhisper (260207) | Qwen3 default beam (260212) | **Qwen3 beam=100** |
|--|-------------------------|----------------------------|-------------------|
| **Mean r** | 0.925 | 0.936 | **0.942** |
| **Min r** | 0.849 | 0.877 | **0.888** |
| **Excluded** | 28 | 12 | **15** |
| **n** | 157 | 178 | **174** |
| **ASR gaps** | — | 0 | **0** |

Notes:
- Qwen3 beam=100 achieves the highest mean r (0.942) and highest min r (0.888)
- 3 more files excluded vs previous run (incomplete manual transcription detected)
- n=174 still larger than CrisperWhisper (n=157) due to fewer preamble issues
- All 9 measures show r > 0.87, all p < .001
- Results consistent across ST1 and ST2 tasks

---

## Alignment Quality

### ALL_085_F_CCT_ENG_ST2 (161s, Cantonese L1 — worst-case file)

| Method | MFA/Total | Matched | Median | P90 | Trailing |
|--------|-----------|---------|--------|-----|----------|
| Default beam | 80/241 | 73/272 | — | — | 18.6s |
| **beam=100** | **241/241** | **208/272** | **20ms** | **95ms** | **0.9s** |
| CrisperWhisper | — | 224/272 | 30ms | 115ms | — |

### Overall (190 files)

| Item | Default beam | **beam=100** |
|------|-------------|-------------|
| Total words | 36,011 | 36,011 |
| MFA aligned | ~34,000 | **35,690 (99.1%)** |
| Rough fallback | ~2,000 | **321 (0.9%)** |
| Fillers injected | ~10,800 | 10,796 |
