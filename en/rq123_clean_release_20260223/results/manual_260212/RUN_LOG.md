# Manual CAF Pipeline Run Log

**Date:** 2026-02-12
**Purpose:** Re-run manual TextGrids with clause segmenter v3 (neural disfluency detection).

---

## Changes from manual_260207

1. **Clause segmentation**: Upgraded from v2 to v3
   - Replaced rule-based filler/repetition preprocessing with neural xlm-roberta-base disfluency detection model
   - All other clause rules (Vercellotti & Hall 2024) unchanged
   - Same Rule 8, aligner distance guard, single-word filter, progressive VBG fix, fragment recovery, verbless merger as v2

---

## Input Data

| Item | Path |
|------|------|
| **Manual TextGrids (preprocessed)** | `en/data/allsstar_full_manual/manual_textgrids/preprocessed/` |
| **Files processed** | 190 |

---

## Pipeline Steps

### Step 1: Clause Segmentation (V3)

```bash
python en/scripts/textgrid_caf_segmenter_v3.py \
  -i en/data/allsstar_full_manual/manual_textgrids/preprocessed \
  -o en/results/manual_260212/clauses
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
  en/results/manual_260212/clauses \
  --output en/results/manual_260212/caf_results_manual.csv
```

| Item | Value |
|------|-------|
| **Script** | `scripts/caf_calculator.py` |
| **Files** | 190 |

---

## CAF Statistics

| Measure | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| AR | 3.770 | 0.563 | 2.417 | 5.309 |
| SR | 2.345 | 0.635 | 0.401 | 4.021 |
| MLR | 6.082 | 2.279 | 2.330 | 16.380 |
| MCPR | 0.099 | 0.052 | 0.013 | 0.271 |
| ECPR | 0.090 | 0.022 | 0.031 | 0.168 |
| PR | 0.188 | 0.063 | 0.061 | 0.439 |
| MCPD | 0.715 | 0.208 | 0.353 | 1.581 |
| ECPD | 1.323 | 1.614 | 0.419 | 20.025 |
| MPD | 1.009 | 0.727 | 0.391 | 7.983 |

---

## Comparison: manual v2 (260207) vs manual v3 (260212)

| Measure | v2 (260207) | v3 (260212) |
|---------|-------------|-------------|
| AR | 3.770 | 3.770 |
| SR | 2.345 | 2.345 |
| MLR | 6.082 | 6.082 |
| MCPR | 0.100 | 0.099 |
| ECPR | 0.089 | 0.090 |
| PR | 0.188 | 0.188 |
| MCPD | 0.722 | 0.715 |
| ECPD | 1.302 | 1.323 |
| MPD | 1.009 | 1.009 |

Note: v2 and v3 produce nearly identical CAF values on manual TextGrids, confirming the neural disfluency model does not alter clause segmentation logic — it only changes how disfluent words are identified for removal.
