# V7 Verification Notes (Subagent Team Audit)

This note documents the final audit used for `draft/paper_draftv7.tex`.

## Team Members (Subagent Roles)

1. **Methods/Flow Auditor** (`peer-review` + `scientific-critical-thinking`)
2. **Statistics Verifier** (`statistical-analysis`)
3. **Claims Discipline Editor** (`scientific-writing`)

## 1) Architecture / Flow / Algorithm Alignment Checks

### A. Pipeline stage order and script correspondence
- Verified manuscript stage flow against release pipeline docs and scripts.
- Sources:
  - `README.md`
  - `en/rq123_clean_release_20260223/scripts/asr/asr_qwen3_mfa_en.py`
  - `ja/rq3_v16_clean_release_20260223/scripts/asr/asr_qwen3_mfa_ja_v2_spanfix_b.py`
  - `en/rq123_clean_release_20260223/scripts/textgrid_caf_segmenter_v3.py`
  - `ja/rq3_v16_clean_release_20260223/scripts/ja_clause_segmenter_v16.py`
  - `en/rq123_clean_release_20260223/scripts/caf_calculator_vad_classifier.py`
  - `ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_gap_classifier.py`

### B. Filler-augmented MFA constants and logic
- Confirmed constants and formula match manuscript:
  - `GAP_MIN=0.40`, `GAP_OFFSET=0.35`, `GAP_STEP=0.55`, `FILLER_MAX=3`
  - MFA options: `--beam 100 --retry_beam 400`
- Sources:
  - `en/rq123_clean_release_20260223/scripts/asr/asr_qwen3_mfa_en.py`
  - `ja/rq3_v16_clean_release_20260223/scripts/asr/asr_qwen3_mfa_ja_v2_spanfix_b.py`

### C. Sentence segmentation wording check
- Confirmed:
  - EN uses precomputed wtpsplit sentence JSON inputs (plus fallback behavior).
  - JA uses GiNZA sentence boundaries with additional re-splitting logic for long unpunctuated spans.
- Sources:
  - `en/rq123_clean_release_20260223/scripts/textgrid_caf_segmenter_v3.py`
  - `ja/rq3_v16_clean_release_20260223/scripts/ja_clause_segmenter_v16.py`

### D. Pause MCP/ECP classification logic check
- Confirmed implemented logic in both EN/JA calculators:
  1. If pause onset is within 150 ms of any clause end: ECP.
  2. Else, if pause midpoint is inside a clause span: MCP.
  3. Else: ECP.
- Sources:
  - `en/rq123_clean_release_20260223/scripts/caf_calculator_vad_classifier.py`
  - `ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_gap_classifier.py`

### E. Filler model performance values
- Confirmed values used in manuscript:
  - validation F1 `0.9409` -> `.941`
  - test F1 `0.9328` -> `.933`
- Sources:
  - `shared/filler_classifier/model_podcastfillers_neural_v1_full/metrics.json`
  - `shared/filler_classifier/model_podcastfillers_neural_v1_full/README.md`

## 2) Numeric Reconciliation Checks

### A. English RQ1 (from per-file CSV)
- Recomputed micro metrics from sums:
  - Overall: P `.848`, R `.842`, F1 `.845`, kappa `.816`, boundaries `1131`
  - ST1: F1 `.869`, kappa `.845`
  - ST2: F1 `.826`, kappa `.795`
- Recomputed macro/distribution stats used in prose:
  - mean F1 `.846` (SD `.090`)
  - mean kappa `.819` (SD `.105`)
  - F1 min `.618`, max `1.000`, median `.856`
  - mean WER `.121`
  - two lowest-kappa files:
    - `ALL_013_M_JPN_ENG_ST2`: kappa `.567`, WER `.199`
    - `ALL_102_M_SHS_ENG_ST2`: kappa `.578`, WER `.083`
- Source:
  - `en/rq123_clean_release_20260223/analysis/rq1/rq1_clause_boundary_gold.csv`

### B. English RQ2
- Confirmed manuscript table values against release report:
  - Overall: kappa `.840`, accuracy `.921`, MCP F1 `.929`, ECP F1 `.912`, pauses `1902`
  - ST1: kappa `.873`, accuracy `.937`, MCP F1 `.941`, ECP F1 `.932`
  - ST2: kappa `.815`, accuracy `.909`, MCP F1 `.920`, ECP F1 `.894`
- Recomputed per-file accuracy distribution from CSV:
  - macro mean `.922`, SD `.056`, median `.926`, range `.790`--`1.000`
  - perfect-accuracy files: `6`
  - minimum-accuracy file: `ALL_102_M_SHS_ENG_ST2` (`.7895`)
- Sources:
  - `en/rq123_clean_release_20260223/RQ1_RQ2_REPORT.md`
  - `en/rq123_clean_release_20260223/analysis/rq2/rq2_pause_location_gold.csv`

### C. English RQ3 (quality-filtered N=39)
- Confirmed all table values and duration-bias numbers:
  - mean Pearson r across 9 measures: `.936`
  - range: `.821`--`.988`
  - duration bias (`auto - manual`): MCPD `+0.046`, ECPD `+0.042`, MPD `+0.032`
- Sources:
  - `analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_all_subsets.csv`

### D. Japanese RQ3 (N=40)
- Confirmed all table values and duration-bias numbers:
  - mean Pearson r across 9 measures: `.953`
  - range: `.903`--`.992`
  - duration bias (`auto - manual`): MCPD `+0.141`, ECPD `+0.221`, MPD `+0.176`
  - weakest task-specific value: ECPR ST1 `r=.872`
- Sources:
  - `analysis_final_taskwise_correlations_20260224/ja/ja_correlation_final_vad_classifier_all_subsets.csv`

### E. Error-propagation check used in Discussion
- WER-stratified mean kappa (RQ1):
  - WER `> .20`: `.769` -> reported `.77`
  - WER `<= .20`: `.830` -> reported `.83`
- Source:
  - `en/rq123_clean_release_20260223/analysis/rq1/rq1_clause_boundary_gold.csv`

## 3) v7 Corrections Applied in This Pass

1. Corrected RQ2 Methods wording to exact implemented MCP/ECP rule (150 ms onset check + midpoint-in-clause check + ECP default).
2. Corrected RQ1 macro summary statistics in prose (SD/median values).
3. Corrected RQ2 per-file macro accuracy summary in prose.
4. Removed potentially overstated mean-correlation CI phrasings in abstract/results/conclusion.
5. Added/expanded inline `% Verified source(s): ...` comments around major result blocks in `paper_draftv7.tex`.

## 4) Build Check

- Compiled successfully:
  - `draft/paper_draftv7.pdf`
- Command used:
  - `LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error paper_draftv7.tex`
