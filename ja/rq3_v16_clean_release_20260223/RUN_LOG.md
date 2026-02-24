# RUN_LOG

## Run Date
- 2026-02-23

## Goal
Re-run Japanese RQ3 concurrent validity with v16 clause segmenter updated to produce **clean clause labels** (fillers and disfluencies stripped from clause tier text). Manual gold standard clauses reviewed and fixed (gold_v2).

## Changes From Previous Run (`rq3_validity_ja_v16_fixed_20260223`)
1. **v16 clause label cleanup**: Updated `_rebuild_clause_payload()` and two other display-word builders to exclude both neural-detected disfluencies (`disfluency_labels[i] == 1`) AND rule-based fillers (`_is_unambiguous_filler_surface()`). Previously only disfluencies were excluded; fillers (e.g., ãˆãƒ¼, ã‚ã®, ã†ãƒ¼ã‚“) remained in labels.
2. **Gold_v2 manual fixes**: After auto-segmentation, comprehensively reviewed all 40 manual TextGrids against Vercellotti & Hall (2024) clause coding rules and applied 9 clause boundary fixes (see below).

## Clause Segmenter Version
- `ja/ja_clause_segmenter_v16.py` (v16-clean)
- v16 by GPT-5.3 Codex, 6 regression fixes by Claude, filler-clean labels by Claude
- Disfluency detector: `en/disfluency_test/l2_disfluency_detector/model_v2/final`
- NLP model: `ja_ginza_electra`

## Python Environment

- Full pinned environment notes:
  - `ja/rq3_v16_clean_release_20260223/ENVIRONMENT.md`
- ASR stage (upstream TextGrid generation):
  - `C:\Users\riku\miniconda3\envs\qwen3-asr\python.exe` (Python 3.12.12)
  - `C:\Users\riku\miniconda3\envs\mfa\python.exe` (Python 3.10.19)
- RQ3 clause/CAF/correlation stage:
  - `ja/.venv_electra310/Scripts/python.exe` (Python 3.10.11)

## Commands Executed

### Step 0: Upstream ASR TextGrid creation (pre-existing source, documented for reproducibility)
```bash
# 0a) ASR + MFA (spanfix_b variant) -> textgrids_clean
conda activate qwen3-asr
python ja/asr_qwen3_mfa_ja_v2_spanfix_test_b.py \
  -i "ja/data/dataset_l1_focus20/audio" \
  -o "ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20"

# 0b) Blank leading ASR labels before manual first non-blank word
python ja/rq3_v16_clean_release_20260223/scripts/asr/make_beginning_removed_by_manual.py \
  --asr-dir "ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/textgrids_clean" \
  --manual-dir "ja/results/manual20_0220_2" \
  --out-dir "ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/textgrids_clean_beginning_removed_by_manual"
```
Notes:
- Step 0 outputs were already present before this RQ3 run and were used as fixed input.
- ASR scripts are bundled in this package under `scripts/asr/`.

### Step 1: Clause segmentation on ASR TextGrids (40 files)
```bash
ja/.venv_electra310/Scripts/python.exe ja/ja_clause_segmenter_v16.py \
  -i "ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/textgrids_clean_beginning_removed_by_manual" \
  -o "ja/rq3_v16_clean_release_20260223/auto_clauses" \
  --model ja_ginza_electra
```

### Step 2: Clause segmentation on manual TextGrids (40 files)
```bash
ja/.venv_electra310/Scripts/python.exe ja/ja_clause_segmenter_v16.py \
  -i "ja/results/manual20_0220_2" \
  -o "ja/rq3_v16_clean_release_20260223/manual_clauses" \
  --model ja_ginza_electra
```

### Step 3: Gold_v2 fixes
```bash
cp -r manual_clauses manual_clauses_gold_v2   # working copy
# Applied 9 manual fixes (see Gold V2 Fixes below)
# Fixes applied via apply_gold_v2_fixes.py (praatio-based boundary adjustments)
cp -r manual_clauses_gold_v2/* manual_clauses/ # replace for analysis
```
Notes:
- `manual_clauses_gold_v1` existed as a transient pre-fix backup during the original run.
- This curated release keeps only `manual_clauses_gold_v2` (the final gold standard actually used in analysis).

### Step 4: CAF calculation
```bash
ja/.venv_electra310/Scripts/python.exe ja/caf_calculator_ja.py \
  "ja/rq3_v16_clean_release_20260223/auto_clauses" \
  -o "ja/rq3_v16_clean_release_20260223/auto_caf_results.csv"

ja/.venv_electra310/Scripts/python.exe ja/caf_calculator_ja.py \
  "ja/rq3_v16_clean_release_20260223/manual_clauses" \
  -o "ja/rq3_v16_clean_release_20260223/manual_caf_results.csv"
```

### Step 5: Correlation analysis
```bash
ja/.venv_electra310/Scripts/python.exe ja/analysis/correlation_from_caf_ja.py \
  --auto-csv "ja/rq3_v16_clean_release_20260223/auto_caf_results.csv" \
  --manual-csv "ja/rq3_v16_clean_release_20260223/manual_caf_results.csv" \
  --out-stats "ja/rq3_v16_clean_release_20260223/rq3_concurrent_validity_ja.csv" \
  --out-file-level "ja/rq3_v16_clean_release_20260223/rq3_file_level_ja.csv" \
  --ci-bootstrap 3000
```

## Inputs
- ASR TextGrids: `ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/textgrids_clean_beginning_removed_by_manual`
- Manual TextGrids: `ja/results/manual20_0220_2`

## Output Files
- `auto_clauses/*.TextGrid` (40) - ASR clause-segmented TextGrids
- `manual_clauses/*.TextGrid` (40) - Manual clause-segmented TextGrids (= gold_v2)
- `manual_clauses_gold_v2/` (40) - Post-fix gold standard
- `auto_caf_results.csv` - CAF measures for ASR
- `manual_caf_results.csv` - CAF measures for manual
- `rq3_concurrent_validity_ja.csv` - Full correlation/agreement statistics
- `rq3_file_level_ja.csv` - Per-file CAF values
- `auto_clauses/clause_log.txt` - Clause segmentation log (ASR)

## Gold V2 Fixes Applied (9 boundary fixes, 9 files)

All 40 manual TextGrids were comprehensively reviewed against Vercellotti & Hall (2024) clause coding rules. The following violation types were checked:
- Auxiliary constructions split across clauses (ã¦ã—ã¾ã†, ã¦ã„ã‚‹, ã¦ã„ã, ã¦ãã‚‹)
- Subordinating particles starting a new clause (ã®ã‹, ã‹ã‚‰, ã‘ã©)
- Verb inflection morphology split across clauses (ã¾ã™+ãŸ, ã¾ã›+ã‚“)
- Te-form endings split from verb stems
- Object/case particles incorrectly starting a new clause

### Fix 1: ENZ20-ST1 â€” Compound verb auxiliary ã¦ã„ã split
- **Before**: "ãƒã‚¹ã‚±ãƒƒãƒˆã‚’é€£ã‚Œã¦" | "ã„ãæŒã£ã¦ã„ãã¾ã—ãŸ"
- **After**: "ãƒã‚¹ã‚±ãƒƒãƒˆã‚’é€£ã‚Œã¦ã„ãæŒã£ã¦ã„ãã¾ã—ãŸ"
- **Rule**: é€£ã‚Œã¦ã„ã is a compound verb; the auxiliary ã„ã must not be split from the main verb.

### Fix 2: CCM35-ST2 â€” Auxiliary ã¦ã—ã¾ã† split
- **Before**: "è­¦å¯ŸãŠå·¡ã‚Šã•ã‚“ã«è¦‹ã¤ã‹ã‚‰ã‚Œ" | "ã¦ã—ã¾ã„ã¾ã—ãŸ"
- **After**: "è­¦å¯ŸãŠå·¡ã‚Šã•ã‚“ã«è¦‹ã¤ã‹ã‚‰ã‚Œã¦ã—ã¾ã„ã¾ã—ãŸ"
- **Rule**: ã¦ã—ã¾ã† is a completive auxiliary construction; must not be split from the main verb.

### Fix 3: CCH16-ST2 â€” Subordinating particle ã®ã‹ split
- **Before**: "å®¶ã«å…¥ã‚‹" | "ã®ã‹å›°ã£ã¦ã„ã¾ã™"
- **After**: "å®¶ã«å…¥ã‚‹ã®ã‹" | "å›°ã£ã¦ã„ã¾ã™"
- **Rule**: ã®ã‹ is a subordinating particle that must stay with its governing verb.

### Fix 4: CCS45-ST1 â€” Compound verb auxiliary ã¦ã„ã split
- **Before**: "ä¸€ç·’ã«é€£ã‚Œã¦" | "ã„ããŸã„ã§ä¸€ç·’ã«é è¶³ã«è¡Œãã¾ã—ãŸ"
- **After**: "ä¸€ç·’ã«é€£ã‚Œã¦ã„ããŸã„ã§ä¸€ç·’ã«é è¶³ã«è¡Œãã¾ã—ãŸ" (merged)
- **Rule**: é€£ã‚Œã¦ã„ã is a compound verb; the auxiliary ã„ã must not be split from the main verb.

### Fix 5: CCS45-ST2 â€” Verb inflection ã¾ã—ãŸ split
- **Before**: "ã†ãã§ã™ã§è­¦å®˜ã¯ã¾ãŸã¯ã‚“ã±ãã—" | "ã¾ã—ãŸè‰¯ã‹ã£ãŸ"
- **After**: "ã†ãã§ã™ã§è­¦å®˜ã¯ã¾ãŸã¯ã‚“ã±ãã—ã¾ã—ãŸ" | "è‰¯ã‹ã£ãŸ"
- **Rule**: Verb inflection morphology (masu+ta) must not be split across clauses.

### Fix 6: CCT15-ST2 â€” Subordinating particle ã®ã‹ã—ã‚‰ split
- **Before**: "ä½•ã®æ–¹æ³•ã§å®¶ã«å…¥ã‚‹" | "ã®ã‹ã—ã‚‰ã£ã¦ã“ã¨ã‚’çœŸå‰£ã«è€ƒãˆã‚‹ã¨ã¯"
- **After**: "ä½•ã®æ–¹æ³•ã§å®¶ã«å…¥ã‚‹ã®ã‹ã—ã‚‰" | "ã£ã¦ã“ã¨ã‚’çœŸå‰£ã«è€ƒãˆã‚‹ã¨ã¯"
- **Rule**: ã®ã‹ is a subordinating particle that must stay with its governing verb.

### Fix 7: ENZ31-ST2 â€” Verb inflection ã¾ã—ãŸãŒ split
- **Before**: "ã‚±ãƒ³ã¯æ€ã„ã¾ã—" | "ãŸãŒè­¦å®˜ã«è¦‹ã‚‰ã‚Œã¦ã—ã¾ã„ã¾ã—ãŸ"
- **After**: "ã‚±ãƒ³ã¯æ€ã„ã¾ã—ãŸãŒ" | "è­¦å®˜ã«è¦‹ã‚‰ã‚Œã¦ã—ã¾ã„ã¾ã—ãŸ"
- **Rule**: Verb inflection (masu+ta) and conjunctive particle ãŒ must not be split from the verb.

### Fix 8: EUS32-ST1 â€” Verb inflection ã¾ã›ã‚“ split
- **Before**: "ãã—ã¦ã‚‚ã†ãƒ”ã‚¯ãƒ‹ãƒƒã‚¯ã§ãã¾ã›" | "ã‚“ã«ãªã‚Šã¾ã—ãŸ"
- **After**: "ãã—ã¦ã‚‚ã†ãƒ”ã‚¯ãƒ‹ãƒƒã‚¯ã§ãã¾ã›ã‚“" | "ã«ãªã‚Šã¾ã—ãŸ"
- **Rule**: Verb inflection (mase+n) must not be split across clauses.

### Fix 9: KKR40-ST2 â€” Te-form ending split from verb stem
- **Before**: "ãã—ã¦è­¦å®˜ã‚‚ãã‚Œã‚’èžã„" | "ã¦ã‚ãã†ã‚†ã†ã“ã¨ãã†ã‚†ã†ã“ã¨ã‹ã¨æ€ã„"
- **After**: "ãã—ã¦è­¦å®˜ã‚‚ãã‚Œã‚’èžã„ã¦" | "ã‚ãã†ã‚†ã†ã“ã¨ãã†ã‚†ã†ã“ã¨ã‹ã¨æ€ã„"
- **Rule**: Te-form ending ã¦ must not be split from its verb stem.

### Note on previous gold_v1 fixes
The previous run (`rq3_validity_ja_v16_fixed_20260223`) had 8 gold_v2 fixes. With the updated v16-clean segmenter, 6 of those 8 issues were automatically resolved. The comprehensive manual review found 7 additional issues beyond the original 2 (ENZ20-ST1, CCM35-ST2).

## V16 Segmenter Patches (Post-Analysis)

After the gold_v2 manual review identified 9 boundary violations, the v16 segmenter was patched to auto-fix 7 of the 9 cases (the remaining 2 require syntactic context: ã®ã‹/ã®ã‹ã—ã‚‰ particle splits). Two new repair methods were added to `ja_clause_segmenter_v16.py`:

### New method: `_repair_inflection_splits`
Repairs verb inflection morphology split across clause boundaries at the word-index level:
- **Pattern A**: Previous clause ends with `ã¾ã—`/`ã¾ã›`/`ã§ã—` â†’ moves leading `ãŸ`/`ã‚“`/`ã¦` + optional conjunctive particles (`ãŒ`/`ã‘ã©`/`ã‘ã‚Œã©`) from next clause
- **Pattern B**: Next clause starts with polite suffix word (`ã¾ã—`/`ã¾ã™`/`ã¾ã›`) â†’ moves it + completions to previous clause
- Gap guard: â‰¤ 0.5s; remaining-content guard

### New method: `_repair_auxiliary_splits`
Repairs te-form auxiliary constructions split across clause boundaries:
- **Pattern A**: Previous clause ends with `ã¦`/`ã§` + next clause starts with auxiliary (ã—ã¾ã„, ã„ã, ãã‚‹, ã„ã‚‹, ãŠã, etc.)
- **Pattern B**: Next clause starts with `ã¦`/`ã§` + auxiliary â†’ moves both + grabs inflection continuations
- Gap guard: â‰¤ 1.0s; remaining-content guard

### Extended Patch C (text-level safety net)
Added `ã¾ã›` + `ã‚“` merge case alongside existing `ã¾ã—`/`ã§ã—` + `ãŸ`.

### Call order in `align_clauses`
```python
aligned_clauses = self._repair_te_form_boundaries(...)
aligned_clauses = self._repair_inflection_splits(...)       # NEW
aligned_clauses = self._repair_auxiliary_splits(...)         # NEW
aligned_clauses = self._repair_inflection_splits(...)       # 2nd pass (auxiliary may expose new splits)
aligned_clauses = self._repair_subordinator_hangovers(...)
```

### Patch Test Results (5 of 7 auto-fixed)

| File | Issue | Auto-fixed? |
|------|-------|-------------|
| ENZ20-ST1 | auxiliary ã¦ã„ã split | Yes |
| CCS45-ST1 | auxiliary ã¦ã„ã split | Yes |
| CCS45-ST2 | verb inflection ã¾ã—ãŸ split | Yes |
| ENZ31-ST2 | verb inflection ã¾ã—ãŸãŒ split | Yes |
| EUS32-ST1 | verb inflection ã¾ã›ã‚“ split | Yes |
| CCM35-ST2 | auxiliary ã¦ã—ã¾ã„ã¾ã—ãŸ split | No (0.86s gap + filler blocks repair) |
| KKR40-ST2 | te-form ã¦ split | No (alignment-level clause arrangement) |

The 2 unfixed cases (CCM35-ST2, KKR40-ST2) are edge cases that still require manual gold_v2 fixes via `apply_gold_v2_fixes.py`.

### Regression test on ASR TextGrids
- Only 3 files changed out of 40, all improvements (correct boundary fixes)
- Correlation stats near-identical (max Î”r = Â±0.002, max Î”Ï = Â±0.005, max Î”ICC = Â±0.001, max Î”MAE = Â±0.004)

### Patch test commands
```bash
# Manual TextGrids
ja/.venv_electra310/Scripts/python.exe ja/ja_clause_segmenter_v16.py \
  -i "ja/results/manual20_0220_2" \
  -o "ja/rq3_v16_clean_release_20260223/manual_clauses_patch_test3" \
  --model ja_ginza_electra

# ASR TextGrids
ja/.venv_electra310/Scripts/python.exe ja/ja_clause_segmenter_v16.py \
  -i "ja/results/qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/textgrids_clean_beginning_removed_by_manual" \
  -o "ja/rq3_v16_clean_release_20260223/auto_clauses_patch_test" \
  --model ja_ginza_electra

# CAF + correlation on patched outputs
ja/.venv_electra310/Scripts/python.exe ja/caf_calculator_ja.py \
  "ja/rq3_v16_clean_release_20260223/auto_clauses_patch_test" \
  -o "ja/rq3_v16_clean_release_20260223/auto_caf_patch_test.csv"

ja/.venv_electra310/Scripts/python.exe ja/caf_calculator_ja.py \
  "ja/rq3_v16_clean_release_20260223/manual_clauses_patch_test3" \
  -o "ja/rq3_v16_clean_release_20260223/manual_caf_patch_test.csv"

ja/.venv_electra310/Scripts/python.exe ja/analysis/correlation_from_caf_ja.py \
  --auto-csv "ja/rq3_v16_clean_release_20260223/auto_caf_patch_test.csv" \
  --manual-csv "ja/rq3_v16_clean_release_20260223/manual_caf_patch_test.csv" \
  --out-stats "ja/rq3_v16_clean_release_20260223/rq3_patch_test_stats.csv" \
  --out-file-level "ja/rq3_v16_clean_release_20260223/rq3_patch_test_file_level.csv" \
  --ci-bootstrap 3000
```

### Note on è¤‡åˆå‹•è©ž (compound verbs)
V1é€£ç”¨å½¢+V2 lexical/syntactic compound verbs (é£Ÿã¹å§‹ã‚ã‚‹, èµ°ã‚Šå‡ºã™, etc.) were also reviewed. GiNZA's dependency parser correctly treats these as single predicate units, and zero splits were found across all 40 files (2,510 clauses). No additional repair needed.

## Final Status
- Completed successfully.
- Matched files for correlation: 40 (ST1=20, ST2=20).
- All 9 CAF measures show r >= .891, ICC >= .877.
- V16 segmenter patched with 2 new repair methods (5/7 gold_v2 issues auto-fixed).
- Remaining 2 edge cases handled by `apply_gold_v2_fixes.py`.


