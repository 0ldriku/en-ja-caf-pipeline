# EN+JA Unified Release Bundle (2026-02-24)

This folder packages the current EN and JA production pipelines plus shared models in one portable unit.

## Included

- `en/rq123_clean_release_20260223/`
- `ja/rq3_v16_clean_release_20260223/`
- `shared/disfluency_detector/model_v2/final/`
- `shared/filler_classifier/model_podcastfillers_neural_v1_full/`
- Shared helper scripts/docs:
  - `shared/filler_classifier/README.md`
  - `shared/filler_classifier/candidate_review_textgrids.py`
  - `shared/filler_classifier/filler_model_inference.py`
  - `shared/filler_classifier/train_podcastfillers_neural_classifier.py`
  - `shared/disfluency_detector/README.md`

## End-to-end pipeline (what script, what output)

### EN pipeline

1. ASR -> word TextGrid/JSON  
   Script: `en/rq123_clean_release_20260223/asr_qwen3_mfa_en.py`  
   Output: `en/rq123_clean_release_20260223/results/qwen3_filler_mfa_beam100/textgrids_clean/` and `.../json/`

2. Manual-span blanking step  
   **Not used in EN.**

3. Clause segmentation  
   Script: `en/rq123_clean_release_20260223/scripts/textgrid_caf_segmenter_v3.py`  
   Output: clause TextGrids in `.../results/*/clauses/` (auto and manual sides)

4. Gap-only neural filler candidate scoring  
   Script: `en/postprocess_vad_filler_classifier_en.py` with `--gap-only --threshold 0.50`  
   Output: `en/rq123_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/candidates/per_file/*_vad_classifier.csv`

5. Auto CAF (with candidate fillers applied)  
   Script: `en/rq123_clean_release_20260223/scripts/caf_calculator_vad_classifier.py`  
   Output: `.../analysis/rq3_gaponly_neural_t050_freshrun_20260224/auto_caf_gold40_gaponly_neural_t050.csv`

6. Manual CAF  
   Script: `en/rq123_clean_release_20260223/scripts/caf_calculator.py`  
   Output: manual CAF CSV for correlation

7. Correlation summary (VAD+classifier vs manual)
   Script: `en/rq123_clean_release_20260223/scripts/run_rq3_vad_classifier_correlation_en.py`
   Output (source): `.../analysis/rq3_gaponly_neural_t050_freshrun_20260224/correlation/rq3_vad_classifier_correlation_summary.csv`
   Final taskwise export: `analysis_final_taskwise_correlations_20260224/en/en_correlation_quality39_{overall,st1,st2}.csv`

### JA pipeline

1. ASR -> word TextGrid/JSON  
   Script: `ja/rq3_v16_clean_release_20260223/scripts/asr/asr_qwen3_mfa_ja_v2_spanfix_b.py`  
   Output: ASR word TextGrids/JSON source set

2. Manual-span blanking step (JA-only for this dataset)  
   Script: `ja/rq3_v16_clean_release_20260223/scripts/asr/make_beginning_removed_by_manual.py`  
   Purpose: blank ASR labels before manual first word (and optional end) to match manual annotation coverage in this specific dataset.  
   Output: ASR TextGrids with leading/outside-manual labels blanked.

3. Clause segmentation  
   Script: `ja/rq3_v16_clean_release_20260223/scripts/ja_clause_segmenter_v16.py --model ja_ginza_electra`  
   Output: `ja/rq3_v16_clean_release_20260223/auto_clauses/` and `.../manual_clauses/`

4. Manual gold reference  
   `ja/rq3_v16_clean_release_20260223/manual_clauses_gold_v2/` is the final gold-standard manual clause set used for JA validity analysis.

5. Gap-only neural filler candidate scoring  
   Script: `en/postprocess_vad_filler_classifier_en.py` with `--gap-only --threshold 0.50`  
   Output: `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/candidates/per_file/*_vad_classifier.csv`

6. Auto CAF (with candidate fillers applied)  
   Script: `ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja_gap_classifier.py`  
   Output: `.../analysis/rq3_gaponly_neural_t050_freshrun_20260224/auto_caf_gaponly_neural_t050.csv`

7. Manual CAF + correlation (VAD+classifier vs manual)
   Scripts:
   - `ja/rq3_v16_clean_release_20260223/scripts/caf_calculator_ja.py`
   - `ja/rq3_v16_clean_release_20260223/scripts/run_rq3_vad_classifier_correlation_ja.py`
   Output (source): `ja/rq3_v16_clean_release_20260223/analysis/rq3_gaponly_neural_t050_freshrun_20260224/correlation/correlation_summary.csv`
   Final taskwise export: `analysis_final_taskwise_correlations_20260224/ja/ja_correlation_final_vad_classifier_{overall,st1,st2}.csv`

## Important note about Step 2 (manual-span blanking)

- EN: not used.
- JA: used only because this JA manual dataset has coverage-span mismatch between manual annotations and ASR TextGrid timeline.
- General/real deployment: this is normally **not required**. Use it only when manual and ASR timeline coverage differ.

## Environment setup

Two virtual envs are required. Create them inside this folder before running the validate scripts:

```powershell
# 1) venv — used by all EN steps + JA filler scoring (step 5)
python -m venv envs/venv
envs/venv/Scripts/pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
envs/venv/Scripts/pip install -r requirements_venv.txt

# 2) venv_electra310 — used by JA clause segmentation, CAF, and correlation (steps 3, 6, 7)
python -m venv envs/venv_electra310
envs/venv_electra310/Scripts/pip install torch==2.10.0 --extra-index-url https://download.pytorch.org/whl/cpu
envs/venv_electra310/Scripts/pip install -r requirements_venv_electra310.txt
```

The `envs/` folder is gitignored and must be created locally on each machine.

## Run logs and reproducibility

- EN run log: `en/rq123_clean_release_20260223/RUN_LOG.md`
- JA run log: `ja/rq3_v16_clean_release_20260223/RUN_LOG.md`
- EN report: `en/rq123_clean_release_20260223/RQ1_RQ2_REPORT.md`
- JA report: `ja/rq3_v16_clean_release_20260223/RQ3_VALIDITY_REPORT_JA.md`
- Final taskwise correlation bundle: `analysis_final_taskwise_correlations_20260224/TASKWISE_CORRELATION_REPORT.md`

## Git usage

From this folder:

```powershell
git init
git add .
git commit -m "EN+JA release bundle with shared models"
git remote add origin <YOUR_REMOTE_URL>
git push -u origin main
```

If your remote enforces large-file storage, install/configure Git LFS before pushing.
