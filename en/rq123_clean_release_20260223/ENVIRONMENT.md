# Environment and Dependencies (EN)

This file records the exact runtime environments used for the English clean release.

## 1) ASR + MFA stage

### 1.1 ASR environment

- Env name: `qwen3-asr` (conda)
- Interpreter: `C:\Users\riku\miniconda3\envs\qwen3-asr\python.exe`
- Python: `3.12.12`

Core packages (observed on this machine):

- `qwen-asr==0.0.6`
- `torch==2.5.1+cu121`
- `transformers==4.57.6`
- `spacy==3.8.11`
- `praatio==6.2.2`
- `soundfile==0.13.1`
- `numpy==2.3.5`

Models/resources used by scripts:

- ASR model: `Qwen/Qwen3-ASR-1.7B`
- Forced aligner model: `Qwen/Qwen3-ForcedAligner-0.6B`

### 1.2 MFA environment

- Env name: `mfa` (conda)
- Interpreter: `C:\Users\riku\miniconda3\envs\mfa\python.exe`
- Python: `3.10.19`

Core packages:

- `Montreal_Forced_Aligner==3.3.9`
- `soundfile==0.13.1`
- `numpy==2.2.6`
- `scipy==1.15.2`

MFA resources used in run logs:

- Acoustic model: `english_us_arpa`
- Dictionary: `english_us_arpa`

Note:

- `mfa.exe` can fail when launched without library-path setup on this machine.
- The ASR script sets MFA runtime paths (Library/bin) before invoking MFA.

## 2) Clause segmentation + CAF + RQ analysis stage

- Env name: `qwen3-asr` (same as above)
- Interpreter: `C:\Users\riku\miniconda3\envs\qwen3-asr\python.exe`

Additional core packages used by `analysis/rq1|rq2|rq3`:

- `pandas==3.0.0`
- `scipy==1.17.0`
- `scikit-learn==1.8.0`
- `praatio==6.2.2`

Models/resources:

- spaCy model for clause segmentation:
  - `en_ud_L1L2e_combined_trf` (resolved by `scripts/textgrid_caf_segmenter_v3.py` via configured model path candidates)
- Disfluency detector model:
  - `shared/disfluency_detector/model_v2/final` (preferred canonical path)
  - `en/disfluency_test/l2_disfluency_detector/model_v2/final` (legacy fallback)

## 3) Windows console note

To avoid Unicode print errors in analysis scripts, use:

```powershell
$env:PYTHONIOENCODING='utf-8'
```
