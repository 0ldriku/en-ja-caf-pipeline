# Environment and Dependencies (JA)

This file records the runtime environments used for the Japanese clean release.

## 1) ASR + MFA stage (upstream source TextGrids)

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
- `librosa==0.11.0`
- `fugashi==1.5.2`
- `numpy==2.3.5`
- `pandas==3.0.0`
- `scipy==1.17.0`

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

- Acoustic model: `japanese_mfa`
- Dictionary: `japanese_mfa`

Note:

- `mfa.exe` can fail when launched without library-path setup on this machine.
- The ASR script sets MFA runtime paths (Library/bin) before invoking MFA.

## 2) Clause segmentation + CAF + correlation stage

- Env type: venv
- Interpreter: `ja/.venv_electra310/Scripts/python.exe`
- Python: `3.10.11`

Core packages (observed on this machine):

- `torch==2.10.0`
- `transformers==4.25.1`
- `spacy==3.8.11`
- `ginza==5.2.0`
- `ja-ginza-electra==5.2.0`
- `SudachiPy==0.6.10`
- `SudachiDict-core==20260116`
- `praatio==6.2.2`
- `pandas==2.3.3`
- `numpy==2.2.6`
- `scipy==1.15.3`

Models/resources:

- spaCy/GiNZA model: `ja_ginza_electra`
- Disfluency detector model:
  - `shared/disfluency_detector/model_v2/final` (preferred canonical path)
  - `en/disfluency_test/l2_disfluency_detector/model_v2/final` (legacy fallback)

## 3) Windows console note

To avoid Unicode print errors in scripts, use:

```powershell
$env:PYTHONIOENCODING='utf-8'
```
