"""
JA pipeline: steps 1-2 (ASR + MFA, then manual-span blanking) for the 40 gold files.
Run from: release_en_ja_shared_20260224/

Usage:
    python run_asr_ja.py

Step 1: ASR + filler-augmented MFA
  Output: validation_run_asr/ja/
    textgrids/, textgrids_clean/, json/

Step 2: Manual-span blanking (JA dataset-specific)
  Output: validation_run_asr/ja/textgrids_clean_beginning_removed_by_manual/

After this, run validate_pipeline_ja.py for steps 3-7.
"""

import subprocess
import sys
import json
from pathlib import Path

# ── configure ──────────────────────────────────────────────────────────────────
RELEASE    = Path(__file__).parent.resolve()
PYTHON_ASR = str(RELEASE / "envs/venv/Scripts/python.exe")
JA         = "ja/rq3_v16_clean_release_20260223"
OUT_ASR    = "validation_run_asr/ja"
# ───────────────────────────────────────────────────────────────────────────────


def run(cmd):
    print("\n>>>", " ".join(str(x) for x in cmd))
    r = subprocess.run([str(x) for x in cmd], cwd=str(RELEASE))
    if r.returncode != 0:
        print(f"FAILED (exit {r.returncode})")
        sys.exit(r.returncode)


# Load gold file list (40 files)
with open(f"{JA}/analysis/rq3_gaponly_neural_t050_freshrun_20260224/file_list_40.json", "r") as f:
    file_list = json.load(f)["all_selected"]

print(f"Gold files: {len(file_list)}")

# Step 1 — ASR + filler-augmented MFA (40 gold files only)
run([PYTHON_ASR, f"{JA}/scripts/asr/asr_qwen3_mfa_ja_v2_spanfix_b.py",
     "-i", "ja/data/manual20_0220_2",
     "-o", OUT_ASR,
     "--file"] + file_list)

# Step 2 — Manual-span blanking (JA dataset-specific)
run([PYTHON_ASR, f"{JA}/scripts/asr/make_beginning_removed_by_manual.py",
     "--asr-dir",    f"{OUT_ASR}/textgrids_clean",
     "--manual-dir", f"{JA}/manual_clauses_gold_v2",
     "--out-dir",    f"{OUT_ASR}/textgrids_clean_beginning_removed_by_manual"])

print(f"\nDone. Outputs in: {RELEASE / OUT_ASR}/")
print("Next: run validate_pipeline_ja.py for steps 3-7.")
