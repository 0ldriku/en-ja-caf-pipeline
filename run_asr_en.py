"""
EN pipeline: step 1 (ASR + MFA) for the 40 gold files only.
Run from: release_en_ja_shared_20260224/

Usage:
    python run_asr_en.py

Output: validation_run_asr/en/
  textgrids/, textgrids_clean/, json/

After this, run validate_pipeline_en.py for steps 3-7.
"""

import subprocess
import sys
import json
from pathlib import Path

# ── configure ──────────────────────────────────────────────────────────────────
RELEASE = Path(__file__).parent.resolve()
PYTHON  = str(RELEASE / "envs/venv/Scripts/python.exe")
EN      = "en/rq123_clean_release_20260223"
OUT_ASR = "validation_run_asr/en"
# ───────────────────────────────────────────────────────────────────────────────


def run(cmd):
    print("\n>>>", " ".join(str(x) for x in cmd))
    r = subprocess.run([str(x) for x in cmd], cwd=str(RELEASE))
    if r.returncode != 0:
        print(f"FAILED (exit {r.returncode})")
        sys.exit(r.returncode)


# Load gold file list (40 files)
with open(f"{EN}/annotation/selected_files.json", "r") as f:
    file_list = json.load(f)["all_selected"]

print(f"Gold files: {len(file_list)}")

# Step 1 — ASR + filler-augmented MFA (40 gold files only)
run([PYTHON, f"{EN}/scripts/asr/asr_qwen3_mfa_en.py",
     "-i", "en/data/allsstar_full_manual/wav",
     "-o", OUT_ASR,
     "--file"] + file_list)

print(f"\nDone. Outputs in: {RELEASE / OUT_ASR}/")
print("Next: run validate_pipeline_en.py for steps 3-7.")
