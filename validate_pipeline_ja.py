"""
JA pipeline validation: steps 3-7
Run from: release_en_ja_shared_20260224/

Usage:
    python validate_pipeline_ja.py
"""

import subprocess
import sys
import json
import shutil
from pathlib import Path

# ── configure ──────────────────────────────────────────────────────────────────
RELEASE    = Path(__file__).parent.resolve()
PYTHON_ASR = str(RELEASE / "envs/venv/Scripts/python.exe")             # filler scoring
PYTHON_JA  = str(RELEASE / "envs/venv_electra310/Scripts/python.exe") # clause/CAF
JA         = "ja/rq3_v16_clean_release_20260223"
OUT        = "validation_run/ja"
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

# Create temp dir with only gold files for step 3
TG_CLEAN = RELEASE / OUT_ASR / "textgrids_clean_beginning_removed_by_manual"
TG_GOLD40 = RELEASE / OUT / "textgrids_gold40"
TG_GOLD40.mkdir(parents=True, exist_ok=True)
for fname in file_list:
    src = TG_CLEAN / f"{fname}.TextGrid"
    if src.exists():
        shutil.copy(src, TG_GOLD40 / f"{fname}.TextGrid")
print(f"Copied {len(file_list)} gold files to {TG_GOLD40}")

# Step 3 — Clause segmentation (auto side only) - only 40 gold files
run([PYTHON_JA, f"{JA}/scripts/ja_clause_segmenter_v16.py",
     "-i", str(TG_GOLD40),
     "-o", f"{OUT}/auto_clauses",
     "--model", "ja_ginza_electra"])

# Step 4 — Manual gold reference (pre-existing, not re-segmented)
# Gold clauses: ja/rq3_v16_clean_release_20260223/manual_clauses_gold_v2/

# Step 5 — Gap-only neural filler candidate scoring
run([PYTHON_ASR, "shared/postprocess_vad_filler_classifier_en.py",
     "--file-list-json", f"{JA}/analysis/rq3_gaponly_neural_t050_freshrun_20260224/file_list_40.json",
     "--file-list-key", "all_selected",
     "--audio-dir", "ja/data/manual20_0220_2",
     "--asr-json-dir", f"{OUT_ASR}/json",
     "--model-path", "shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt",
     "--threshold", "0.50",
     "--gap-only",
     "--out-dir", f"{OUT}/candidates"])

# Step 6 — Auto CAF with candidate fillers applied
run([PYTHON_JA, f"{JA}/scripts/caf_calculator_ja_gap_classifier.py",
     f"{OUT}/auto_clauses",
     "--candidate-dir", f"{OUT}/candidates/per_file",
     "--file-list-json", f"{JA}/analysis/rq3_gaponly_neural_t050_freshrun_20260224/file_list_40.json",
     "--file-list-key", "all_selected",
     "--output", f"{OUT}/auto_caf.csv"])

# Step 7a — Manual CAF (uses gold clause TextGrids — not re-segmented)
run([PYTHON_JA, f"{JA}/scripts/caf_calculator_ja.py",
     f"{JA}/results/manual_clauses_gold_v2",
     "--output", f"{OUT}/manual_caf.csv"])

# Step 7b — Correlation summary
run([PYTHON_JA, f"{JA}/scripts/run_rq3_vad_classifier_correlation_ja.py",
     "--clf-auto-csv", f"{OUT}/auto_caf.csv",
     "--manual-csv",   f"{OUT}/manual_caf.csv",
     "--out-summary",  f"{OUT}/correlation/correlation_summary.csv"])

print(f"\nDone. Outputs in: {RELEASE / OUT}")
