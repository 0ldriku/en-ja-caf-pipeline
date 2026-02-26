"""
EN pipeline validation: steps 3-7
Run from: release_en_ja_shared_20260224/

Usage:
    python validate_pipeline_en.py
"""

import subprocess
import sys
import json
import shutil
from pathlib import Path

# ── configure ──────────────────────────────────────────────────────────────────
RELEASE = Path(__file__).parent.resolve()
PYTHON  = str(RELEASE / "envs/venv/Scripts/python.exe")
EN      = "en/rq123_clean_release_20260223"
OUT     = "validation_run/en"
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

# Create temp dir with only gold files for step 3
TG_CLEAN = RELEASE / OUT_ASR / "textgrids_clean"
TG_GOLD40 = RELEASE / OUT / "textgrids_gold40"
TG_GOLD40.mkdir(parents=True, exist_ok=True)
for fname in file_list:
    src = TG_CLEAN / f"{fname}.TextGrid"
    if src.exists():
        shutil.copy(src, TG_GOLD40 / f"{fname}.TextGrid")
print(f"Copied {len(file_list)} gold files to {TG_GOLD40}")

# Step 3 — Clause segmentation (auto side) - only 40 gold files
run([PYTHON, f"{EN}/scripts/textgrid_caf_segmenter_v3.py",
     "-i", str(TG_GOLD40),
     "-o", f"{OUT}/auto_clauses"])

# Step 4 — Gap-only neural filler candidate scoring
run([PYTHON, "shared/postprocess_vad_filler_classifier_en.py",
     "--file-list-json", f"{EN}/annotation/selected_files.json",
     "--file-list-key", "all_selected",
     "--audio-dir", "en/data/allsstar_full_manual/wav",
     "--asr-json-dir", f"{OUT_ASR}/json",
     "--model-path", "shared/filler_classifier/model_podcastfillers_neural_v1_full/model.pt",
     "--threshold", "0.50",
     "--gap-only",
     "--out-dir", f"{OUT}/candidates"])

# Step 5 — Auto CAF with candidate fillers applied
run([PYTHON, f"{EN}/scripts/caf_calculator_vad_classifier.py",
     f"{OUT}/auto_clauses",
     "--candidate-dir", f"{OUT}/candidates/per_file",
     "--file-list-json", f"{EN}/annotation/selected_files.json",
     "--file-list-key", "all_selected",
     "--output", f"{OUT}/auto_caf.csv"])

# Step 6 — Manual CAF (gold clause TextGrids — not re-segmented)
run([PYTHON, f"{EN}/scripts/caf_calculator.py",
     f"{EN}/results/manual_clauses_gold",
     "--output", f"{OUT}/manual_caf.csv"])

# Step 7 — Correlation summary
# --exclude: quality-filter (manual preamble mismatch)
run([PYTHON, f"{EN}/scripts/run_rq3_vad_classifier_correlation_en.py",
     "--auto-clf-csv", f"{OUT}/auto_caf.csv",
     "--manual-csv",   f"{OUT}/manual_caf.csv",
     "--exclude",      "ALL_139_M_PBR_ENG_ST1",
     "--out-dir",      f"{OUT}/correlation"])

print(f"\nDone. Outputs in: {RELEASE / OUT}")
