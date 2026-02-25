#!/usr/bin/env python
"""
Qwen3-ASR + Filler-Augmented MFA Pipeline (English)

Pipeline:
  1. Qwen3 ASR + ForcedAligner -> full transcript + rough word timestamps
     (with disfluency prompt to capture fillers)
  2. Filler-augmented MFA with high beam (--beam 100 --retry_beam 400):
     inject filler placeholders into gaps, run MFA on full file.
     High beam prevents alignment drift on long files (>30s).
     Filler injection prevents stretching over pause regions.
  3. Build final TextGrid + JSON
  4. (Optional) Compare against manual and CrisperWhisper TextGrids

Usage:
    conda activate qwen3-asr
    python asr_qwen3_mfa_en.py -i data/allsstar_full_manual/wav -o results/qwen3_mfa_en
    python asr_qwen3_mfa_en.py -i data/allsstar_full_manual/wav -o results/qwen3_mfa_en --file ALL_003_F_RUS_ENG_ST1

Requires:
  - qwen3-asr conda env (for ASR + rough alignment)
  - mfa conda env (for precise alignment, called as subprocess)
"""

import os
import sys
import io
import re
import json
import math
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import soundfile as sf
from praatio import textgrid
from praatio.utilities.constants import Interval

from qwen_asr import Qwen3ASRModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Config ──────────────────────────────────────────────────────────────────
ASR_MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"
FORCED_ALIGNER_PATH = "Qwen/Qwen3-ForcedAligner-0.6B"
SAMPLE_RATE = 16000
AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
PAD_SEC = 0.15
FILLER_CONTEXT = "IMPORTANT: Transcribe ALL disfluencies exactly as spoken. Include every filled pause, hesitation, repetition, false start, and self-correction. Never clean up or omit any disfluency."

# MFA config
MFA_ENV = os.path.join(os.path.expanduser("~"), "miniconda3", "envs", "mfa")
MFA_EXE = os.path.join(MFA_ENV, "Scripts", "mfa.exe")
MFA_LIB_BIN = os.path.join(MFA_ENV, "Library", "bin")
MFA_ACOUSTIC = "english_us_arpa"
MFA_DICT = "english_us_arpa"

# Filler injection params
FILLER_TOKEN = "uh"
GAP_MIN = 0.4        # minimum gap (s) to inject a filler
GAP_OFFSET = 0.35    # offset for filler count formula
GAP_STEP = 0.55      # step for additional fillers
FILLER_MAX = 3        # max fillers per gap

# Comparison paths (relative to script dir)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MANUAL_DIR = os.path.join(SCRIPT_DIR, "results", "manual_260207", "clauses_v3")
AUTO_DIR = os.path.join(SCRIPT_DIR, "results", "auto_260207", "clauses")


# ── Step 1: Qwen3 ASR ──────────────────────────────────────────────────────

def load_models():
    """Load ASR model with integrated ForcedAligner for chunked alignment."""
    print("Loading Qwen3-ASR model (with integrated aligner)...")
    asr = Qwen3ASRModel.from_pretrained(
        ASR_MODEL_PATH,
        forced_aligner=FORCED_ALIGNER_PATH,
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=32,
        max_new_tokens=4096,
    )
    print("Models loaded!")
    return asr


def step1_transcribe(audio_path: str, asr_model,
                     language: str = "English") -> Tuple[List[Dict], float, str]:
    """Step 1: ASR + chunked forced alignment via return_time_stamps.
    Uses FILLER_CONTEXT to encourage disfluency transcription."""
    info = sf.info(audio_path)
    duration = info.duration

    results = asr_model.transcribe(
        audio=audio_path, language=language, context=FILLER_CONTEXT,
        return_time_stamps=True)
    r = results[0]
    transcript = r.text or ""

    if not transcript.strip():
        return [], duration, ""

    words = []
    if r.time_stamps:
        for ts in r.time_stamps:
            text = ts.text.strip()
            if text:
                words.append({
                    'word': text,
                    'start': ts.start_time,
                    'end': ts.end_time,
                })

    return words, duration, transcript


# ── Step 2: Filler-Augmented MFA (full-file, high beam) ─────────────────────

def _norm_token(s: str) -> str:
    """Normalize token for MFA: lowercase, strip punctuation edges."""
    s = str(s).strip().lower()
    s = re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", s)
    s = re.sub(r"'+", "'", s)
    return s


def _filler_count_for_gap(gap: float) -> int:
    """How many filler tokens to inject for a given gap duration."""
    if gap < GAP_MIN:
        return 0
    k = math.floor((gap - GAP_OFFSET) / GAP_STEP) + 1
    if k < 0:
        return 0
    return min(FILLER_MAX, k)


def _run_mfa(corpus_dir: str, output_dir: str, output_format: str = "json") -> bool:
    """Run MFA align as subprocess."""
    env = os.environ.copy()
    env["PATH"] = MFA_LIB_BIN + os.pathsep + env.get("PATH", "")
    cmd = [
        MFA_EXE, "align",
        corpus_dir, MFA_DICT, MFA_ACOUSTIC, output_dir,
        "--clean", "--single_speaker",
        "--overwrite", "--output_format", output_format,
        "--beam", "100", "--retry_beam", "400",
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        ext = ".json" if output_format == "json" else ".TextGrid"
        out_files = list(Path(output_dir).glob(f"*{ext}"))
        if out_files:
            return True
        print(f"    MFA error: {result.stderr[-500:] if result.stderr else 'no stderr'}")
        return False
    return True


def _parse_mfa_json(jpath: str) -> List[Dict]:
    """Parse word entries from MFA JSON output."""
    data = json.load(open(jpath, encoding='utf-8'))
    entries = data.get('tiers', {}).get('words', {}).get('entries', [])
    out = []
    for s, e, w in entries:
        w = str(w)
        if w == '' or w == '<eps>':
            continue
        out.append({'word': w, 'start': float(s), 'end': float(e),
                     'token': _norm_token(w)})
    return out


def _build_augmented_sequence(words: List[Dict]) -> Tuple[List[Dict], int]:
    """Build token sequence with filler placeholders injected into gaps.
    Returns (sequence, n_injected).
    Each item: {'token': str, 'raw_index': int|None, 'kind': 'raw'|'filler'}"""
    seq = []
    injected = 0
    for i, w in enumerate(words):
        tok = _norm_token(w.get('word', ''))
        if tok:
            seq.append({'token': tok, 'raw_index': i, 'kind': 'raw'})
        if i + 1 < len(words):
            gap = float(words[i + 1]['start']) - float(words[i]['end'])
            k = _filler_count_for_gap(gap)
            for _ in range(k):
                seq.append({'token': FILLER_TOKEN, 'raw_index': None, 'kind': 'filler'})
                injected += 1
    return seq, injected


def _map_back_to_raw(raw_words: List[Dict], seq: List[Dict],
                     mfa_words: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Map MFA-aligned words back to original raw words.
    Filler tokens are consumed (discarded), real words get MFA timestamps.
    Any unmatched raw words fall back to rough timestamps."""
    aligned = [None] * len(raw_words)
    j = 0
    consumed = 0
    skipped = 0

    for item in seq:
        expected = item['token']
        while j < len(mfa_words) and mfa_words[j]['token'] != expected:
            skipped += 1
            j += 1
        if j >= len(mfa_words):
            continue
        hit = mfa_words[j]
        j += 1
        if item['raw_index'] is None:
            consumed += 1
            continue
        idx = item['raw_index']
        aligned[idx] = {
            'word': raw_words[idx]['word'],
            'start': round(float(hit['start']), 3),
            'end': round(float(hit['end']), 3),
            'source': 'mfa',
        }

    fallback = 0
    for i, w in enumerate(aligned):
        if w is not None:
            continue
        rs = float(raw_words[i]['start'])
        re_ = float(raw_words[i]['end'])
        if re_ <= rs:
            re_ = rs + 0.001
        aligned[i] = {
            'word': raw_words[i]['word'],
            'start': round(rs, 3),
            'end': round(re_, 3),
            'source': 'rough',
        }
        fallback += 1

    stats = {
        'mfa_word_count': len(mfa_words),
        'filler_injected': sum(1 for x in seq if x['kind'] == 'filler'),
        'filler_consumed': consumed,
        'fallback_count': fallback,
        'skipped_mfa_tokens': skipped,
    }
    return aligned, stats


def step2_filler_augmented_mfa(rough_words: List[Dict], audio_path: str,
                                duration: float) -> Tuple[List[Dict], Dict]:
    """Step 2: Filler-augmented MFA alignment (full-file, high beam).
    1. Inject filler placeholders into gaps between words
    2. Run MFA on the full file with --beam 100 to prevent drift
    3. Map MFA results back to original words, discarding fillers
    """
    if not rough_words:
        return [], {}

    # Build augmented sequence
    seq, injected = _build_augmented_sequence(rough_words)
    if not seq:
        return [], {}

    tokens = [x['token'] for x in seq]
    print(f"    {len(rough_words)} words + {injected} filler placeholders = {len(tokens)} tokens")

    # Prepare temp corpus
    tmp = tempfile.mkdtemp(prefix="mfa_filler_")
    corpus = os.path.join(tmp, "corpus")
    out = os.path.join(tmp, "output")
    os.makedirs(corpus)
    os.makedirs(out)

    shutil.copy2(audio_path, os.path.join(corpus, "utt.wav"))
    with open(os.path.join(corpus, "utt.lab"), 'w', encoding='utf-8') as f:
        f.write(' '.join(tokens) + '\n')

    # Run MFA with high beam to prevent drift on long files
    print(f"    Running MFA (beam=100)...")
    if not _run_mfa(corpus, out, output_format="json"):
        print(f"    MFA failed, falling back to rough timestamps")
        shutil.rmtree(tmp, ignore_errors=True)
        fallback = [{
            'word': w['word'],
            'start': round(float(w['start']), 3),
            'end': round(float(w['end']), 3),
            'source': 'rough',
        } for w in rough_words]
        return fallback, {'fallback_count': len(rough_words)}

    # Parse MFA output and map back
    jpath = os.path.join(out, "utt.json")
    if not os.path.exists(jpath):
        print(f"    MFA output not found, falling back to rough timestamps")
        shutil.rmtree(tmp, ignore_errors=True)
        fallback = [{
            'word': w['word'],
            'start': round(float(w['start']), 3),
            'end': round(float(w['end']), 3),
            'source': 'rough',
        } for w in rough_words]
        return fallback, {'fallback_count': len(rough_words)}

    mfa_words = _parse_mfa_json(jpath)
    aligned, stats = _map_back_to_raw(rough_words, seq, mfa_words)

    shutil.rmtree(tmp, ignore_errors=True)

    n_mfa = sum(1 for w in aligned if w['source'] == 'mfa')
    n_rough = sum(1 for w in aligned if w['source'] == 'rough')
    print(f"    {len(aligned)} words ({n_mfa} MFA, {n_rough} rough fallback)")
    print(f"    Fillers: {stats['filler_injected']} injected, {stats['filler_consumed']} consumed")

    return aligned, stats


# ── Step 3: Build outputs ─────────────────────────────────────────────────

def clean_word_text(word: str) -> str:
    """Remove punctuation from word edges."""
    cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', word)
    cleaned = re.sub(r"[^\w']", '', cleaned)
    return cleaned.lower()


def build_final_textgrid(words: List[Dict], duration: float, path: str,
                         clean: bool = False):
    """Build TextGrid with word and pause tiers."""
    entries = []
    last_end = 0.0
    MIN_DUR = 0.001

    for w in words:
        text = clean_word_text(w['word']) if clean else w['word']
        if not text:
            continue

        ws, we = w['start'], w['end']
        if we <= ws:
            we = ws + MIN_DUR
        if ws < last_end:
            ws = last_end
        if we <= ws:
            we = ws + MIN_DUR

        if ws > last_end + 0.001:
            entries.append(Interval(last_end, ws, ""))
        entries.append(Interval(ws, we, text))
        last_end = we

    if last_end < duration:
        entries.append(Interval(last_end, duration, ""))

    tg = textgrid.Textgrid()
    words_tier = textgrid.IntervalTier("words", entries, minT=0, maxT=duration)
    tg.addTier(words_tier)

    pause_entries = []
    for e in entries:
        if e.label.strip() == "" and (e.end - e.start) >= 0.25:
            pause_entries.append(Interval(e.start, e.end, "sp"))
        else:
            pause_entries.append(Interval(e.start, e.end, ""))
    pause_tier = textgrid.IntervalTier("pauses", pause_entries, minT=0, maxT=duration)
    tg.addTier(pause_tier)

    tg.save(path, format="long_textgrid", includeBlankSpaces=True)


def save_json(words: List[Dict], duration: float, transcript: str, path: str,
              stats: Dict = None):
    """Save results to JSON."""
    data = {
        'duration': duration,
        'transcript': transcript,
        'n_words': len(words),
        'words': [{k: v for k, v in w.items()} for w in words],
    }
    if stats:
        data['filler_augmentation'] = stats
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Comparison with manual & CrisperWhisper ───────────────────────────────

def _parse_short_tg_tier(lines: List[str], tier_name: str) -> List[Dict]:
    """Parse a tier from short-format TextGrid."""
    words = []
    i = 0
    while i < len(lines):
        line = lines[i].strip().strip('"')
        if line == "IntervalTier":
            next_name = lines[i + 1].strip().strip('"')
            if next_name == tier_name:
                n = int(lines[i + 4].strip())
                i += 5
                for _ in range(n):
                    s = float(lines[i].strip())
                    e = float(lines[i + 1].strip())
                    t = lines[i + 2].strip().strip('"')
                    i += 3
                    if t and t not in ('', 'sp', 'sil'):
                        words.append({'word': t.lower().strip(), 'start': s, 'end': e})
                break
            else:
                i += 1
        else:
            i += 1
    return words


def _parse_long_tg_tier(lines: List[str], tier_name: str) -> List[Dict]:
    """Parse a tier from long-format TextGrid."""
    words = []
    i = 0
    found = False
    while i < len(lines):
        line = lines[i].strip()
        if 'name = ' in line and tier_name in line:
            found = True
            i += 1
            continue
        if found and 'text = ' in line:
            text_match = re.search(r'text\s*=\s*"(.*?)"', line)
            text = text_match.group(1) if text_match else ""
            xmin_val = xmax_val = None
            for back in range(1, 5):
                prev = lines[i - back].strip()
                if 'xmin' in prev and xmin_val is None:
                    xmin_val = float(re.search(r'[\d.]+', prev).group())
                if 'xmax' in prev and xmax_val is None:
                    xmax_val = float(re.search(r'[\d.]+', prev).group())
            if text and text not in ('', 'sp', 'sil') and xmin_val is not None:
                words.append({'word': text.lower().strip(), 'start': xmin_val, 'end': xmax_val})
        if found and 'class = "IntervalTier"' in line and i > 10:
            break
        i += 1
    return words


def parse_reference_tg(filepath: str, tier_name: str) -> List[Dict]:
    """Parse words from a TextGrid file (auto-detect format)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    content = ''.join(lines[:10])
    if 'item [' in content or 'name = ' in content:
        return _parse_long_tg_tier(lines, tier_name)
    return _parse_short_tg_tier(lines, tier_name)


def _norm(w: str) -> str:
    return re.sub(r'[^\w\']', '', w.lower().strip())


def _match_words(gold: List[Dict], pred: List[Dict]):
    """Time-proximity matching: for each gold word, find nearest pred word
    with same text within 2s window."""
    matches = []
    used = set()
    for g in gold:
        gc = _norm(g['word'])
        if not gc:
            continue
        best_k = None
        best_dist = float('inf')
        for k, p in enumerate(pred):
            if k in used:
                continue
            if _norm(p['word']) == gc:
                dist = abs(g['start'] - p['start'])
                if dist < best_dist and dist < 2.0:
                    best_dist = dist
                    best_k = k
        if best_k is not None:
            matches.append((g, pred[best_k]))
            used.add(best_k)
    return matches


def compare_with_references(basename: str, our_words: List[Dict]):
    """Compare our MFA output against manual and CrisperWhisper."""
    manual_tg = os.path.join(MANUAL_DIR, f"{basename}.TextGrid")
    auto_tg = os.path.join(AUTO_DIR, f"{basename}.TextGrid")

    if not os.path.exists(manual_tg):
        return

    # Parse manual
    manual_words = parse_reference_tg(manual_tg, "utt - words")
    if not manual_words:
        manual_words = parse_reference_tg(manual_tg, "words")
    if not manual_words:
        return

    print(f"\n  --- Comparison vs Manual ({len(manual_words)} gold words) ---")

    # Our pipeline vs manual
    our_matches = _match_words(manual_words, our_words)
    if our_matches:
        diffs = [abs(g['start'] - p['start']) for g, p in our_matches] + \
                [abs(g['end'] - p['end']) for g, p in our_matches]
        print(f"  Qwen3+MFA:       {len(our_matches)} matched, "
              f"mean={np.mean(diffs)*1000:.0f}ms, "
              f"median={np.median(diffs)*1000:.0f}ms, "
              f"p90={np.percentile(diffs, 90)*1000:.0f}ms")

    # CrisperWhisper vs manual
    if os.path.exists(auto_tg):
        cw_words = parse_reference_tg(auto_tg, "words")
        cw_matches = _match_words(manual_words, cw_words)
        if cw_matches:
            diffs = [abs(g['start'] - p['start']) for g, p in cw_matches] + \
                    [abs(g['end'] - p['end']) for g, p in cw_matches]
            print(f"  CrisperWhisper:  {len(cw_matches)} matched, "
                  f"mean={np.mean(diffs)*1000:.0f}ms, "
                  f"median={np.median(diffs)*1000:.0f}ms, "
                  f"p90={np.percentile(diffs, 90)*1000:.0f}ms")


# ── Main ────────────────────────────────────────────────────────────────────

def process_file(audio_path: str, asr_model,
                 output_dir: str, language: str = "English"):
    """Full pipeline for one file."""
    basename = os.path.splitext(os.path.basename(audio_path))[0]

    # Step 1: ASR + rough alignment (with filler prompt)
    print(f"  Step 1: ASR + rough alignment...")
    rough_words, duration, transcript = step1_transcribe(
        audio_path, asr_model, language)
    print(f"    Transcript: {len(transcript)} chars, {len(rough_words)} words")

    if not rough_words:
        print(f"    No words detected, skipping.")
        return None

    # Step 2: Segmented filler-augmented MFA alignment
    print(f"  Step 2: Segmented filler-augmented MFA alignment...")
    precise_words, stats = step2_filler_augmented_mfa(
        rough_words, audio_path, duration)

    # Step 3: Build outputs
    print(f"  Step 3: Building outputs...")
    tg_dir = os.path.join(output_dir, "textgrids")
    tg_clean_dir = os.path.join(output_dir, "textgrids_clean")
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(tg_dir, exist_ok=True)
    os.makedirs(tg_clean_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    build_final_textgrid(precise_words, duration,
                         os.path.join(tg_dir, f"{basename}.TextGrid"), clean=False)
    build_final_textgrid(precise_words, duration,
                         os.path.join(tg_clean_dir, f"{basename}.TextGrid"), clean=True)
    save_json(precise_words, duration, transcript,
              os.path.join(json_dir, f"{basename}.json"), stats=stats)

    # Compare with references
    compare_with_references(basename, precise_words)

    n_mfa = sum(1 for w in precise_words if w.get('source') == 'mfa')
    n_rough = sum(1 for w in precise_words if w.get('source') == 'rough')
    return {
        'name': basename,
        'duration': duration,
        'n_rough_words': len(rough_words),
        'n_precise_words': len(precise_words),
        'n_mfa': n_mfa,
        'n_rough_fallback': n_rough,
        'filler_injected': stats.get('filler_injected', 0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR + Filler-Augmented MFA Pipeline (English)")
    parser.add_argument("-i", "--input", required=True,
                        help="Input folder with audio files")
    parser.add_argument("-o", "--output", required=True,
                        help="Output folder")
    parser.add_argument("--lang", "-l", default="English",
                        help="Language (default: English)")
    parser.add_argument("--file", nargs="+", help="Process specific files by name (without extension)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input folder not found: {args.input}")
        sys.exit(1)

    asr_model = load_models()

    # Get files
    if args.file:
        audio_files = []
        for name in args.file:
            found = False
            for ext in AUDIO_EXTS:
                candidate = f"{name}{ext}"
                if os.path.exists(os.path.join(args.input, candidate)):
                    audio_files.append(candidate)
                    found = True
                    break
            if not found:
                print(f"Warning: File not found: {name}")
        if not audio_files:
            print(f"Error: No files found")
            sys.exit(1)
    else:
        audio_files = sorted([
            f for f in os.listdir(args.input)
            if f.lower().endswith(AUDIO_EXTS)
        ])

    print(f"\nPipeline: Qwen3 ASR -> Filler-Augmented MFA -> TextGrid")
    print(f"Processing {len(audio_files)} files...")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("-" * 60)

    results = []
    failed = []
    for i, audio_file in enumerate(audio_files, 1):
        audio_path = os.path.join(args.input, audio_file)
        basename = os.path.splitext(audio_file)[0]
        print(f"\n[{i}/{len(audio_files)}] {basename}")

        try:
            result = process_file(audio_path, asr_model,
                                  args.output, args.lang)
            if result:
                results.append(result)
                print(f"  Done: {result['n_precise_words']} words "
                      f"({result['n_mfa']} MFA, {result['n_rough_fallback']} rough, "
                      f"{result['filler_injected']} fillers injected)")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append(basename)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print(f"Done! {len(results)}/{len(audio_files)} files processed")
    if results:
        total_mfa = sum(r['n_mfa'] for r in results)
        total_rough = sum(r['n_rough_fallback'] for r in results)
        total_words = sum(r['n_precise_words'] for r in results)
        total_fillers = sum(r['filler_injected'] for r in results)
        print(f"Total words: {total_words} ({total_mfa} MFA, {total_rough} rough)")
        print(f"Total fillers injected: {total_fillers}")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
