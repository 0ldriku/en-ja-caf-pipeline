#!/usr/bin/env python3
"""
RQ2: Pause Location Agreement — Auto MCP/ECP vs Gold MCP/ECP

Auto pipeline: qwen3_filler_mfa_beam100 (Qwen3 ASR + MFA beam=100 + segmenter v3)
Gold standard: 10 adjudicated blind files + 30 LLM production files (expert-reviewed)

Method:
  For each pause (≥250ms silent interval) in the auto TextGrid words tier:
    (a) Auto label: classify as MCP or ECP using auto clause intervals (clauses tier)
    (b) Gold label: classify as MCP or ECP using gold clause boundaries mapped to
        auto word timing via edit-distance alignment

  Gold clause text segments are defined on the canonical (manual) transcript.
  Auto word timing comes from the ASR words tier. We use edit-distance alignment
  (SCTK-style, following Matsuura et al. 2025) to map gold clause boundaries
  from canonical word space to ASR word timing.

  Compare auto vs gold labels. Metrics: Cohen's κ, accuracy, P/R/F1 per class.

Run with:  ../.venv/bin/python rq2/run_rq2_gold.py   (from en/analysis/)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score
from praatio import textgrid

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent          # en/
AUTO_CLAUSES_DIR = BASE / 'results' / 'qwen3_filler_mfa_beam100' / 'clauses'
ANNOTATION_DIR = BASE / 'annotation'
TRANSCRIPTS_DIR = ANNOTATION_DIR / 'transcripts'
GOLD_BLIND_DIR = ANNOTATION_DIR / 'boundary_agreement_260213' / 'final_correct_segments'
GOLD_PRODUCTION_DIR = ANNOTATION_DIR / 'llm_output' / 'production_30'
SELECTED_FILE = ANNOTATION_DIR / 'selected_files.json'

# ── Parameters ───────────────────────────────────────────────────────────────
MIN_PAUSE = 0.25          # seconds — same as caf_calculator.py
END_CLAUSE_TOL = 0.15     # seconds — same as caf_calculator.py


# ═════════════════════════════════════════════════════════════════════════════
#  TEXT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def normalize(text):
    return re.sub(r'\s+', ' ', text.strip().lower())


def load_gold_segments(file_id):
    """Load gold clause segments (one clause per line) from the appropriate dir."""
    for d in [GOLD_BLIND_DIR, GOLD_PRODUCTION_DIR]:
        p = d / f'{file_id}.txt'
        if p.exists():
            segs = [normalize(ln) for ln in p.read_text(encoding='utf-8-sig').splitlines()]
            return [s for s in segs if s]
    return None


# ═════════════════════════════════════════════════════════════════════════════
#  EDIT-DISTANCE ALIGNMENT (SCTK-style)
# ═════════════════════════════════════════════════════════════════════════════

def edit_distance_align(ref, hyp):
    """
    Align two word sequences using minimum edit distance (Levenshtein).
    Returns list of (ref_idx or None, hyp_idx or None, op) tuples.
    op: 'C' (correct), 'S' (substitution), 'I' (insertion), 'D' (deletion)
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],
                    dp[i - 1][j],
                    dp[i][j - 1],
                )

    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            alignment.append((i - 1, j - 1, 'C'))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append((i - 1, j - 1, 'S'))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((i - 1, None, 'D'))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            alignment.append((None, j - 1, 'I'))
            j -= 1
        else:
            break

    alignment.reverse()
    return alignment


# ═════════════════════════════════════════════════════════════════════════════
#  TEXTGRID EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def tg_words_with_timing(tg_path):
    """Return list of dicts for non-empty word intervals."""
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    tier = tg.getTier('words')
    return [{'word': e.label.strip().lower(), 'start': e.start, 'end': e.end}
            for e in tier.entries if e.label.strip()]


def tg_clause_intervals(tg_path):
    """Return list of dicts for non-empty clause intervals."""
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    tier = tg.getTier('clauses')
    return [{'start': e.start, 'end': e.end, 'text': e.label.strip()}
            for e in tier.entries if e.label.strip()]


def tg_pauses(tg_path, min_pause=MIN_PAUSE):
    """Return pauses (empty intervals >= min_pause) from words tier."""
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    tier = tg.getTier('words')
    return [{'start': e.start, 'end': e.end, 'dur': e.end - e.start}
            for e in tier.entries if e.label.strip() == '' and (e.end - e.start) >= min_pause]


# ═════════════════════════════════════════════════════════════════════════════
#  GOLD CLAUSE → TIME MAPPING (via edit-distance alignment)
# ═════════════════════════════════════════════════════════════════════════════

def gold_clauses_to_time(gold_segments, auto_words):
    """
    Map gold text clause boundaries to time coordinates using auto word timing.
    Uses edit-distance alignment to handle ASR vs manual word differences.
    Returns list of clause dicts {'start', 'end'} or empty list on failure.
    """
    # Build flat gold word list and track segment ranges
    gold_tokens = []
    seg_ranges = []   # (start_idx, end_idx_inclusive) in gold_tokens
    for seg in gold_segments:
        ws = seg.split()
        start = len(gold_tokens)
        gold_tokens.extend(ws)
        seg_ranges.append((start, len(gold_tokens) - 1))

    auto_tokens = [w['word'] for w in auto_words]

    # Align gold (canonical/manual) ↔ auto (ASR) words via edit distance
    alignment = edit_distance_align(gold_tokens, auto_tokens)

    # Build gold→auto index mapping (C and S positions)
    g2a = {}
    for g_idx, a_idx, op in alignment:
        if op in ('C', 'S') and g_idx is not None and a_idx is not None:
            g2a[g_idx] = a_idx

    clauses = []
    for seg, (gs, ge) in zip(gold_segments, seg_ranges):
        # Find first aligned auto word for segment start
        first_ai = None
        for g in range(gs, ge + 1):
            if g in g2a:
                first_ai = g2a[g]
                break
        # Find last aligned auto word for segment end
        last_ai = None
        for g in range(ge, gs - 1, -1):
            if g in g2a:
                last_ai = g2a[g]
                break
        if first_ai is not None and last_ai is not None:
            clauses.append({
                'start': auto_words[first_ai]['start'],
                'end':   auto_words[last_ai]['end'],
            })

    return clauses


# ═════════════════════════════════════════════════════════════════════════════
#  PAUSE CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════

def classify_pause(pause, clauses, tol=END_CLAUSE_TOL):
    """Classify a pause as MCP or ECP given clause time intervals."""
    mid = (pause['start'] + pause['end']) / 2
    for c in clauses:
        if abs(pause['start'] - c['end']) < tol:
            return 'ECP'
        if c['start'] < mid < c['end']:
            return 'MCP'
    return 'ECP'


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    with open(SELECTED_FILE) as f:
        all_files = json.load(f)['all_selected']

    print('=' * 70)
    print('RQ2: PAUSE LOCATION AGREEMENT (MCP vs ECP)')
    print('    Auto clause boundaries vs Gold clause boundaries (time-mapped)')
    print('    Alignment: edit-distance (Matsuura et al. 2025 / SCTK-style)')
    print('=' * 70)
    print(f'\nAuto clauses dir : {AUTO_CLAUSES_DIR}')
    print(f'Gold blind (10)  : {GOLD_BLIND_DIR}')
    print(f'Gold prod  (30)  : {GOLD_PRODUCTION_DIR}')
    print(f'Selected files   : {len(all_files)}')
    print(f'Min pause        : {MIN_PAUSE}s')
    print(f'End-clause tol   : {END_CLAUSE_TOL}s')

    all_auto_labels = []
    all_gold_labels = []
    all_tasks = []
    file_rows = []
    skipped = []

    for fid in sorted(all_files):
        tg_path = AUTO_CLAUSES_DIR / f'{fid}.TextGrid'
        task = 'ST1' if 'ST1' in fid else 'ST2'

        if not tg_path.exists():
            skipped.append((fid, 'no auto TextGrid'))
            continue

        gold_segs = load_gold_segments(fid)
        if gold_segs is None:
            skipped.append((fid, 'no gold file'))
            continue

        auto_words = tg_words_with_timing(tg_path)
        pauses = tg_pauses(tg_path)
        auto_clause_ivs = tg_clause_intervals(tg_path)
        gold_clause_ivs = gold_clauses_to_time(gold_segs, auto_words)

        if not gold_clause_ivs or not pauses:
            file_rows.append(dict(
                file=fid, task=task, n_pauses=len(pauses),
                n_agree=0, accuracy=0.0,
                gold_clauses_mapped=len(gold_clause_ivs),
            ))
            continue

        file_auto = [classify_pause(p, auto_clause_ivs) for p in pauses]
        file_gold = [classify_pause(p, gold_clause_ivs) for p in pauses]

        all_auto_labels.extend(file_auto)
        all_gold_labels.extend(file_gold)
        all_tasks.extend([task] * len(pauses))

        agree = sum(a == g for a, g in zip(file_auto, file_gold))
        file_rows.append(dict(
            file=fid, task=task, n_pauses=len(pauses),
            n_agree=agree,
            accuracy=round(agree / len(pauses), 4) if pauses else 0.0,
            gold_clauses_mapped=len(gold_clause_ivs),
        ))

    # ── Report ───────────────────────────────────────────────────────────
    auto_arr = np.array(all_auto_labels)
    gold_arr = np.array(all_gold_labels)
    task_arr = np.array(all_tasks)

    def rq2_metrics(auto, gold, name):
        n = len(auto)
        if n < 2:
            return dict(name=name, n=n, kappa=0, accuracy=0,
                        mcp_p=0, mcp_r=0, mcp_f1=0, ecp_p=0, ecp_r=0, ecp_f1=0)
        kappa = cohen_kappa_score(gold, auto)
        acc = np.mean(auto == gold)
        return dict(
            name=name, n=n, kappa=float(kappa), accuracy=float(acc),
            mcp_p=float(precision_score(gold, auto, pos_label='MCP', zero_division=0)),
            mcp_r=float(recall_score(gold, auto, pos_label='MCP', zero_division=0)),
            mcp_f1=float(f1_score(gold, auto, pos_label='MCP', zero_division=0)),
            ecp_p=float(precision_score(gold, auto, pos_label='ECP', zero_division=0)),
            ecp_r=float(recall_score(gold, auto, pos_label='ECP', zero_division=0)),
            ecp_f1=float(f1_score(gold, auto, pos_label='ECP', zero_division=0)),
        )

    st1_mask = task_arr == 'ST1'
    st2_mask = task_arr == 'ST2'

    ov = rq2_metrics(auto_arr, gold_arr, 'Overall')
    s1 = rq2_metrics(auto_arr[st1_mask], gold_arr[st1_mask], 'ST1')
    s2 = rq2_metrics(auto_arr[st2_mask], gold_arr[st2_mask], 'ST2')

    df = pd.DataFrame(file_rows)
    st1_df = df[df['task'] == 'ST1']
    st2_df = df[df['task'] == 'ST2']

    print(f'\nFiles: {len(df)}  (ST1: {len(st1_df)}, ST2: {len(st2_df)})')
    if skipped:
        print(f'Skipped ({len(skipped)}):')
        for f, r in skipped:
            print(f'  {f}: {r}')
    print(f'Pauses: {ov["n"]}  (ST1: {s1["n"]}, ST2: {s2["n"]})')

    print(f"\n{'Metric':<16} {'Overall':>10} {'ST1':>10} {'ST2':>10}")
    print('-' * 50)
    for key in ['kappa', 'accuracy', 'mcp_p', 'mcp_r', 'mcp_f1',
                'ecp_p', 'ecp_r', 'ecp_f1']:
        vals = [ov[key], s1[key], s2[key]]
        print(f'{key:<16} {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f}')

    # ── Save CSV ─────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / 'rq2_pause_location_gold.csv'
    df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')

    print('\n' + '=' * 70)
    print(f'SUMMARY — RQ2 pause location:   κ = {ov["kappa"]:.3f},  '
          f'accuracy = {ov["accuracy"]:.3f}  ({ov["n"]} pauses)')
    print(f'  MCP F1 = {ov["mcp_f1"]:.3f},  ECP F1 = {ov["ecp_f1"]:.3f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
