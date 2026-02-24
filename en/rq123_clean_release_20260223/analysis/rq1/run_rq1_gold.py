#!/usr/bin/env python3
"""
RQ1: Clause Boundary Agreement — Auto (spaCy on ASR) vs Gold (LLM + expert review)

Auto pipeline: qwen3_filler_mfa_beam100 (Qwen3 ASR + MFA beam=100 + segmenter v3)
Gold standard: 10 adjudicated blind files + 30 LLM production files (expert-reviewed)
Canonical transcript: annotation/transcripts/ (from manual_260206_2 clause tier)

Evaluation unit (Matsuura-style word-boundary labels):
  For each inter-word position in the aligned sequence, label whether a clause
  boundary occurs (1/0). Compare gold vs auto labels. Compute Cohen's κ, P/R/F1.

Alignment method:
  Gold segments the canonical (manual) transcript → direct boundary labels.
  Auto segments the ASR transcript → boundary labels on ASR words.
  We align canonical ↔ ASR words via **edit-distance alignment** (Levenshtein),
  following Matsuura et al. (2025) who used NIST's SCTK (Chen & Yoon, 2012).

  Edit-distance alignment explicitly handles:
    - Correct (C): same word at same position → compare labels directly
    - Substitution (S): different word at same position → compare labels directly
    - Deletion (D): manual word not in ASR → auto label = 0 (system never saw it)
    - Insertion (I): ASR word not in manual → gold label = 0 (no reference)

  This is stricter than SequenceMatcher (LCS), which skips substituted words and
  can "heal" boundary disagreements at ASR error positions.

Run with:  ../.venv/bin/python rq1/run_rq1_gold.py   (from en/analysis/)
"""

import json
import re
from pathlib import Path

import pandas as pd
from praatio import textgrid

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent          # en/
AUTO_CLAUSES_DIR = BASE / 'results' / 'qwen3_filler_mfa_beam100' / 'clauses'
ANNOTATION_DIR = BASE / 'annotation'
TRANSCRIPTS_DIR = ANNOTATION_DIR / 'transcripts'
GOLD_BLIND_DIR = ANNOTATION_DIR / 'boundary_agreement_260213' / 'final_correct_segments'
GOLD_PRODUCTION_DIR = ANNOTATION_DIR / 'llm_output' / 'production_30'
SELECTED_FILE = ANNOTATION_DIR / 'selected_files.json'


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


def load_canonical(file_id):
    return normalize((TRANSCRIPTS_DIR / f'{file_id}.txt').read_text(encoding='utf-8-sig'))




# ═════════════════════════════════════════════════════════════════════════════
#  EDIT-DISTANCE ALIGNMENT (SCTK-style, following Matsuura et al. 2025)
# ═════════════════════════════════════════════════════════════════════════════

def edit_distance_align(ref, hyp):
    """
    Align two word sequences using minimum edit distance (Levenshtein).
    Returns list of (ref_idx or None, hyp_idx or None, op) tuples.
    op: 'C' (correct), 'S' (substitution), 'I' (insertion), 'D' (deletion)

    This replicates the alignment logic of NIST's SCTK sclite tool.
    """
    n, m = len(ref), len(hyp)
    # Cost matrix: sub=1, ins=1, del=1 (standard Levenshtein)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]        # correct
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],               # substitution
                    dp[i - 1][j],                    # deletion
                    dp[i][j - 1],                    # insertion
                )

    # Backtrace (prefer C > S > D > I for tie-breaking)
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
            # Fallback (shouldn't happen with correct DP)
            break

    alignment.reverse()
    return alignment


# ═════════════════════════════════════════════════════════════════════════════
#  BOUNDARY LABELS
# ═════════════════════════════════════════════════════════════════════════════

def boundaries_from_segments(segments):
    """
    Return set of boundary positions from clause segments.
    Position k means "a clause boundary exists before word k" (i.e., word k
    starts a new clause). Equivalently, boundary after word k-1.
    The first clause start (position 0) and utterance-final are excluded.
    """
    positions = set()
    idx = 0
    for seg in segments:
        n = len(seg.split())
        if idx > 0:                     # skip position 0 (utterance start)
            positions.add(idx)
        idx += n
    return positions


def boundary_labels_on_sequence(n_words, boundary_positions):
    """
    Build a boolean array of length n_words where labels[i] = True if a clause
    boundary exists before word i (i.e., word i starts a new clause).
    """
    return [i in boundary_positions for i in range(n_words)]


# ═════════════════════════════════════════════════════════════════════════════
#  TEXTGRID EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def auto_clause_segments(tg_path):
    """Get auto clause text segments from the clauses tier."""
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    tier = tg.getTier('clauses')
    return [normalize(e.label) for e in tier.entries if e.label.strip()]


# ═════════════════════════════════════════════════════════════════════════════
#  RQ1: Per-file clause boundary comparison
# ═════════════════════════════════════════════════════════════════════════════

def rq1_file(gold_segs, auto_segs, canonical_tokens):
    """
    Compare clause boundaries between gold and auto using edit-distance
    alignment in the canonical (manual) token space.

    1. Gold boundaries on canonical words (direct — gold segments the canonical).
    2. Auto boundaries on ASR clause-text words.
    3. Align canonical ↔ ASR clause-text via edit distance.
    4. At each alignment position, extract both labels and compare:
       - C/S: both labels defined → compare
       - D (manual word, no ASR): auto label = 0
       - I (ASR word, no manual): gold label = 0
    5. Compute TP/FP/FN/TN, P/R/F1, κ.
    """
    # Gold boundary positions on canonical
    gold_tokens = []
    for seg in gold_segs:
        gold_tokens.extend(seg.split())
    gold_bounds = boundaries_from_segments(gold_segs)
    gold_labels = boundary_labels_on_sequence(len(gold_tokens), gold_bounds)

    # Auto boundary positions on ASR clause-text
    auto_clause_tokens = []
    for seg in auto_segs:
        auto_clause_tokens.extend(seg.split())
    auto_bounds = boundaries_from_segments(auto_segs)
    auto_labels = boundary_labels_on_sequence(len(auto_clause_tokens), auto_bounds)

    # Map gold labels to canonical space (handle minor diffs)
    if gold_tokens != canonical_tokens:
        align_gc = edit_distance_align(gold_tokens, canonical_tokens)
        can_gold_labels = [False] * len(canonical_tokens)
        for g_idx, c_idx, op in align_gc:
            if op in ('C', 'S') and g_idx is not None and c_idx is not None:
                can_gold_labels[c_idx] = gold_labels[g_idx]
            elif op == 'I' and c_idx is not None:
                can_gold_labels[c_idx] = False    # no gold word → no boundary
        can_gold_labels = can_gold_labels
    else:
        can_gold_labels = gold_labels

    # Align canonical ↔ ASR clause-text via edit distance
    alignment = edit_distance_align(canonical_tokens, auto_clause_tokens)

    if len(alignment) < 2:
        return None

    # Build aligned label pairs for EVERY alignment position
    # We compare "is there a boundary BEFORE this word?"
    pairs_gold = []
    pairs_auto = []
    for ref_idx, hyp_idx, op in alignment:
        if op == 'C' or op == 'S':
            pairs_gold.append(can_gold_labels[ref_idx])
            pairs_auto.append(auto_labels[hyp_idx])
        elif op == 'D':
            # Manual word exists, ASR word doesn't → auto couldn't predict
            pairs_gold.append(can_gold_labels[ref_idx])
            pairs_auto.append(False)
        elif op == 'I':
            # ASR word exists, manual doesn't → no gold reference
            pairs_gold.append(False)
            pairs_auto.append(auto_labels[hyp_idx])

    # Compute metrics (exclude position 0: utterance start is never a "boundary")
    # Actually, position 0 in the alignment corresponds to the first word — we
    # defined boundary as "before word i" so word 0 is never a boundary in either
    # system. But alignment shuffles indices. We compare all positions as-is;
    # the boundaries_from_segments already excludes position 0.
    n = len(pairs_gold)
    tp = sum(a and g for a, g in zip(pairs_auto, pairs_gold))
    fp = sum(a and not g for a, g in zip(pairs_auto, pairs_gold))
    fn = sum(not a and g for a, g in zip(pairs_auto, pairs_gold))
    tn = sum(not a and not g for a, g in zip(pairs_auto, pairs_gold))

    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    obs = (tp + tn) / n if n else 1.0
    pyp = (tp + fp) / n if n else 0.0
    pyg = (tp + fn) / n if n else 0.0
    exp = pyp * pyg + (1 - pyp) * (1 - pyg)
    kappa = ((obs - exp) / (1 - exp)) if abs(1 - exp) > 1e-12 else 1.0

    # Alignment stats
    n_correct = sum(1 for _, _, op in alignment if op == 'C')
    n_sub = sum(1 for _, _, op in alignment if op == 'S')
    n_del = sum(1 for _, _, op in alignment if op == 'D')
    n_ins = sum(1 for _, _, op in alignment if op == 'I')
    wer = (n_sub + n_del + n_ins) / max(len(canonical_tokens), 1)

    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn, n_positions=n,
        precision=p, recall=r, f1=f1, kappa=kappa,
        n_correct=n_correct, n_sub=n_sub, n_del=n_del, n_ins=n_ins, wer=round(wer, 3),
    )


# ═════════════════════════════════════════════════════════════════════════════
#  AGGREGATE METRICS
# ═════════════════════════════════════════════════════════════════════════════

def micro_from_totals(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    obs = (tp + tn) / total if total else 1.0
    pyp = (tp + fp) / total if total else 0.0
    pyg = (tp + fn) / total if total else 0.0
    exp = pyp * pyg + (1 - pyp) * (1 - pyg)
    kappa = ((obs - exp) / (1 - exp)) if abs(1 - exp) > 1e-12 else 1.0
    return dict(precision=p, recall=r, f1=f1, kappa=kappa, n_boundaries=tp + fn)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    with open(SELECTED_FILE) as f:
        all_files = json.load(f)['all_selected']

    print('=' * 70)
    print('RQ1: CLAUSE BOUNDARY AGREEMENT')
    print('    Auto (spaCy on ASR) vs Gold (LLM + expert review)')
    print('    Alignment: edit-distance (Matsuura et al. 2025 / SCTK-style)')
    print('=' * 70)
    print(f'\nAuto clauses dir : {AUTO_CLAUSES_DIR}')
    print(f'Gold blind (10)  : {GOLD_BLIND_DIR}')
    print(f'Gold prod  (30)  : {GOLD_PRODUCTION_DIR}')
    print(f'Transcripts      : {TRANSCRIPTS_DIR}')
    print(f'Selected files   : {len(all_files)}')

    rows = []
    tot = dict(tp=0, fp=0, fn=0, tn=0)
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

        canonical = load_canonical(fid)
        can_tokens = canonical.split()
        auto_segs = auto_clause_segments(tg_path)

        m = rq1_file(gold_segs, auto_segs, can_tokens)
        if m is None:
            skipped.append((fid, 'alignment failed'))
            continue

        for k in ['tp', 'fp', 'fn', 'tn']:
            tot[k] += m[k]

        rows.append(dict(
            file=fid, task=task,
            gold_clauses=len(gold_segs), auto_clauses=len(auto_segs),
            canonical_words=len(can_tokens),
            **m,
        ))

    # ── Report ───────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    st1 = df[df['task'] == 'ST1']
    st2 = df[df['task'] == 'ST2']
    overall_m = micro_from_totals(**tot)

    def task_micro(sub):
        return micro_from_totals(int(sub['tp'].sum()), int(sub['fp'].sum()),
                                 int(sub['fn'].sum()), int(sub['tn'].sum()))
    st1_m = task_micro(st1) if len(st1) else {}
    st2_m = task_micro(st2) if len(st2) else {}

    print(f'\nFiles analysed: {len(df)}  (ST1: {len(st1)}, ST2: {len(st2)})')
    if skipped:
        print(f'Skipped ({len(skipped)}):')
        for f, r in skipped:
            print(f'  {f}: {r}')
    print(f'Mean WER (canonical vs ASR clause-text): {df["wer"].mean():.3f}')

    print(f"\n{'Metric':<16} {'Overall':>10} {'ST1':>10} {'ST2':>10}")
    print('-' * 50)
    for key in ['n_boundaries', 'precision', 'recall', 'f1', 'kappa']:
        vals = [overall_m.get(key, '-'), st1_m.get(key, '-'), st2_m.get(key, '-')]
        fmt = [f'{v:>10.3f}' if isinstance(v, float) else f'{v:>10}' for v in vals]
        print(f'{key:<16} ' + ' '.join(fmt))

    print(f'\nMacro mean F1 : {df["f1"].mean():.3f}')
    print(f'Macro mean κ  : {df["kappa"].mean():.3f}')

    # ── Save CSV ─────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / 'rq1_clause_boundary_gold.csv'
    df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')

    print('\n' + '=' * 70)
    print(f'SUMMARY — RQ1 clause boundary:  micro F1 = {overall_m["f1"]:.3f},  '
          f'κ = {overall_m["kappa"]:.3f}  ({len(df)} files, '
          f'{overall_m["n_boundaries"]} gold boundaries)')
    print('=' * 70)


if __name__ == '__main__':
    main()
