#!/usr/bin/env python3
"""
Apply gold_v2 clause boundary fixes to TextGrid files.
Run with: ja/.venv_electra310/Scripts/python.exe <this_script>
"""

import os
import sys
from praatio import textgrid

GOLD_DIR = os.path.dirname(os.path.abspath(__file__))
GOLD_V2 = os.path.join(GOLD_DIR, "manual_clauses_gold_v2")


def find_word_near(word_tier, label, near_time, tolerance=2.0):
    """Find a word by label closest to near_time within tolerance."""
    best = None
    best_dist = float("inf")
    for entry in word_tier.entries:
        if entry.label == label:
            dist = abs(entry.start - near_time)
            if dist < best_dist and dist < tolerance:
                best_dist = dist
                best = entry
    if best is None:
        raise ValueError(f"Word '{label}' not found near time {near_time:.3f}")
    return best


def apply_boundary_move(tg_path, clause_text_1, clause_text_2,
                         last_word_label, new_label_1, new_label_2):
    """Move boundary: find last_word in word tier near old boundary, set new boundary there."""
    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
    clause_tier = tg.getTier("clauses")
    word_tier = tg.getTier("words")
    entries = list(clause_tier.entries)

    idx1 = idx2 = None
    for i, e in enumerate(entries):
        if clause_text_1 in e.label and idx1 is None:
            idx1 = i
        elif clause_text_2 in e.label and idx1 is not None and idx2 is None:
            idx2 = i
            break

    if idx1 is None or idx2 is None:
        raise ValueError(f"Could not find clause pair: '{clause_text_1}' + '{clause_text_2}'")

    e1, e2 = entries[idx1], entries[idx2]
    old_boundary = e1.end

    word = find_word_near(word_tier, last_word_label, old_boundary)
    new_boundary = word.end

    new_entries = []
    for j, entry in enumerate(entries):
        if j == idx1:
            new_entries.append(entry._replace(end=new_boundary, label=new_label_1))
        elif j == idx2:
            new_entries.append(entry._replace(start=new_boundary, label=new_label_2))
        elif idx1 < j < idx2:
            new_entries.append(entry._replace(start=new_boundary, end=new_boundary))
        else:
            new_entries.append(entry)

    _save_clause_tier(tg, clause_tier, new_entries, tg_path)
    log(f"  OK: boundary {old_boundary:.3f} -> {new_boundary:.3f}")


def apply_full_merge(tg_path, clause_text_1, clause_text_2, new_label):
    """Merge two clauses (and any gaps between) into one."""
    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
    clause_tier = tg.getTier("clauses")
    entries = list(clause_tier.entries)

    idx1 = idx2 = None
    for i, e in enumerate(entries):
        if clause_text_1 in e.label and idx1 is None:
            idx1 = i
        elif clause_text_2 in e.label and idx1 is not None and idx2 is None:
            idx2 = i
            break

    if idx1 is None or idx2 is None:
        raise ValueError(f"Could not find clause pair: '{clause_text_1}' + '{clause_text_2}'")

    e1, e2 = entries[idx1], entries[idx2]
    new_end = e2.end

    new_entries = []
    for j, entry in enumerate(entries):
        if j == idx1:
            new_entries.append(entry._replace(end=new_end, label=new_label))
        elif idx1 < j <= idx2:
            pass  # absorbed
        else:
            new_entries.append(entry)

    _save_clause_tier(tg, clause_tier, new_entries, tg_path)
    log(f"  OK: merged [{e1.start:.2f}-{e1.end:.2f}] + [{e2.start:.2f}-{e2.end:.2f}]")


def _save_clause_tier(tg, old_tier, new_entries, tg_path):
    """Replace clause tier with new entries and save."""
    filtered = [(e.start, e.end, e.label) for e in new_entries if e.end > e.start + 0.0001]
    new_tier = textgrid.IntervalTier(
        old_tier.name, filtered, old_tier.minTimestamp, old_tier.maxTimestamp,
    )
    tg.removeTier(old_tier.name)
    tg.addTier(new_tier)
    tg.save(tg_path, format="short_textgrid", includeBlankSpaces=True)


def log(msg):
    sys.stdout.buffer.write((msg + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def main():
    fixes = [
        # Fix 1: CCS45-ST1 - auxiliary split
        {"type": "merge", "file": "CCS45-ST1.TextGrid",
         "clause_1": "一緒に連れて", "clause_2": "いきたいで",
         "label": "一緒に連れていきたいで一緒に遠足に行きました",
         "rule": "auxiliary ていく must not split from main verb"},
        # Fix 2: CCS45-ST2 - verb inflection split
        {"type": "move", "file": "CCS45-ST2.TextGrid",
         "clause_1": "はんぱくし", "clause_2": "ました良かった",
         "last_word": "た", "label_1": "うそですで警官はまたはんぱくしました",
         "label_2": "良かった",
         "rule": "verb inflection (masu+ta) must not split"},
        # Fix 3: CCT15-ST2 - particle split
        {"type": "move", "file": "CCT15-ST2.TextGrid",
         "clause_1": "家に入る", "clause_2": "のかしら",
         "last_word": "かしら",
         "label_1": "その後はしょうがないのでやっぱり何の方法で家に入るのかしら",
         "label_2": "ってことを真剣に考えるとは",
         "rule": "のか subordinating particle must stay with verb"},
        # Fix 4: ENZ31-ST2 - verb inflection split
        {"type": "move", "file": "ENZ31-ST2.TextGrid",
         "clause_1": "思いまし", "clause_2": "たが警官",
         "last_word": "が",
         "label_1": "ケンは思いましたが",
         "label_2": "警官に見られてしまいました",
         "rule": "verb inflection (masu+ta+ga) must not split"},
        # Fix 5: EUS32-ST1 - verb inflection split
        {"type": "move", "file": "EUS32-ST1.TextGrid",
         "clause_1": "できませ", "clause_2": "んになりました",
         "last_word": "ん",
         "label_1": "そしてもうピクニックできません",
         "label_2": "になりました",
         "rule": "verb inflection (mase+n) must not split"},
        # Fix 6: KKR40-ST2 - te-form split
        {"type": "move", "file": "KKR40-ST2.TextGrid",
         "clause_1": "聞い", "clause_2": "てあそうゆう",
         "last_word": "て",
         "label_1": "そして警官もそれを聞いて",
         "label_2": "あそうゆうことそうゆうことかと思い",
         "rule": "te-form ending must not split from verb stem"},
    ]

    log(f"Applying {len(fixes)} fixes to gold_v2 TextGrids...")
    log(f"Directory: {GOLD_V2}")

    ok_count = 0
    for i, fix in enumerate(fixes, 1):
        tg_path = os.path.join(GOLD_V2, fix["file"])
        log(f"\nFix {i}: {fix['file']} ({fix['rule']})")

        if not os.path.exists(tg_path):
            log(f"  ERROR: File not found")
            continue

        try:
            if fix["type"] == "merge":
                apply_full_merge(tg_path, fix["clause_1"], fix["clause_2"], fix["label"])
            else:
                apply_boundary_move(
                    tg_path, fix["clause_1"], fix["clause_2"],
                    fix["last_word"], fix["label_1"], fix["label_2"]
                )
            ok_count += 1
        except Exception as e:
            log(f"  ERROR: {e}")

    log(f"\n=== {ok_count}/{len(fixes)} fixes applied ===")


if __name__ == "__main__":
    main()
