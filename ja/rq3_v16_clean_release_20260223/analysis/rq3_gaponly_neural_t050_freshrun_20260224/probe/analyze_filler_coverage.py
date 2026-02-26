#!/usr/bin/env python3
"""
Filler coverage analysis for Japanese gold files (40-file RQ3 set).

What this script does:
1) Extract filler intervals from manual and auto TextGrid "words" tiers.
2) Optionally add classifier-positive candidate intervals.
3) Compare overlap against manual filler intervals.
4) Write per-file and summary TSV/Markdown reports.

Outputs (in --out-dir):
  - filler_coverage_per_file.tsv
  - filler_coverage_summary.tsv
  - filler_coverage_summary.md
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Interval = Tuple[float, float]


# Conservative: higher confidence lexical fillers/backchannels.
CONSERVATIVE_FILLERS = {
    "えっと",
    "えーと",
    "えと",
    "えっとー",
    "えーっと",
    "ええと",
    "ええっと",
    "えー",
    "あー",
    "うー",
    "うーん",
    "あのー",
    "そのー",
    "まあ",
    "まあまあ",
    "なんか",
    "なんていうか",
    "なんだっけ",
    "うん",
    "ええ",
    "はい",
    "ああ",
    "へー",
    "ふーん",
    "ね",
    "ねー",
    "ちょっと",
}

# Permissive: includes highly ambiguous short forms/demonstratives.
PERMISSIVE_EXTRA = {"あの", "その", "あ", "え", "ん", "ま"}


EDGE_PUNCT_RE = re.compile(
    r'^[\s\u3000\.,!?！？。、・「」『』\(\)（）\[\]【】]+|[\s\u3000\.,!?！？。、・「」『』\(\)（）\[\]【】]+$'
)


def normalize_surface(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.replace("〜", "ー").replace("～", "ー")
    t = t.replace(" ", "").replace("\u3000", "")
    t = EDGE_PUNCT_RE.sub("", t)
    return t


def parse_short_textgrid_words(lines: Sequence[str], tier_name: str = "words") -> List[Tuple[float, float, str]]:
    out: List[Tuple[float, float, str]] = []
    i = 0
    while i < len(lines):
        token = lines[i].strip().strip('"')
        if token == "IntervalTier":
            if i + 4 >= len(lines):
                break
            name = lines[i + 1].strip().strip('"')
            if name == tier_name:
                n = int(float(lines[i + 4].strip()))
                j = i + 5
                for _ in range(n):
                    start = float(lines[j].strip())
                    end = float(lines[j + 1].strip())
                    label = lines[j + 2].strip().strip('"')
                    out.append((start, end, label))
                    j += 3
                break
        i += 1
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def parse_long_textgrid_words(lines: Sequence[str], tier_name: str = "words") -> List[Tuple[float, float, str]]:
    out: List[Tuple[float, float, str]] = []
    in_target_tier = False

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("name = "):
            in_target_tier = f'"{tier_name}"' in s
            continue

        if in_target_tier and s.startswith("text = "):
            m = re.search(r'text\s*=\s*"(.*?)"', s)
            label = m.group(1) if m else ""

            start = None
            end = None
            for back in range(1, 6):
                if i - back < 0:
                    break
                p = lines[i - back].strip()
                if start is None and p.startswith("xmin"):
                    start = float(re.search(r"[-+]?\d*\.?\d+", p).group())
                if end is None and p.startswith("xmax"):
                    end = float(re.search(r"[-+]?\d*\.?\d+", p).group())

            if start is not None and end is not None:
                out.append((start, end, label))

    out.sort(key=lambda x: (x[0], x[1]))
    return out


def parse_textgrid_words(path: Path, tier_name: str = "words") -> List[Tuple[float, float, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    head = text[:300]
    if "item [" in head or "name = " in head:
        return parse_long_textgrid_words(lines, tier_name=tier_name)
    return parse_short_textgrid_words(lines, tier_name=tier_name)


def merge_intervals(intervals: Iterable[Interval], gap_tolerance: float) -> List[Interval]:
    sorted_iv = sorted((s, e) for s, e in intervals if e > s)
    if not sorted_iv:
        return []

    out: List[List[float]] = [[sorted_iv[0][0], sorted_iv[0][1]]]
    for start, end in sorted_iv[1:]:
        if start <= out[-1][1] + gap_tolerance:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(a, b) for a, b in out]


def overlap_duration(a: Sequence[Interval], b: Sequence[Interval]) -> float:
    i = 0
    j = 0
    overlap = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            overlap += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return overlap


def count_interval_matches(reference: Sequence[Interval], predicted: Sequence[Interval]) -> int:
    matched = 0
    j = 0
    for rs, re_ in reference:
        while j < len(predicted) and predicted[j][1] <= rs:
            j += 1
        k = j
        hit = False
        while k < len(predicted) and predicted[k][0] < re_:
            if min(re_, predicted[k][1]) > max(rs, predicted[k][0]):
                hit = True
                break
            k += 1
        if hit:
            matched += 1
    return matched


def duration(intervals: Sequence[Interval]) -> float:
    return sum(e - s for s, e in intervals)


def read_classifier_intervals(csv_path: Path, merge_gap: float) -> List[Interval]:
    if not csv_path.exists():
        return []

    rows: List[Interval] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = row.get("pred_filler", "")
            if str(pred).strip() not in {"1", "1.0"}:
                continue
            s = row.get("candidate_start", "")
            e = row.get("candidate_end", "")
            if s == "" or e == "":
                continue
            start = float(s)
            end = float(e)
            if end > start:
                rows.append((start, end))
    return merge_intervals(rows, gap_tolerance=merge_gap)


def extract_filler_intervals(words: Sequence[Tuple[float, float, str]], lexicon: set[str], merge_gap: float) -> List[Interval]:
    intervals: List[Interval] = []
    for start, end, label in words:
        if end <= start:
            continue
        if normalize_surface(label) in lexicon:
            intervals.append((start, end))
    return merge_intervals(intervals, gap_tolerance=merge_gap)


def safe_div(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return num / den


def micro_f1(recall_value: float, precision_value: float) -> float:
    if math.isnan(recall_value) or math.isnan(precision_value):
        return float("nan")
    if recall_value + precision_value == 0:
        return float("nan")
    return 2.0 * recall_value * precision_value / (recall_value + precision_value)


def nanmean(values: Sequence[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def analyze(
    manual_dir: Path,
    auto_dir: Path,
    candidate_dir: Path,
    out_dir: Path,
    word_merge_gap: float,
    classifier_merge_gap: float,
) -> None:
    manual_files = {p.stem for p in manual_dir.glob("*.TextGrid")}
    auto_files = {p.stem for p in auto_dir.glob("*.TextGrid")}
    file_ids = sorted(manual_files & auto_files)

    if len(file_ids) == 0:
        raise RuntimeError("No overlapping TextGrid files found between manual and auto directories.")

    lexicons = {
        "conservative": set(CONSERVATIVE_FILLERS),
        "permissive": set(CONSERVATIVE_FILLERS) | set(PERMISSIVE_EXTRA),
    }

    per_file_rows: List[Dict[str, object]] = []

    for file_id in file_ids:
        manual_words = parse_textgrid_words(manual_dir / f"{file_id}.TextGrid", tier_name="words")
        auto_words = parse_textgrid_words(auto_dir / f"{file_id}.TextGrid", tier_name="words")
        classifier_iv = read_classifier_intervals(
            candidate_dir / f"{file_id}_vad_classifier.csv",
            merge_gap=classifier_merge_gap,
        )

        for lex_name, lexicon in lexicons.items():
            manual_iv = extract_filler_intervals(manual_words, lexicon, merge_gap=word_merge_gap)
            auto_word_iv = extract_filler_intervals(auto_words, lexicon, merge_gap=word_merge_gap)
            auto_plus_iv = merge_intervals(auto_word_iv + classifier_iv, gap_tolerance=classifier_merge_gap)

            pred_sets = {
                "words_only": auto_word_iv,
                "words_plus_classifier": auto_plus_iv,
            }

            for pred_set_name, predicted_iv in pred_sets.items():
                manual_dur = duration(manual_iv)
                pred_dur = duration(predicted_iv)
                ov_dur = overlap_duration(manual_iv, predicted_iv)

                manual_n = len(manual_iv)
                pred_n = len(predicted_iv)
                matched_manual_n = count_interval_matches(manual_iv, predicted_iv) if manual_n else 0
                matched_pred_n = count_interval_matches(predicted_iv, manual_iv) if pred_n else 0

                dur_recall = safe_div(ov_dur, manual_dur)
                dur_precision = safe_div(ov_dur, pred_dur)
                dur_f1 = micro_f1(dur_recall, dur_precision)
                ev_recall = safe_div(float(matched_manual_n), float(manual_n))
                ev_precision = safe_div(float(matched_pred_n), float(pred_n))

                per_file_rows.append(
                    {
                        "file_id": file_id,
                        "lexicon_mode": lex_name,
                        "pred_set": pred_set_name,
                        "manual_event_count": manual_n,
                        "pred_event_count": pred_n,
                        "matched_manual_event_count": matched_manual_n,
                        "matched_pred_event_count": matched_pred_n,
                        "manual_filler_duration_sec": round(manual_dur, 6),
                        "pred_filler_duration_sec": round(pred_dur, 6),
                        "overlap_duration_sec": round(ov_dur, 6),
                        "duration_recall": dur_recall,
                        "duration_precision": dur_precision,
                        "duration_f1": dur_f1,
                        "event_recall": ev_recall,
                        "event_precision": ev_precision,
                        "classifier_interval_count": len(classifier_iv),
                        "classifier_duration_sec": round(duration(classifier_iv), 6),
                    }
                )

    # Aggregate summaries.
    summary_rows: List[Dict[str, object]] = []
    for lex_name in ("conservative", "permissive"):
        for pred_set_name in ("words_only", "words_plus_classifier"):
            rows = [
                r
                for r in per_file_rows
                if r["lexicon_mode"] == lex_name and r["pred_set"] == pred_set_name
            ]

            manual_d = sum(float(r["manual_filler_duration_sec"]) for r in rows)
            pred_d = sum(float(r["pred_filler_duration_sec"]) for r in rows)
            ov_d = sum(float(r["overlap_duration_sec"]) for r in rows)
            mn = sum(int(r["manual_event_count"]) for r in rows)
            pn = sum(int(r["pred_event_count"]) for r in rows)
            mh = sum(int(r["matched_manual_event_count"]) for r in rows)
            ph = sum(int(r["matched_pred_event_count"]) for r in rows)

            dur_recall_micro = safe_div(ov_d, manual_d)
            dur_precision_micro = safe_div(ov_d, pred_d)
            dur_f1_micro = micro_f1(dur_recall_micro, dur_precision_micro)
            ev_recall_micro = safe_div(float(mh), float(mn))
            ev_precision_micro = safe_div(float(ph), float(pn))

            summary_rows.append(
                {
                    "lexicon_mode": lex_name,
                    "pred_set": pred_set_name,
                    "n_files": len(rows),
                    "files_with_manual_fillers": sum(int(int(r["manual_event_count"]) > 0) for r in rows),
                    "files_with_pred_fillers": sum(int(int(r["pred_event_count"]) > 0) for r in rows),
                    "manual_event_count": mn,
                    "pred_event_count": pn,
                    "matched_manual_event_count": mh,
                    "matched_pred_event_count": ph,
                    "event_recall_micro": ev_recall_micro,
                    "event_precision_micro": ev_precision_micro,
                    "manual_filler_duration_sec": manual_d,
                    "pred_filler_duration_sec": pred_d,
                    "overlap_duration_sec": ov_d,
                    "duration_recall_micro": dur_recall_micro,
                    "duration_precision_micro": dur_precision_micro,
                    "duration_f1_micro": dur_f1_micro,
                    "duration_recall_macro": nanmean([float(r["duration_recall"]) for r in rows]),
                    "duration_precision_macro": nanmean([float(r["duration_precision"]) for r in rows]),
                    "duration_f1_macro": nanmean([float(r["duration_f1"]) for r in rows]),
                    "event_recall_macro": nanmean([float(r["event_recall"]) for r in rows]),
                    "event_precision_macro": nanmean([float(r["event_precision"]) for r in rows]),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    per_file_path = out_dir / "filler_coverage_per_file.tsv"
    summary_path = out_dir / "filler_coverage_summary.tsv"
    summary_md_path = out_dir / "filler_coverage_summary.md"

    per_file_header = list(per_file_rows[0].keys())
    with per_file_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_file_header, delimiter="\t")
        writer.writeheader()
        for row in per_file_rows:
            writer.writerow(row)

    summary_header = list(summary_rows[0].keys())
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_header, delimiter="\t")
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    # Minimal markdown report for direct manuscript integration.
    lines = [
        "# Filler Coverage Summary",
        "",
        f"- Files analyzed: {len(file_ids)}",
        f"- Manual TextGrids: `{manual_dir}`",
        f"- Auto TextGrids: `{auto_dir}`",
        f"- Classifier candidate intervals: `{candidate_dir}`",
        "",
        "## Micro Metrics",
        "",
        "| Lexicon | Pred Set | Dur Recall | Dur Precision | Dur F1 | Event Recall | Event Precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {lex} | {pred} | {dr:.3f} | {dp:.3f} | {df1:.3f} | {er:.3f} | {ep:.3f} |".format(
                lex=row["lexicon_mode"],
                pred=row["pred_set"],
                dr=float(row["duration_recall_micro"]) if not math.isnan(float(row["duration_recall_micro"])) else float("nan"),
                dp=float(row["duration_precision_micro"]) if not math.isnan(float(row["duration_precision_micro"])) else float("nan"),
                df1=float(row["duration_f1_micro"]) if not math.isnan(float(row["duration_f1_micro"])) else float("nan"),
                er=float(row["event_recall_micro"]) if not math.isnan(float(row["event_recall_micro"])) else float("nan"),
                ep=float(row["event_precision_micro"]) if not math.isnan(float(row["event_precision_micro"])) else float("nan"),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `conservative` excludes highly ambiguous short forms (`あの/その/あ/え/ん/ま`).",
            "- `permissive` includes those forms as filler tokens.",
            "- `words_plus_classifier` adds predicted filler intervals from per-file classifier CSVs (`pred_filler == 1`).",
        ]
    )
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {per_file_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {summary_md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze filler interval coverage against manual Japanese TextGrids.")
    parser.add_argument(
        "--manual-dir",
        type=Path,
        default=Path("ja/rq3_v16_clean_release_20260223/results/manual_clauses_gold_v2"),
        help="Manual gold TextGrid directory",
    )
    parser.add_argument(
        "--auto-dir",
        type=Path,
        default=Path(
            "ja/rq3_v16_clean_release_20260223/results/"
            "qwen3_filler_mfa_ja_v2_spanfix_b_l1_focus20/textgrids_clean_beginning_removed_by_manual"
        ),
        help="Auto TextGrid directory",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path(
            "ja/rq3_v16_clean_release_20260223/analysis/"
            "rq3_gaponly_neural_t050_freshrun_20260224/candidates/per_file"
        ),
        help="Classifier candidate CSV directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "ja/rq3_v16_clean_release_20260223/analysis/"
            "rq3_gaponly_neural_t050_freshrun_20260224/probe"
        ),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--word-merge-gap",
        type=float,
        default=0.05,
        help="Merge gap tolerance (seconds) for adjacent filler word intervals",
    )
    parser.add_argument(
        "--classifier-merge-gap",
        type=float,
        default=0.02,
        help="Merge gap tolerance (seconds) for classifier candidate intervals",
    )
    args = parser.parse_args()

    analyze(
        manual_dir=args.manual_dir,
        auto_dir=args.auto_dir,
        candidate_dir=args.candidate_dir,
        out_dir=args.out_dir,
        word_merge_gap=args.word_merge_gap,
        classifier_merge_gap=args.classifier_merge_gap,
    )


if __name__ == "__main__":
    main()

