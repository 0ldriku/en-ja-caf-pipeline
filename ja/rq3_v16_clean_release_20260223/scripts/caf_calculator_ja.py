#!/usr/bin/env python
"""
Japanese CAF Measures Calculator

Calculates the same 9 CAF measures as en/scripts/caf_calculator.py,
but uses mora counts for Japanese text.

Measures:
  Speed:
    - AR: Articulation rate (mora / phonation time)
  Breakdown:
    - MCPR: Mid-clause pause ratio (mid-clause pauses / mora)
    - ECPR: End-clause pause ratio (end-clause pauses / mora)
    - PR: Pause ratio (all pauses / mora)
    - MCPD: Mid-clause pause duration (mean)
    - ECPD: End-clause pause duration (mean)
    - MPD: Mean pause duration
  Composite:
    - SR: Speech rate (mora / total duration)
    - MLR: Mean length of run (mean mora between pauses)

Usage:
    python ja/caf_calculator_ja.py <textgrid_dir_or_file> --output <results.csv>
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from praatio import textgrid


@dataclass
class WordInterval:
    start: float
    end: float
    text: str
    is_pause: bool
    mora_count: int = 0


@dataclass
class PauseInterval:
    start: float
    end: float
    duration: float
    location: str = "unknown"


@dataclass
class ClauseInterval:
    start: float
    end: float
    text: str
    clause_type: str = "clause"


class TextGridReaderJA:
    """Read TextGrid files with words (+ optional clauses tier) for Japanese CAF."""

    PAUSE_MARKERS = {
        "", "sp", "sil", "spn", "noise", "<sil>", "<sp>", "<spn>",
        "<p>", "<pause>", "pause", "breath", "<breath>", "silB", "silE", "#", "..."
    }
    FILLER_MARKERS = {"<filler_speech>", "<filler>", "filler_speech"}

    # Small kana are absorbed into previous mora. Long vowel mark is treated as small here
    # to match current ja_clause_segmenter mora heuristic.
    SMALL_KANA = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮっッー")

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.tg = textgrid.openTextgrid(filepath, includeEmptyIntervals=True)
        self.words: List[WordInterval] = []
        self.clauses: List[ClauseInterval] = []
        self.total_mora = 0
        self._extract_tiers()

    def _extract_tiers(self):
        tier_names = self.tg.tierNames

        word_tier_name = None
        for name in ["words", "word", "Word", "Words", "単語", "transcription"]:
            if name in tier_names:
                word_tier_name = name
                break
        if not word_tier_name:
            word_tier_name = tier_names[0]

        clause_tier_name = None
        for name in ["clauses", "clause", "Clauses", "Clause"]:
            if name in tier_names:
                clause_tier_name = name
                break

        word_tier = self.tg.getTier(word_tier_name)
        for entry in word_tier.entries:
            text = entry.label.strip()
            is_pause = text.lower() in self.PAUSE_MARKERS or text == ""
            is_filler_marker = text.lower() in self.FILLER_MARKERS
            mora = 0 if (is_pause or is_filler_marker) else self._count_mora(text)
            self.words.append(
                WordInterval(
                    start=float(entry.start),
                    end=float(entry.end),
                    text=text,
                    is_pause=is_pause,
                    mora_count=mora,
                )
            )
            if (not is_pause) and (not is_filler_marker):
                self.total_mora += mora

        if clause_tier_name:
            clause_tier = self.tg.getTier(clause_tier_name)
            for entry in clause_tier.entries:
                txt = entry.label.strip()
                if txt:
                    self.clauses.append(
                        ClauseInterval(
                            start=float(entry.start),
                            end=float(entry.end),
                            text=txt,
                        )
                    )

    def _count_mora(self, text: str) -> int:
        count = 0
        for ch in text:
            if ch.isspace():
                continue
            if ch in self.SMALL_KANA:
                continue
            count += 1
        return max(count, 1) if text else 0

    def get_pauses(self, min_duration: float = 0.25) -> List[PauseInterval]:
        pauses = []
        for w in self.words:
            if w.is_pause:
                dur = w.end - w.start
                if dur >= min_duration:
                    pauses.append(PauseInterval(start=w.start, end=w.end, duration=dur))
        return pauses


class CAFCalculatorJA:
    """Calculate 9 CAF measures from Japanese TextGrid with clause annotations."""

    def __init__(self, tg_reader: TextGridReaderJA, min_pause: float = 0.25):
        self.tg = tg_reader
        self.min_pause = min_pause
        self.total_mora = tg_reader.total_mora
        self.clauses = tg_reader.clauses
        self._classify_pauses()

    def _classify_pauses(self):
        self.pauses = self.tg.get_pauses(self.min_pause)
        self.mid_clause_pauses = []
        self.end_clause_pauses = []

        for pause in self.pauses:
            pause_mid = 0.5 * (pause.start + pause.end)
            is_mid = False
            is_end = False

            for clause in self.clauses:
                # Same rule as EN calculator
                if abs(pause.start - clause.end) < 0.15:
                    is_end = True
                    break
                if clause.start < pause_mid < clause.end:
                    is_mid = True
                    break

            if is_end:
                pause.location = "end-clause"
                self.end_clause_pauses.append(pause)
            elif is_mid:
                pause.location = "mid-clause"
                self.mid_clause_pauses.append(pause)
            else:
                pause.location = "end-clause"
                self.end_clause_pauses.append(pause)

    def _calculate_mora_runs(self) -> List[int]:
        runs = []
        cur = 0
        for w in self.tg.words:
            if w.is_pause:
                if (w.end - w.start) >= self.min_pause:
                    if cur > 0:
                        runs.append(cur)
                    cur = 0
            else:
                cur += w.mora_count
        if cur > 0:
            runs.append(cur)
        return runs

    def calculate_all(self) -> Dict:
        if not self.tg.words:
            return {"error": "No words found"}

        total_duration = self.tg.words[-1].end - self.tg.words[0].start
        total_pause_duration = sum(p.duration for p in self.pauses)
        phonation_time = total_duration - total_pause_duration
        mora = self.total_mora

        n_mid = len(self.mid_clause_pauses)
        n_end = len(self.end_clause_pauses)
        n_all = len(self.pauses)

        mid_durs = [p.duration for p in self.mid_clause_pauses]
        end_durs = [p.duration for p in self.end_clause_pauses]
        all_durs = [p.duration for p in self.pauses]

        ar = mora / phonation_time if phonation_time > 0 else 0
        mcpr = n_mid / mora if mora > 0 else 0
        ecpr = n_end / mora if mora > 0 else 0
        pr = n_all / mora if mora > 0 else 0
        mcpd = float(np.mean(mid_durs)) if mid_durs else 0
        ecpd = float(np.mean(end_durs)) if end_durs else 0
        mpd = float(np.mean(all_durs)) if all_durs else 0
        sr = mora / total_duration if total_duration > 0 else 0
        runs = self._calculate_mora_runs()
        mlr = float(np.mean(runs)) if runs else 0

        return {
            "AR": round(ar, 3),
            "SR": round(sr, 3),
            "MLR": round(mlr, 2),
            "MCPR": round(mcpr, 4),
            "ECPR": round(ecpr, 4),
            "PR": round(pr, 4),
            "MCPD": round(mcpd, 3),
            "ECPD": round(ecpd, 3),
            "MPD": round(mpd, 3),
            "total_mora": mora,
            "total_duration": round(total_duration, 2),
            "phonation_time": round(phonation_time, 2),
            "n_pauses": n_all,
            "n_mid_clause_pauses": n_mid,
            "n_end_clause_pauses": n_end,
            "n_clauses": len(self.clauses),
        }


def process_textgrid(filepath: str, min_pause: float = 0.25) -> Dict:
    tg_reader = TextGridReaderJA(filepath)
    calc = CAFCalculatorJA(tg_reader, min_pause=min_pause)
    out = calc.calculate_all()
    out["file"] = os.path.basename(filepath)
    return out


def process_directory(input_dir: str, min_pause: float = 0.25) -> pd.DataFrame:
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".textgrid")]
    rows = []
    for fname in sorted(files):
        path = os.path.join(input_dir, fname)
        try:
            row = process_textgrid(path, min_pause=min_pause)
            rows.append(row)
            print(f"Processed: {fname}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            rows.append({"file": fname, "error": str(e)})
    return pd.DataFrame(rows)


def print_measures_table(df: pd.DataFrame):
    cols_main = ["file", "AR", "SR", "MLR", "MCPR", "ECPR", "PR", "MCPD", "ECPD", "MPD"]
    cols = [c for c in cols_main if c in df.columns]
    print("\n" + "=" * 100)
    print("JAPANESE CAF MEASURES (9 measures, mora-based)")
    print("=" * 100)
    print(df[cols].to_string(index=False))

    stat_cols = ["AR", "SR", "MLR", "MCPR", "ECPR", "PR", "MCPD", "ECPD", "MPD"]
    stat_cols = [c for c in stat_cols if c in df.columns]
    print("\n" + "=" * 100)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 100)
    print(df[stat_cols].describe().round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description="Calculate Japanese CAF measures from TextGrid files.")
    parser.add_argument("input", help="TextGrid file or directory")
    parser.add_argument("--min-pause", "-p", type=float, default=0.25)
    parser.add_argument("--output", "-o", help="Output CSV path")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        df = pd.DataFrame([process_textgrid(args.input, min_pause=args.min_pause)])
    else:
        df = process_directory(args.input, min_pause=args.min_pause)

    print_measures_table(df)
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
