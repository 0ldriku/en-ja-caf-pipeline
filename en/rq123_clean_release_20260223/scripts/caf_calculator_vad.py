"""
CAF Measures Calculator with optional audio-based pause refinement.

This is a non-breaking variant of caf_calculator.py. It keeps the same 9 CAF
measures, and optionally refines pause intervals using audio VAD-like splitting
inside empty-word intervals.

Why:
  In ASR TextGrids, missed filler tokens can merge multiple hesitation segments
  into one long "silent" interval, inflating MCPD. This script can split such
  intervals when speech activity is detected inside them.

Usage examples:
  Baseline-compatible:
    python caf_calculator_vad.py <textgrid_dir> -o out.csv

  With audio pause refinement:
    python caf_calculator_vad.py <textgrid_dir> --use-audio-vad \
      --audio-dir <wav_dir> -o out.csv

  Gold-40 subset run:
    python caf_calculator_vad.py <textgrid_dir> --use-audio-vad \
      --audio-dir <wav_dir> \
      --file-list-json annotation/selected_files.json \
      --file-list-key all_selected \
      -o out_gold40.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from praatio import textgrid


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class WordInterval:
    start: float
    end: float
    text: str
    is_pause: bool
    syllable_count: int = 0


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


@dataclass
class VadRefineStats:
    pause_candidates: int = 0
    pauses_split: int = 0
    inserted_speech_islands: int = 0
    refined_pause_count: int = 0


# ==============================================================================
# TextGrid Reader
# ==============================================================================


class TextGridReader:
    """Read TextGrid files with words, phones, and clauses tiers."""

    PAUSE_MARKERS = {
        "sp",
        "sil",
        "spn",
        "noise",
        "",
        "<sil>",
        "<sp>",
        "<spn>",
        "<p>",
        "<pause>",
        "breath",
        "<breath>",
    }
    FILLER_MARKERS = {"<filler_speech>", "<filler>", "filler_speech"}

    VOWELS = {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AX",
        "AXR",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IX",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
        "UX",
    }

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_id = Path(filepath).stem
        self.tg = textgrid.openTextgrid(filepath, includeEmptyIntervals=True)
        self.words: List[WordInterval] = []
        self.phones: List[Tuple[float, float, str]] = []
        self.clauses: List[ClauseInterval] = []
        self.total_syllables = 0
        self._extract_tiers()

    def _extract_tiers(self):
        tier_names = self.tg.tierNames

        # Find word tier
        word_tier_name = None
        for name in ["words", "word", "Word", "Words"]:
            if name in tier_names:
                word_tier_name = name
                break
        if not word_tier_name:
            word_tier_name = tier_names[0]

        # Find phone tier
        phone_tier_name = None
        for name in ["phones", "phone", "Phone", "Phones"]:
            if name in tier_names:
                phone_tier_name = name
                break

        # Find clause tier
        clause_tier_name = None
        for name in ["clauses", "clause", "Clauses", "Clause"]:
            if name in tier_names:
                clause_tier_name = name
                break

        # Extract phones
        if phone_tier_name:
            phone_tier = self.tg.getTier(phone_tier_name)
            for entry in phone_tier.entries:
                self.phones.append((entry.start, entry.end, entry.label.strip()))

        # Extract words
        word_tier = self.tg.getTier(word_tier_name)
        for entry in word_tier.entries:
            text = entry.label.strip()
            is_pause = text.lower() in self.PAUSE_MARKERS or text == ""
            is_filler_marker = text.lower() in self.FILLER_MARKERS

            syllables = 0
            if is_filler_marker:
                syllables = 0
            elif not is_pause and self.phones:
                syllables = self._count_syllables_in_range(entry.start, entry.end)
            elif not is_pause:
                syllables = self._estimate_syllables(text)

            self.words.append(
                WordInterval(
                    start=entry.start,
                    end=entry.end,
                    text=text,
                    is_pause=is_pause,
                    syllable_count=syllables,
                )
            )

            if (not is_pause) and (not is_filler_marker):
                self.total_syllables += syllables

        # Extract clauses
        if clause_tier_name:
            clause_tier = self.tg.getTier(clause_tier_name)
            for entry in clause_tier.entries:
                text = entry.label.strip()
                if text:
                    self.clauses.append(
                        ClauseInterval(start=entry.start, end=entry.end, text=text)
                    )

    def _count_syllables_in_range(self, start: float, end: float) -> int:
        count = 0
        for p_start, p_end, label in self.phones:
            if p_start >= start and p_end <= end:
                base_phone = label.split(",")[0].strip()
                base_phone = re.sub(r"[0-9*]", "", base_phone).upper()
                if base_phone in self.VOWELS:
                    count += 1
        return max(count, 1) if count == 0 else count

    def _estimate_syllables(self, text: str) -> int:
        text = text.lower().strip()
        if not text:
            return 0

        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        if text.endswith("e") and count > 1:
            count -= 1

        return max(count, 1)

    def get_pauses(self, min_duration: float = 0.25) -> List[PauseInterval]:
        pauses = []
        for w in self.words:
            if w.is_pause:
                duration = w.end - w.start
                if duration >= min_duration:
                    pauses.append(PauseInterval(start=w.start, end=w.end, duration=duration))
        return pauses


# ==============================================================================
# Audio Pause Refiner
# ==============================================================================


class AudioPauseRefiner:
    """
    Split long empty-word intervals when speech activity is detected inside.

    Uses librosa.effects.split as a practical VAD-like detector:
      - detects non-silent islands in each pause interval
      - if speech occupancy is plausible (not too tiny / not fully dense),
        splits the pause into silent sub-intervals
    """

    def __init__(
        self,
        audio_path: str,
        vad_top_db: float = 30.0,
        min_voiced_dur: float = 0.12,
        merge_gap: float = 0.08,
        min_occupancy: float = 0.12,
        max_occupancy: float = 0.85,
        min_pause: float = 0.25,
    ):
        self.audio_path = audio_path
        self.vad_top_db = vad_top_db
        self.min_voiced_dur = min_voiced_dur
        self.merge_gap = merge_gap
        self.min_occupancy = min_occupancy
        self.max_occupancy = max_occupancy
        self.min_pause = min_pause
        self.samples, self.sr = self._load_audio(audio_path)

    @staticmethod
    def _load_audio(path: str) -> Tuple[np.ndarray, int]:
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y.astype(np.float32), int(sr)

    def refine(self, pauses: List[PauseInterval]) -> Tuple[List[PauseInterval], VadRefineStats]:
        stats = VadRefineStats(pause_candidates=len(pauses))
        refined: List[PauseInterval] = []

        for p in pauses:
            split_result = self._split_pause_with_vad(p.start, p.end)
            if split_result is None:
                refined.append(p)
                continue

            silence_parts, speech_parts = split_result
            if not silence_parts:
                # If everything looks speechy, keep original to avoid dropping pauses.
                refined.append(p)
                continue

            stats.pauses_split += 1
            stats.inserted_speech_islands += len(speech_parts)
            for s, e in silence_parts:
                refined.append(PauseInterval(start=s, end=e, duration=e - s))

        stats.refined_pause_count = len(refined)
        refined.sort(key=lambda x: (x.start, x.end))
        return refined, stats

    def _split_pause_with_vad(
        self, start: float, end: float
    ) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        if end <= start:
            return None

        s_idx = max(0, int(start * self.sr))
        e_idx = min(len(self.samples), int(end * self.sr))
        if e_idx <= s_idx:
            return None
        segment = self.samples[s_idx:e_idx]
        seg_dur = len(segment) / self.sr

        # Lazy import so baseline mode has no librosa requirement.
        import librosa

        voiced = librosa.effects.split(
            segment,
            top_db=self.vad_top_db,
            frame_length=1024,
            hop_length=256,
        )

        speech_parts: List[Tuple[float, float]] = []
        for vs, ve in voiced:
            ss = start + (vs / self.sr)
            ee = start + (ve / self.sr)
            if (ee - ss) >= self.min_voiced_dur:
                speech_parts.append((ss, ee))

        speech_parts = self._merge_close_intervals(speech_parts, self.merge_gap)
        if not speech_parts:
            return None

        speech_total = sum(ee - ss for ss, ee in speech_parts)
        occupancy = speech_total / seg_dur if seg_dur > 0 else 0.0
        if occupancy < self.min_occupancy or occupancy > self.max_occupancy:
            return None

        silence_parts: List[Tuple[float, float]] = []
        cursor = start
        for ss, ee in speech_parts:
            if (ss - cursor) >= self.min_pause:
                silence_parts.append((cursor, ss))
            cursor = max(cursor, ee)
        if (end - cursor) >= self.min_pause:
            silence_parts.append((cursor, end))

        return silence_parts, speech_parts

    @staticmethod
    def _merge_close_intervals(
        intervals: List[Tuple[float, float]], merge_gap: float
    ) -> List[Tuple[float, float]]:
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        out: List[List[float]] = [[intervals[0][0], intervals[0][1]]]
        for s, e in intervals[1:]:
            if s - out[-1][1] <= merge_gap:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return [(s, e) for s, e in out]


# ==============================================================================
# CAF Calculator
# ==============================================================================


class CAFCalculator:
    """Calculate 9 CAF measures from TextGrid with optional audio pause refinement."""

    def __init__(
        self,
        tg_reader: TextGridReader,
        min_pause: float = 0.25,
        use_audio_vad: bool = False,
        audio_path: Optional[str] = None,
        vad_options: Optional[Dict] = None,
    ):
        self.tg = tg_reader
        self.min_pause = min_pause
        self.total_syllables = tg_reader.total_syllables
        self.clauses = tg_reader.clauses
        self.vad_stats = VadRefineStats()

        raw_pauses = self.tg.get_pauses(self.min_pause)
        if use_audio_vad and audio_path:
            refiner = AudioPauseRefiner(audio_path=audio_path, min_pause=min_pause, **(vad_options or {}))
            self.pauses, self.vad_stats = refiner.refine(raw_pauses)
        else:
            self.pauses = raw_pauses

        self._classify_pauses()

    def _classify_pauses(self):
        self.mid_clause_pauses: List[PauseInterval] = []
        self.end_clause_pauses: List[PauseInterval] = []

        for pause in self.pauses:
            pause_mid = (pause.start + pause.end) / 2
            is_mid_clause = False
            is_end_clause = False

            for clause in self.clauses:
                # End-clause tolerance follows the baseline calculator.
                if abs(pause.start - clause.end) < 0.15:
                    is_end_clause = True
                    break
                if clause.start < pause_mid < clause.end:
                    is_mid_clause = True
                    break

            if is_end_clause:
                pause.location = "end-clause"
                self.end_clause_pauses.append(pause)
            elif is_mid_clause:
                pause.location = "mid-clause"
                self.mid_clause_pauses.append(pause)
            else:
                pause.location = "end-clause"
                self.end_clause_pauses.append(pause)

    def calculate_all(self) -> Dict:
        if not self.tg.words:
            return {"error": "No words found"}

        total_duration = self.tg.words[-1].end - self.tg.words[0].start
        total_pause_duration = sum(p.duration for p in self.pauses)
        phonation_time = total_duration - total_pause_duration
        syllables = self.total_syllables

        n_mid_clause = len(self.mid_clause_pauses)
        n_end_clause = len(self.end_clause_pauses)
        n_total_pauses = len(self.pauses)

        mid_clause_durations = [p.duration for p in self.mid_clause_pauses]
        end_clause_durations = [p.duration for p in self.end_clause_pauses]
        all_pause_durations = [p.duration for p in self.pauses]

        AR = syllables / phonation_time if phonation_time > 0 else 0
        MCPR = n_mid_clause / syllables if syllables > 0 else 0
        ECPR = n_end_clause / syllables if syllables > 0 else 0
        PR = n_total_pauses / syllables if syllables > 0 else 0
        MCPD = np.mean(mid_clause_durations) if mid_clause_durations else 0
        ECPD = np.mean(end_clause_durations) if end_clause_durations else 0
        MPD = np.mean(all_pause_durations) if all_pause_durations else 0
        SR = syllables / total_duration if total_duration > 0 else 0

        # Keep baseline behavior for MLR (word-tier pause markers).
        runs = self._calculate_syllable_runs()
        MLR = np.mean(runs) if runs else 0

        return {
            "AR": round(AR, 3),
            "SR": round(SR, 3),
            "MLR": round(MLR, 2),
            "MCPR": round(MCPR, 4),
            "ECPR": round(ECPR, 4),
            "PR": round(PR, 4),
            "MCPD": round(MCPD, 3),
            "ECPD": round(ECPD, 3),
            "MPD": round(MPD, 3),
            "total_syllables": syllables,
            "total_duration": round(total_duration, 2),
            "phonation_time": round(phonation_time, 2),
            "n_pauses": n_total_pauses,
            "n_mid_clause_pauses": n_mid_clause,
            "n_end_clause_pauses": n_end_clause,
            "n_clauses": len(self.clauses),
            "vad_pause_candidates": self.vad_stats.pause_candidates,
            "vad_pauses_split": self.vad_stats.pauses_split,
            "vad_speech_islands": self.vad_stats.inserted_speech_islands,
            "vad_refined_pause_count": self.vad_stats.refined_pause_count,
        }

    def _calculate_syllable_runs(self) -> List[int]:
        runs = []
        current_run = 0

        for word in self.tg.words:
            if word.is_pause:
                if (word.end - word.start) >= self.min_pause:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            else:
                current_run += word.syllable_count

        if current_run > 0:
            runs.append(current_run)

        return runs


# ==============================================================================
# Processing Functions
# ==============================================================================


def _find_audio_for_file(audio_dir: Optional[str], file_id: str) -> Optional[str]:
    if not audio_dir:
        return None
    base = Path(audio_dir)
    for ext in [".wav", ".flac", ".mp3", ".m4a"]:
        cand = base / f"{file_id}{ext}"
        if cand.exists():
            return str(cand)
    return None


def process_textgrid(
    filepath: str,
    min_pause: float = 0.25,
    use_audio_vad: bool = False,
    audio_path: Optional[str] = None,
    vad_options: Optional[Dict] = None,
) -> Dict:
    tg_reader = TextGridReader(filepath)
    calculator = CAFCalculator(
        tg_reader=tg_reader,
        min_pause=min_pause,
        use_audio_vad=use_audio_vad,
        audio_path=audio_path,
        vad_options=vad_options,
    )
    measures = calculator.calculate_all()
    measures["file"] = os.path.basename(filepath)
    if use_audio_vad:
        measures["audio_path"] = audio_path or ""
    return measures


def process_directory(
    input_dir: str,
    min_pause: float = 0.25,
    use_audio_vad: bool = False,
    audio_dir: Optional[str] = None,
    file_ids_filter: Optional[set] = None,
    vad_options: Optional[Dict] = None,
) -> pd.DataFrame:
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".textgrid")]

    if file_ids_filter:
        files = [f for f in files if Path(f).stem in file_ids_filter]

    results = []
    for filename in sorted(files):
        filepath = os.path.join(input_dir, filename)
        file_id = Path(filename).stem
        try:
            audio_path = _find_audio_for_file(audio_dir, file_id) if use_audio_vad else None
            measures = process_textgrid(
                filepath=filepath,
                min_pause=min_pause,
                use_audio_vad=use_audio_vad,
                audio_path=audio_path,
                vad_options=vad_options,
            )
            results.append(measures)
            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append({"file": filename, "error": str(e)})

    return pd.DataFrame(results)


def print_measures_table(df: pd.DataFrame):
    main_cols = ["file", "AR", "SR", "MLR", "MCPR", "ECPR", "PR", "MCPD", "ECPD", "MPD"]
    display_cols = [c for c in main_cols if c in df.columns]
    print("\n" + "=" * 100)
    print("CAF MEASURES (9 measures)")
    print("=" * 100)
    print(df[display_cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 100)
    stats_cols = ["AR", "SR", "MLR", "MCPR", "ECPR", "PR", "MCPD", "ECPD", "MPD"]
    stats_cols = [c for c in stats_cols if c in df.columns]
    print(df[stats_cols].describe().round(3).to_string())

    if "vad_pauses_split" in df.columns:
        print("\n" + "=" * 100)
        print("VAD REFINEMENT SUMMARY")
        print("=" * 100)
        totals = {
            "vad_pause_candidates": int(pd.to_numeric(df["vad_pause_candidates"], errors="coerce").fillna(0).sum()),
            "vad_pauses_split": int(pd.to_numeric(df["vad_pauses_split"], errors="coerce").fillna(0).sum()),
            "vad_speech_islands": int(pd.to_numeric(df["vad_speech_islands"], errors="coerce").fillna(0).sum()),
        }
        print(totals)


def _load_file_id_filter(json_path: Optional[str], key: str) -> Optional[set]:
    if not json_path:
        return None
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    values = data.get(key)
    if not isinstance(values, list):
        raise ValueError(f"Key '{key}' not found or not a list in {json_path}")
    return set(values)


# ==============================================================================
# Main
# ==============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate CAF measures from TextGrid files with optional audio pause refinement."
    )
    parser.add_argument("input", help="TextGrid file or directory containing TextGrid files")
    parser.add_argument(
        "--min-pause",
        "-p",
        type=float,
        default=0.25,
        help="Minimum pause duration in seconds (default: 0.25)",
    )
    parser.add_argument("--output", "-o", help="Output CSV file path (optional)")

    parser.add_argument(
        "--use-audio-vad",
        action="store_true",
        help="Refine pause intervals by detecting non-silent islands in audio.",
    )
    parser.add_argument(
        "--audio-dir",
        help="Directory with audio files named <file_id>.wav (or flac/mp3/m4a).",
    )
    parser.add_argument("--vad-top-db", type=float, default=30.0, help="librosa split top_db (default: 30)")
    parser.add_argument("--vad-min-voiced", type=float, default=0.12, help="Min voiced island duration in sec")
    parser.add_argument("--vad-merge-gap", type=float, default=0.08, help="Merge speech islands within this gap")
    parser.add_argument(
        "--vad-min-occupancy",
        type=float,
        default=0.12,
        help="Only split pauses if speech occupancy >= this ratio",
    )
    parser.add_argument(
        "--vad-max-occupancy",
        type=float,
        default=0.85,
        help="Only split pauses if speech occupancy <= this ratio",
    )

    parser.add_argument(
        "--file-list-json",
        help="Optional JSON file containing a list of file_ids (e.g., selected_files.json).",
    )
    parser.add_argument(
        "--file-list-key",
        default="all_selected",
        help="Key in --file-list-json to use as file_id list (default: all_selected).",
    )

    args = parser.parse_args()

    if args.use_audio_vad and not args.audio_dir:
        raise ValueError("--audio-dir is required when --use-audio-vad is set")

    file_filter = _load_file_id_filter(args.file_list_json, args.file_list_key)
    vad_options = {
        "vad_top_db": args.vad_top_db,
        "min_voiced_dur": args.vad_min_voiced,
        "merge_gap": args.vad_merge_gap,
        "min_occupancy": args.vad_min_occupancy,
        "max_occupancy": args.vad_max_occupancy,
    }

    if os.path.isfile(args.input):
        audio_path = _find_audio_for_file(args.audio_dir, Path(args.input).stem) if args.use_audio_vad else None
        measures = process_textgrid(
            filepath=args.input,
            min_pause=args.min_pause,
            use_audio_vad=args.use_audio_vad,
            audio_path=audio_path,
            vad_options=vad_options,
        )
        df = pd.DataFrame([measures])
    else:
        df = process_directory(
            input_dir=args.input,
            min_pause=args.min_pause,
            use_audio_vad=args.use_audio_vad,
            audio_dir=args.audio_dir,
            file_ids_filter=file_filter,
            vad_options=vad_options,
        )

    print_measures_table(df)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
