#!/usr/bin/env python
"""
Japanese CAF calculator with optional audio-based pause refinement (VAD-like).

Keeps the same 9 CAF measures as caf_calculator_ja.py, but can split empty-word
pause intervals when non-silent islands are detected inside them.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


@dataclass
class VadRefineStats:
    pause_candidates: int = 0
    pauses_split: int = 0
    speech_islands: int = 0
    refined_pause_count: int = 0


class TextGridReaderJA:
    PAUSE_MARKERS = {
        "",
        "sp",
        "sil",
        "spn",
        "noise",
        "<sil>",
        "<sp>",
        "<spn>",
        "<p>",
        "<pause>",
        "pause",
        "breath",
        "<breath>",
        "silb",
        "sile",
        "#",
        "...",
    }
    FILLER_MARKERS = {"<filler_speech>", "<filler>", "filler_speech"}

    SMALL_KANA = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮっッー")

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_id = Path(filepath).stem
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
                        ClauseInterval(start=float(entry.start), end=float(entry.end), text=txt)
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


class AudioPauseRefiner:
    """
    Split long empty-word pauses when non-silent islands are found in audio.
    Uses librosa.effects.split (VAD-like silence splitter).
    """

    def __init__(
        self,
        audio_path: str,
        vad_top_db: float = 30.0,
        min_voiced_dur: float = 0.15,
        merge_gap: float = 0.10,
        min_occupancy: float = 0.20,
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
        import librosa

        # librosa handles mp3/wav robustly across environments
        y, sr = librosa.load(path, sr=None, mono=True)
        return y.astype(np.float32), int(sr)

    @staticmethod
    def _merge(intervals: List[Tuple[float, float]], max_gap: float) -> List[Tuple[float, float]]:
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        out = [[intervals[0][0], intervals[0][1]]]
        for s, e in intervals[1:]:
            if s - out[-1][1] <= max_gap:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return [(s, e) for s, e in out]

    def _split_pause(self, start: float, end: float) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        if end <= start:
            return None
        s_idx = max(0, int(start * self.sr))
        e_idx = min(len(self.samples), int(end * self.sr))
        if e_idx <= s_idx:
            return None
        seg = self.samples[s_idx:e_idx]
        seg_dur = len(seg) / self.sr
        if seg_dur <= 0:
            return None

        import librosa

        voiced_raw = librosa.effects.split(seg, top_db=self.vad_top_db, frame_length=1024, hop_length=256)
        voiced = []
        for s, e in voiced_raw:
            ss = start + s / self.sr
            ee = start + e / self.sr
            if (ee - ss) >= self.min_voiced_dur:
                voiced.append((ss, ee))
        voiced = self._merge(voiced, self.merge_gap)
        if not voiced:
            return None

        occ = sum((e - s) for s, e in voiced) / seg_dur
        if occ < self.min_occupancy or occ > self.max_occupancy:
            return None

        silences = []
        cur = start
        for s, e in voiced:
            if (s - cur) >= self.min_pause:
                silences.append((cur, s))
            cur = max(cur, e)
        if (end - cur) >= self.min_pause:
            silences.append((cur, end))
        return silences, voiced

    def refine(self, pauses: List[PauseInterval]) -> Tuple[List[PauseInterval], VadRefineStats]:
        stats = VadRefineStats(pause_candidates=len(pauses))
        refined: List[PauseInterval] = []
        for p in pauses:
            out = self._split_pause(p.start, p.end)
            if out is None:
                refined.append(p)
                continue
            silence_parts, voiced_parts = out
            if not silence_parts:
                refined.append(p)
                continue
            stats.pauses_split += 1
            stats.speech_islands += len(voiced_parts)
            for s, e in silence_parts:
                refined.append(PauseInterval(start=s, end=e, duration=e - s))
        refined.sort(key=lambda x: (x.start, x.end))
        stats.refined_pause_count = len(refined)
        return refined, stats


class CAFCalculatorJA:
    def __init__(
        self,
        tg_reader: TextGridReaderJA,
        min_pause: float = 0.25,
        use_audio_vad: bool = False,
        audio_path: Optional[str] = None,
        vad_options: Optional[Dict] = None,
    ):
        self.tg = tg_reader
        self.min_pause = min_pause
        self.total_mora = tg_reader.total_mora
        self.clauses = tg_reader.clauses
        self.vad_stats = VadRefineStats()

        pauses = self.tg.get_pauses(self.min_pause)
        if use_audio_vad and audio_path:
            refiner = AudioPauseRefiner(audio_path=audio_path, min_pause=min_pause, **(vad_options or {}))
            pauses, self.vad_stats = refiner.refine(pauses)
        self.pauses = pauses
        self._classify_pauses()

    def _classify_pauses(self):
        self.mid_clause_pauses: List[PauseInterval] = []
        self.end_clause_pauses: List[PauseInterval] = []
        for pause in self.pauses:
            pause_mid = 0.5 * (pause.start + pause.end)
            is_mid = False
            is_end = False
            for clause in self.clauses:
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
            "vad_pause_candidates": self.vad_stats.pause_candidates,
            "vad_pauses_split": self.vad_stats.pauses_split,
            "vad_speech_islands": self.vad_stats.speech_islands,
            "vad_refined_pause_count": self.vad_stats.refined_pause_count,
        }


def _find_audio(audio_dir: Optional[str], file_id: str) -> Optional[str]:
    if not audio_dir:
        return None
    base = Path(audio_dir)
    for ext in [".wav", ".mp3", ".flac", ".m4a"]:
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
    tg_reader = TextGridReaderJA(filepath)
    calc = CAFCalculatorJA(
        tg_reader,
        min_pause=min_pause,
        use_audio_vad=use_audio_vad,
        audio_path=audio_path,
        vad_options=vad_options,
    )
    out = calc.calculate_all()
    out["file"] = os.path.basename(filepath)
    if use_audio_vad:
        out["audio_path"] = audio_path or ""
    return out


def process_directory(
    input_dir: str,
    min_pause: float = 0.25,
    use_audio_vad: bool = False,
    audio_dir: Optional[str] = None,
    vad_options: Optional[Dict] = None,
) -> pd.DataFrame:
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".textgrid")]
    rows = []
    for fname in sorted(files):
        path = os.path.join(input_dir, fname)
        try:
            audio_path = _find_audio(audio_dir, Path(fname).stem) if use_audio_vad else None
            row = process_textgrid(
                path,
                min_pause=min_pause,
                use_audio_vad=use_audio_vad,
                audio_path=audio_path,
                vad_options=vad_options,
            )
            rows.append(row)
            print(f"Processed: {fname}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            rows.append({"file": fname, "error": str(e)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="TextGrid file or directory")
    ap.add_argument("--min-pause", type=float, default=0.25)
    ap.add_argument("--output", "-o")
    ap.add_argument("--use-audio-vad", action="store_true")
    ap.add_argument("--audio-dir")
    ap.add_argument("--vad-top-db", type=float, default=30.0)
    ap.add_argument("--vad-min-voiced", type=float, default=0.15)
    ap.add_argument("--vad-merge-gap", type=float, default=0.10)
    ap.add_argument("--vad-min-occupancy", type=float, default=0.20)
    ap.add_argument("--vad-max-occupancy", type=float, default=0.85)
    args = ap.parse_args()

    if args.use_audio_vad and not args.audio_dir:
        raise ValueError("--audio-dir required when --use-audio-vad is set")

    vad_opts = {
        "vad_top_db": args.vad_top_db,
        "min_voiced_dur": args.vad_min_voiced,
        "merge_gap": args.vad_merge_gap,
        "min_occupancy": args.vad_min_occupancy,
        "max_occupancy": args.vad_max_occupancy,
    }

    if os.path.isfile(args.input):
        audio_path = _find_audio(args.audio_dir, Path(args.input).stem) if args.use_audio_vad else None
        df = pd.DataFrame(
            [
                process_textgrid(
                    args.input,
                    min_pause=args.min_pause,
                    use_audio_vad=args.use_audio_vad,
                    audio_path=audio_path,
                    vad_options=vad_opts,
                )
            ]
        )
    else:
        df = process_directory(
            args.input,
            min_pause=args.min_pause,
            use_audio_vad=args.use_audio_vad,
            audio_dir=args.audio_dir,
            vad_options=vad_opts,
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved: {args.output}")
    else:
        print(df.to_string(index=False))

    if "vad_pauses_split" in df.columns:
        for c in ["vad_pause_candidates", "vad_pauses_split", "vad_speech_islands"]:
            if c in df.columns:
                print(f"{c}: {int(pd.to_numeric(df[c], errors='coerce').fillna(0).sum())}")


if __name__ == "__main__":
    main()
