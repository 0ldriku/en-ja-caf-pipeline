#!/usr/bin/env python
"""
Japanese CAF calculator with pause-local VAD + filler-classifier refinement.

This is a non-breaking probe script. It keeps JA CAF computation (mora-based)
and refines pause intervals by:
  1) running VAD inside each pause interval
  2) scoring each VAD island with an acoustic filler classifier
  3) optionally using a bounded VAD fallback for missed low-confidence fillers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
from praatio import textgrid

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, *_THIS.parents]:
    if (_p / "shared").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from shared.filler_classifier.filler_model_inference import FillerProbabilityScorer


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
class ClfRefineStats:
    pause_candidates: int = 0
    pauses_split: int = 0
    vad_islands_detected: int = 0
    clf_islands_scored: int = 0
    predicted_speech_islands: int = 0
    fallback_vad_islands_used: int = 0
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

    def _extract_tiers(self) -> None:
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
        pauses: List[PauseInterval] = []
        for w in self.words:
            if w.is_pause:
                dur = w.end - w.start
                if dur >= min_duration:
                    pauses.append(PauseInterval(start=w.start, end=w.end, duration=dur))
        return pauses


def extract_features(y: np.ndarray, sr: int) -> np.ndarray | None:
    if len(y) < int(0.03 * sr):
        return None
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=512, hop_length=128)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=128)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=512, hop_length=128)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=128)

    feats: List[float] = [
        float(len(y) / float(sr)),
        float(np.mean(rms)),
        float(np.std(rms)),
        float(np.mean(zcr)),
        float(np.std(zcr)),
        float(np.mean(centroid)),
        float(np.std(centroid)),
        float(np.mean(rolloff)),
        float(np.std(rolloff)),
    ]
    feats.extend(np.mean(mfcc, axis=1).tolist())
    feats.extend(np.std(mfcc, axis=1).tolist())
    return np.asarray(feats, dtype=np.float32)


class PauseVADClassifierRefiner:
    def __init__(
        self,
        audio_dir: str,
        model_path: str,
        threshold: float = 0.30,
        min_pause: float = 0.25,
        merge_gap: float = 0.08,
        vad_top_db: float = 30.0,
        min_voiced_dur: float = 0.12,
        min_occupancy: float = 0.12,
        max_occupancy: float = 0.85,
        target_sr: int = 16000,
        fallback_use_vad: bool = False,
        fallback_max_island_dur: float = 0.35,
    ):
        self.audio_dir = Path(audio_dir)
        self.threshold = threshold
        self.min_pause = min_pause
        self.merge_gap = merge_gap
        self.vad_top_db = vad_top_db
        self.min_voiced_dur = min_voiced_dur
        self.min_occupancy = min_occupancy
        self.max_occupancy = max_occupancy
        self.target_sr = target_sr
        self.fallback_use_vad = fallback_use_vad
        self.fallback_max_island_dur = fallback_max_island_dur

        self.scorer = FillerProbabilityScorer(model_path, target_sr=self.target_sr)
        self._audio_cache: Dict[str, Tuple[np.ndarray, int]] = {}

    def _find_audio(self, file_id: str) -> Path:
        for ext in [".wav", ".mp3", ".flac", ".m4a"]:
            p = self.audio_dir / f"{file_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Audio not found for {file_id} under {self.audio_dir}")

    def _load_audio(self, file_id: str) -> Tuple[np.ndarray, int]:
        if file_id in self._audio_cache:
            return self._audio_cache[file_id]
        audio_path = self._find_audio(file_id)
        y, sr = librosa.load(str(audio_path), sr=self.target_sr, mono=True)
        self._audio_cache[file_id] = (y, sr)
        return y, sr

    def _merge_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not intervals:
            return []
        out: List[Tuple[float, float]] = []
        s0, e0 = intervals[0]
        for s1, e1 in intervals[1:]:
            if s1 <= e0 + self.merge_gap:
                e0 = max(e0, e1)
            else:
                out.append((s0, e0))
                s0, e0 = s1, e1
        out.append((s0, e0))
        return out

    def _detect_islands(self, seg: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        raw = librosa.effects.split(seg, top_db=self.vad_top_db, frame_length=1024, hop_length=256)
        islands: List[Tuple[float, float]] = []
        for s, e in raw:
            ss = s / float(sr)
            ee = e / float(sr)
            if (ee - ss) >= self.min_voiced_dur:
                islands.append((ss, ee))
        return self._merge_intervals(islands)

    def _predict_prob(self, clip: np.ndarray, sr: int) -> float | None:
        return self.scorer.predict_proba(clip, sr=sr)

    def refine(self, file_id: str, pauses: List[PauseInterval]) -> Tuple[List[PauseInterval], ClfRefineStats]:
        y, sr = self._load_audio(file_id)
        stats = ClfRefineStats(pause_candidates=len(pauses))
        refined: List[PauseInterval] = []

        for p in pauses:
            s_idx = max(0, int(round(p.start * sr)))
            e_idx = min(len(y), int(round(p.end * sr)))
            if e_idx <= s_idx:
                refined.append(p)
                continue

            seg = y[s_idx:e_idx]
            seg_dur = (e_idx - s_idx) / float(sr)
            islands = self._detect_islands(seg, sr)
            stats.vad_islands_detected += len(islands)

            if not islands or seg_dur <= 0:
                refined.append(p)
                continue

            speech_total = float(sum(e - s for s, e in islands))
            occupancy = speech_total / seg_dur
            if occupancy < self.min_occupancy or occupancy > self.max_occupancy:
                refined.append(p)
                continue

            pos_intervals: List[Tuple[float, float]] = []
            for l0, l1 in islands:
                c0 = max(0, int(round((p.start + l0) * sr)))
                c1 = min(len(y), int(round((p.start + l1) * sr)))
                if c1 <= c0:
                    continue
                prob = self._predict_prob(y[c0:c1], sr=sr)
                if prob is None:
                    continue
                stats.clf_islands_scored += 1
                if prob >= self.threshold:
                    pos_intervals.append((p.start + l0, p.start + l1))

            if not pos_intervals and self.fallback_use_vad:
                fallback = []
                for l0, l1 in islands:
                    if (l1 - l0) <= self.fallback_max_island_dur:
                        fallback.append((p.start + l0, p.start + l1))
                if fallback:
                    pos_intervals = fallback
                    stats.fallback_vad_islands_used += len(fallback)

            if not pos_intervals:
                refined.append(p)
                continue

            overlaps = self._merge_intervals(sorted(pos_intervals, key=lambda x: x[0]))
            silence_parts: List[Tuple[float, float]] = []
            cursor = p.start
            for s, e in overlaps:
                if (s - cursor) >= self.min_pause:
                    silence_parts.append((cursor, s))
                cursor = max(cursor, e)
            if (p.end - cursor) >= self.min_pause:
                silence_parts.append((cursor, p.end))

            # Conservative guard: never delete an entire pause interval.
            if not silence_parts:
                refined.append(p)
                continue

            stats.pauses_split += 1
            stats.predicted_speech_islands += len(overlaps)
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
        refiner: Optional[PauseVADClassifierRefiner] = None,
    ):
        self.tg = tg_reader
        self.min_pause = min_pause
        self.total_mora = tg_reader.total_mora
        self.clauses = tg_reader.clauses
        self.clf_stats = ClfRefineStats()

        raw_pauses = self.tg.get_pauses(self.min_pause)
        if refiner is not None:
            self.pauses, self.clf_stats = refiner.refine(self.tg.file_id, raw_pauses)
        else:
            self.pauses = raw_pauses
            self.clf_stats.refined_pause_count = len(self.pauses)

        self._classify_pauses()

    def _classify_pauses(self) -> None:
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
        runs: List[int] = []
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
            "clf_pause_candidates": self.clf_stats.pause_candidates,
            "clf_pauses_split": self.clf_stats.pauses_split,
            "clf_vad_islands_detected": self.clf_stats.vad_islands_detected,
            "clf_islands_scored": self.clf_stats.clf_islands_scored,
            "clf_predicted_speech_islands": self.clf_stats.predicted_speech_islands,
            "clf_fallback_vad_islands_used": self.clf_stats.fallback_vad_islands_used,
            "clf_refined_pause_count": self.clf_stats.refined_pause_count,
        }


def _load_file_id_filter(json_path: Optional[str], key: str) -> Optional[set]:
    if not json_path:
        return None
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    vals = data.get(key)
    if not isinstance(vals, list):
        raise ValueError(f"Key '{key}' not found or not a list in {json_path}")
    return set(vals)


def process_directory(
    input_dir: str,
    refiner: PauseVADClassifierRefiner,
    min_pause: float = 0.25,
    file_ids_filter: Optional[set] = None,
) -> pd.DataFrame:
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".textgrid")]
    if file_ids_filter:
        files = [f for f in files if Path(f).stem in file_ids_filter]
    rows = []
    for fname in sorted(files):
        path = os.path.join(input_dir, fname)
        try:
            tg_reader = TextGridReaderJA(path)
            calc = CAFCalculatorJA(tg_reader=tg_reader, min_pause=min_pause, refiner=refiner)
            row = calc.calculate_all()
            row["file"] = fname
            rows.append(row)
            print(f"Processed: {fname}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            rows.append({"file": fname, "error": str(e)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="JA CAF with pause-local VAD + filler classifier")
    ap.add_argument("input", help="TextGrid file or directory")
    ap.add_argument("--audio-dir", required=True, help="Directory containing per-file audio")
    ap.add_argument("--model-path", required=True, help="Classifier model path (.joblib or .pt)")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--vad-top-db", type=float, default=30.0)
    ap.add_argument("--vad-min-voiced", type=float, default=0.12)
    ap.add_argument("--vad-merge-gap", type=float, default=0.08)
    ap.add_argument("--vad-min-occupancy", type=float, default=0.12)
    ap.add_argument("--vad-max-occupancy", type=float, default=0.85)
    ap.add_argument("--target-sr", type=int, default=16000)
    ap.add_argument("--fallback-use-vad", action="store_true")
    ap.add_argument("--fallback-max-island-dur", type=float, default=0.35)
    ap.add_argument("--min-pause", type=float, default=0.25)
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--file-list-json", help="Optional JSON file containing file_id list")
    ap.add_argument("--file-list-key", default="all_selected")
    args = ap.parse_args()

    file_filter = _load_file_id_filter(args.file_list_json, args.file_list_key)

    refiner = PauseVADClassifierRefiner(
        audio_dir=args.audio_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        min_pause=args.min_pause,
        merge_gap=args.vad_merge_gap,
        vad_top_db=args.vad_top_db,
        min_voiced_dur=args.vad_min_voiced,
        min_occupancy=args.vad_min_occupancy,
        max_occupancy=args.vad_max_occupancy,
        target_sr=args.target_sr,
        fallback_use_vad=args.fallback_use_vad,
        fallback_max_island_dur=args.fallback_max_island_dur,
    )

    if os.path.isdir(args.input):
        df = process_directory(
            input_dir=args.input,
            refiner=refiner,
            min_pause=args.min_pause,
            file_ids_filter=file_filter,
        )
    else:
        tg_reader = TextGridReaderJA(args.input)
        calc = CAFCalculatorJA(tg_reader=tg_reader, min_pause=args.min_pause, refiner=refiner)
        row = calc.calculate_all()
        row["file"] = os.path.basename(args.input)
        df = pd.DataFrame([row])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")

    for c in [
        "clf_pause_candidates",
        "clf_pauses_split",
        "clf_vad_islands_detected",
        "clf_islands_scored",
        "clf_predicted_speech_islands",
        "clf_fallback_vad_islands_used",
    ]:
        if c in df.columns:
            print(f"{c}: {int(pd.to_numeric(df[c], errors='coerce').fillna(0).sum())}")


if __name__ == "__main__":
    main()
