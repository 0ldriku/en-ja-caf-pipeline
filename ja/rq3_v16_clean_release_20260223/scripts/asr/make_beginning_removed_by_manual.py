#!/usr/bin/env python
"""
Blank ASR TextGrid labels before manual first non-blank word.

This reproduces the "textgrids_clean_beginning_removed_by_manual" style input:
  - Keep audio/time axis unchanged.
  - Keep interval boundaries unchanged.
  - Only blank labels in ASR tiers before manual speech onset.

Optional:
  --use-full-span also blanks labels after manual last non-blank word.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from praatio import textgrid
from praatio.utilities.constants import Interval


def _tier_names(tg: textgrid.Textgrid) -> List[str]:
    if hasattr(tg, "tierNames"):
        return list(getattr(tg, "tierNames"))
    if hasattr(tg, "tierNameList"):
        return list(getattr(tg, "tierNameList"))
    # Fallback for older/newer internal layouts
    if hasattr(tg, "_tierDict"):
        return list(getattr(tg, "_tierDict").keys())
    raise RuntimeError("Could not read tier names from TextGrid.")


def _get_tier(tg: textgrid.Textgrid, name: str):
    if hasattr(tg, "getTier"):
        return tg.getTier(name)
    if hasattr(tg, "_tierDict"):
        return tg._tierDict[name]
    raise RuntimeError("Could not access tier in TextGrid.")


def _find_words_tier_name(tg: textgrid.Textgrid) -> str:
    names = _tier_names(tg)
    for n in names:
        if n.lower() == "words":
            return n
    if not names:
        raise RuntimeError("TextGrid has no tiers.")
    return names[0]


def _manual_span(manual_tg_path: Path) -> Tuple[float, float]:
    tg = textgrid.openTextgrid(str(manual_tg_path), includeEmptyIntervals=True)
    tier_name = _find_words_tier_name(tg)
    tier = _get_tier(tg, tier_name)
    nonblank = [e for e in tier.entries if e.label.strip()]
    if not nonblank:
        raise RuntimeError(f"No non-blank words in manual TextGrid: {manual_tg_path}")
    return float(nonblank[0].start), float(nonblank[-1].end)


def _blank_labels(
    entries: Iterable[Interval],
    cut_start: float,
    cut_end: float | None,
) -> List[Interval]:
    out: List[Interval] = []
    for e in entries:
        label = e.label
        if e.start < cut_start:
            label = ""
        if cut_end is not None and e.end > cut_end:
            label = ""
        out.append(Interval(e.start, e.end, label))
    return out


def process_one(
    asr_tg_path: Path,
    manual_tg_path: Path,
    out_tg_path: Path,
    use_full_span: bool,
) -> Tuple[float, float]:
    start, end = _manual_span(manual_tg_path)
    tg = textgrid.openTextgrid(str(asr_tg_path), includeEmptyIntervals=True)

    names = _tier_names(tg)
    out_tg = textgrid.Textgrid()

    for name in names:
        tier = _get_tier(tg, name)
        new_entries = _blank_labels(
            tier.entries,
            cut_start=start,
            cut_end=(end if use_full_span else None),
        )
        min_t = getattr(tier, "minTimestamp", new_entries[0].start if new_entries else 0.0)
        max_t = getattr(tier, "maxTimestamp", new_entries[-1].end if new_entries else min_t)
        out_tier = textgrid.IntervalTier(name, new_entries, minT=min_t, maxT=max_t)
        out_tg.addTier(out_tier)

    out_tg.save(str(out_tg_path), format="long_textgrid", includeBlankSpaces=True)
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create ASR TextGrids with leading labels blanked by manual onset."
    )
    parser.add_argument(
        "--asr-dir",
        required=True,
        type=Path,
        help="Input ASR TextGrid directory (e.g., textgrids_clean).",
    )
    parser.add_argument(
        "--manual-dir",
        required=True,
        type=Path,
        help="Manual TextGrid directory used as span reference.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for processed ASR TextGrids.",
    )
    parser.add_argument(
        "--use-full-span",
        action="store_true",
        help="Also blank labels after manual last non-blank word.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    asr_files = sorted(args.asr_dir.glob("*.TextGrid"))
    if not asr_files:
        raise SystemExit(f"No TextGrid files found: {args.asr_dir}")

    processed = 0
    skipped = 0
    for asr_path in asr_files:
        manual_path = args.manual_dir / asr_path.name
        if not manual_path.exists():
            print(f"[skip] manual missing: {asr_path.name}")
            skipped += 1
            continue

        out_path = args.out_dir / asr_path.name
        try:
            start, end = process_one(
                asr_tg_path=asr_path,
                manual_tg_path=manual_path,
                out_tg_path=out_path,
                use_full_span=args.use_full_span,
            )
            processed += 1
            mode = "start+end" if args.use_full_span else "start-only"
            print(f"[ok] {asr_path.stem}: {mode} span=({start:.3f}, {end:.3f})")
        except Exception as exc:
            print(f"[error] {asr_path.stem}: {exc}")
            skipped += 1

    print(
        f"Done. processed={processed}, skipped={skipped}, "
        f"out_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()

