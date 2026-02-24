"""
CAF Measures Calculator

Calculates 9 CAF (Complexity, Accuracy, Fluency) measures from TextGrid files
with clause annotations.

Measures:
  Speed:
    - AR: Articulation rate (syllables / phonation time)
  Breakdown:
    - MCPR: Mid-clause pause ratio (mid-clause pauses / syllables)
    - ECPR: End-clause pause ratio (end-clause pauses / syllables)
    - PR: Pause ratio (all pauses / syllables)
    - MCPD: Mid-clause pause duration (mean)
    - ECPD: End-clause pause duration (mean)
    - MPD: Mean pause duration
  Composite:
    - SR: Speech rate (syllables / total duration)
    - MLR: Mean length of run (mean syllables between pauses)

Usage:
    python caf_calculator.py <textgrid_dir> [--min-pause 0.25] [--output results.csv]
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
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


# ==============================================================================
# TextGrid Reader
# ==============================================================================

class TextGridReader:
    """Read TextGrid files with words, phones, and clauses tiers."""
    
    PAUSE_MARKERS = {'sp', 'sil', 'spn', 'noise', '', '<sil>', '<sp>', '<spn>',
                     '<p>', '<pause>', 'breath', '<breath>'}
    FILLER_MARKERS = {'<filler_speech>', '<filler>', 'filler_speech'}
    
    VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY',
              'EH', 'ER', 'EY', 'IH', 'IX', 'IY', 'OW', 'OY',
              'UH', 'UW', 'UX'}
    
    def __init__(self, filepath: str):
        self.filepath = filepath
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
        for name in ['words', 'word', 'Word', 'Words']:
            if name in tier_names:
                word_tier_name = name
                break
        if not word_tier_name:
            word_tier_name = tier_names[0]
        
        # Find phone tier
        phone_tier_name = None
        for name in ['phones', 'phone', 'Phone', 'Phones']:
            if name in tier_names:
                phone_tier_name = name
                break
        
        # Find clause tier
        clause_tier_name = None
        for name in ['clauses', 'clause', 'Clauses', 'Clause']:
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
            is_pause = text.lower() in self.PAUSE_MARKERS or text == ''
            is_filler_marker = text.lower() in self.FILLER_MARKERS
            
            syllables = 0
            if is_filler_marker:
                syllables = 0
            elif not is_pause and self.phones:
                syllables = self._count_syllables_in_range(entry.start, entry.end)
            elif not is_pause:
                # Fallback: estimate syllables from text
                syllables = self._estimate_syllables(text)
            
            self.words.append(WordInterval(
                start=entry.start,
                end=entry.end,
                text=text,
                is_pause=is_pause,
                syllable_count=syllables
            ))
            
            if (not is_pause) and (not is_filler_marker):
                self.total_syllables += syllables
        
        # Extract clauses
        if clause_tier_name:
            clause_tier = self.tg.getTier(clause_tier_name)
            for entry in clause_tier.entries:
                text = entry.label.strip()
                if text:  # Non-empty clause
                    self.clauses.append(ClauseInterval(
                        start=entry.start,
                        end=entry.end,
                        text=text
                    ))
    
    def _count_syllables_in_range(self, start: float, end: float) -> int:
        count = 0
        for p_start, p_end, label in self.phones:
            if p_start >= start and p_end <= end:
                base_phone = label.split(',')[0].strip()
                base_phone = re.sub(r'[0-9*]', '', base_phone).upper()
                if base_phone in self.VOWELS:
                    count += 1
        return max(count, 1) if count == 0 else count
    
    def _estimate_syllables(self, text: str) -> int:
        """Estimate syllables when phone tier not available."""
        text = text.lower().strip()
        if not text:
            return 0
        
        # Simple vowel counting
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        # Handle silent e
        if text.endswith('e') and count > 1:
            count -= 1
        
        return max(count, 1)
    
    def get_pauses(self, min_duration: float = 0.25) -> List[PauseInterval]:
        pauses = []
        for w in self.words:
            if w.is_pause:
                duration = w.end - w.start
                if duration >= min_duration:
                    pauses.append(PauseInterval(
                        start=w.start,
                        end=w.end,
                        duration=duration
                    ))
        return pauses


# ==============================================================================
# CAF Calculator
# ==============================================================================

class CAFCalculator:
    """Calculate 9 CAF measures from TextGrid with clause annotations."""
    
    def __init__(self, tg_reader: TextGridReader, min_pause: float = 0.25):
        self.tg = tg_reader
        self.min_pause = min_pause
        self.total_syllables = tg_reader.total_syllables
        self.clauses = tg_reader.clauses
        self._classify_pauses()
    
    def _classify_pauses(self):
        """Classify pauses as mid-clause or end-clause."""
        self.pauses = self.tg.get_pauses(self.min_pause)
        self.mid_clause_pauses = []
        self.end_clause_pauses = []
        
        for pause in self.pauses:
            pause_mid = (pause.start + pause.end) / 2
            is_mid_clause = False
            is_end_clause = False
            
            for clause in self.clauses:
                # Check if pause is at end of clause (within 150ms tolerance)
                if abs(pause.start - clause.end) < 0.15:
                    is_end_clause = True
                    break
                # Check if pause is within clause
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
                # Default to end-clause for pauses between clauses
                pause.location = "end-clause"
                self.end_clause_pauses.append(pause)
    
    def calculate_all(self) -> Dict:
        """Calculate all 9 CAF measures."""
        if not self.tg.words:
            return {'error': 'No words found'}
        
        # Basic timing
        total_duration = self.tg.words[-1].end - self.tg.words[0].start
        total_pause_duration = sum(p.duration for p in self.pauses)
        phonation_time = total_duration - total_pause_duration
        syllables = self.total_syllables
        
        # Pause counts
        n_mid_clause = len(self.mid_clause_pauses)
        n_end_clause = len(self.end_clause_pauses)
        n_total_pauses = len(self.pauses)
        
        # Pause durations
        mid_clause_durations = [p.duration for p in self.mid_clause_pauses]
        end_clause_durations = [p.duration for p in self.end_clause_pauses]
        all_pause_durations = [p.duration for p in self.pauses]
        
        # Calculate measures
        AR = syllables / phonation_time if phonation_time > 0 else 0
        MCPR = n_mid_clause / syllables if syllables > 0 else 0
        ECPR = n_end_clause / syllables if syllables > 0 else 0
        PR = n_total_pauses / syllables if syllables > 0 else 0
        MCPD = np.mean(mid_clause_durations) if mid_clause_durations else 0
        ECPD = np.mean(end_clause_durations) if end_clause_durations else 0
        MPD = np.mean(all_pause_durations) if all_pause_durations else 0
        SR = syllables / total_duration if total_duration > 0 else 0
        
        # Mean length of run
        runs = self._calculate_syllable_runs()
        MLR = np.mean(runs) if runs else 0
        
        return {
            'AR': round(AR, 3),
            'SR': round(SR, 3),
            'MLR': round(MLR, 2),
            'MCPR': round(MCPR, 4),
            'ECPR': round(ECPR, 4),
            'PR': round(PR, 4),
            'MCPD': round(MCPD, 3),
            'ECPD': round(ECPD, 3),
            'MPD': round(MPD, 3),
            'total_syllables': syllables,
            'total_duration': round(total_duration, 2),
            'phonation_time': round(phonation_time, 2),
            'n_pauses': n_total_pauses,
            'n_mid_clause_pauses': n_mid_clause,
            'n_end_clause_pauses': n_end_clause,
            'n_clauses': len(self.clauses)
        }
    
    def _calculate_syllable_runs(self) -> List[int]:
        """Calculate syllable runs between pauses."""
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

def process_textgrid(filepath: str, min_pause: float = 0.25) -> Dict:
    """Process a single TextGrid file and return CAF measures."""
    tg_reader = TextGridReader(filepath)
    calculator = CAFCalculator(tg_reader, min_pause)
    measures = calculator.calculate_all()
    measures['file'] = os.path.basename(filepath)
    return measures


def process_directory(input_dir: str, min_pause: float = 0.25) -> pd.DataFrame:
    """Process all TextGrid files in a directory."""
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.textgrid')]
    
    results = []
    for filename in sorted(files):
        filepath = os.path.join(input_dir, filename)
        try:
            measures = process_textgrid(filepath, min_pause)
            results.append(measures)
            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append({'file': filename, 'error': str(e)})
    
    return pd.DataFrame(results)


def print_measures_table(df: pd.DataFrame):
    """Print measures in a formatted table."""
    main_cols = ['file', 'AR', 'SR', 'MLR', 'MCPR', 'ECPR', 'PR', 'MCPD', 'ECPD', 'MPD']
    display_cols = [c for c in main_cols if c in df.columns]
    print("\n" + "=" * 100)
    print("CAF MEASURES (9 measures)")
    print("=" * 100)
    print(df[display_cols].to_string(index=False))
    
    print("\n" + "=" * 100)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 100)
    stats_cols = ['AR', 'SR', 'MLR', 'MCPR', 'ECPR', 'PR', 'MCPD', 'ECPD', 'MPD']
    stats_cols = [c for c in stats_cols if c in df.columns]
    print(df[stats_cols].describe().round(3).to_string())


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate CAF measures from TextGrid files with clause annotations."
    )
    parser.add_argument(
        "input",
        help="TextGrid file or directory containing TextGrid files"
    )
    parser.add_argument(
        "--min-pause", "-p",
        type=float,
        default=0.25,
        help="Minimum pause duration in seconds (default: 0.25)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output CSV file path (optional)"
    )
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # Single file
        measures = process_textgrid(args.input, args.min_pause)
        df = pd.DataFrame([measures])
    else:
        # Directory
        df = process_directory(args.input, args.min_pause)
    
    print_measures_table(df)
    
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
