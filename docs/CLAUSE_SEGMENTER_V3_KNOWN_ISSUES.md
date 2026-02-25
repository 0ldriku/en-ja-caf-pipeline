# Clause Segmenter V3 — Known Issues

## Validated: 70 files, 2510 clauses, 99.2% clean (3 remaining >20w, 16 LONG 16-20w)

### Alignment Edge Cases (2 instances in 21 files)

1. **003_ST1 clause 90**: `"and after all the cylinder that he found"` followed by `"served a very good purpose"`.
   These should be one clause but the relative clause `"that he found"` caused the aligner to split the parent clause awkwardly. The verb `"found"` is in the text so verbless merge cannot catch it.

2. **010_ST2 clause 29**: `"a stick and started"` — words from different logical clauses ended up in the same aligned interval due to word-matching drift in long (500+ word) files.

**Root cause**: The alignment algorithm matches logical clause tokens to TextGrid words sequentially. In very long files with heavy disfluency, token matching can drift, placing words from different clauses into one interval.

**Impact**: ~0.2% of clauses. Only occurs in the longest, most complex files (500+ words). Does not affect clause counts significantly.

### Alignment Drift in High-Disfluency Files (fixed — root cause)

Previously, the aligner used **linear interpolation** (`token_idx * n_words / n_tokens`) to estimate where each logical clause starts in the original word list. When 40-50% of words are disfluent (common with CMN L1 speakers), this linear mapping was wildly inaccurate — `estimated_start` could overshoot by 20-50 positions. The distance guard then rejected valid matches, leaving huge stretches unclaimed and producing catastrophic 50-60 word single "clauses" (3 files affected: 016_ST1, 021_ST1, 039_ST1).

**Fix applied**: `preprocess_text` now returns `kept_indices` — a list mapping each surviving processed word to its original word position. In `align_clauses`, a proper **spaCy token → processed word → original word** mapping is built using character offsets (`bisect` on `tok.idx`). This gives accurate `estimated_start` regardless of disfluency rate.

**Result**: All 3 catastrophic files now segment correctly (max clause: 11-17 words).

### Unclaimed Recovery Improvements (fixed)

1. **Verb-containing groups mislabeled as fragment**: The unclaimed recovery re-parsed group text and required ROOT to be a verb. Messy L2 text (e.g., `<unk>`, fillers) confused the re-parse, so groups like "a baby bird with skinny leg pop out" stayed as `fragment` and got merged into adjacent clauses by Pass 3. **Fix**: Simplified to `has_verb + ≥3 non-filler words → parataxis`.

2. **Pass 3 merging large fragments**: Fragments >5 displayed words were blindly merged into adjacent clauses. **Fix**: Reclassify as `parataxis` instead of merging.

3. **Unclaimed group splitting**: Large unclaimed groups with a verb now split at the first conjunction AFTER the last verb/particle. E.g., "a baby bird pop out **|** and the little bird" → verb clause (parataxis) + trailing NP (fragment merged into next clause). Prevents subject NPs from being attached to the wrong clause.

### Pass 4: Split over-long clauses at parataxis boundaries (fixed)

Clauses >20 displayed words are re-parsed with spaCy. If the parse finds a `parataxis` dependency on ROOT, the clause is split at the leftmost token of the parataxis subtree. Both halves must be ≥3 displayed words.

**Example**: 007_ST2 #5 (22w) → split into 14w + 8w:
- `"and also as any other five year old he was a very creative boy"` (ROOT clause)
- `"one of his favorite ways to play was"` (parataxis)

### 3 CRITICAL Clauses (>20w) — all legitimate enumerations, cannot be split

1. **016_ST2 #25** (22w): `"and then all his bubbles including the cars the birds the butterflies the elephants and the tigers broke all in the sky"` — Single clause with enumerated subject NP. One verb ("broke").

2. **030_ST2 #6** (22w): `"and then he made a bunch of different kind of things like a car a sheep or a flowers something like that"` — Single clause with enumerated object. One verb ("made").

3. **044_ST2 #4** (29w): `"and with so many other shapes he made a boat a car a whale sailing boat a butterfly a bird a cab a shoe and so many other shapes"` — Single clause with 8+ enumerated items. One verb ("made").

**Impact**: 3/2510 clauses (0.12%). All are genuine single clauses with long enumerations — no way to split per Vercellotti (one verb, one complement).

### 2-Word Clauses (~4% of all clauses)

Most are legitimate per Vercellotti:
- **Matrix/stance clauses** before ccomp: "he thought", "he decided", "he found" — valid minor clauses
- **Nonfinite with complement**: "wearing hat", "named jim", "watching him" — valid per §5-6
- **Copular**: "it's pretty", "it's funny" — valid

A small number (~2) are nonfinite without complement ("to try", "to express") that should ideally merge into adjacent clauses. These are rare edge cases.

### L2 Speech Artifacts (not bugs)

- **Verb morphology errors**: L2 speakers produce forms like "sore" (for "saw"), "look" (for "looked"), "blewing" (for "blowing"). These are correctly treated as verbs per Vercellotti §Coding complications.
- **Trailing incomplete clauses**: Speakers sometimes get cut off at the end of recording (e.g., `"and who created"`, `"reach john and"`). These are inherent to speech data.
- **Messy L2 text**: Very low-proficiency speakers produce difficult-to-parse text (e.g., `"cuz has a mouse in the little ball the little bubble and the elephant"`). The segmenter handles these as best as the spaCy parser allows.

### Disfluency Model Limitations

- **Content-level self-corrections** (e.g., "to a tree and a [unk] on a very small tree") are sometimes not fully caught by the disfluency model. The model detects word-level disfluencies (fillers, repetitions, false starts) but semantic self-corrections require discourse-level understanding.

### Not Issues (confirmed correct per Vercellotti)

- **Long clauses (16-20 words)**: 16 instances across 70 files. Mix of enumerations, temporal adjuncts, and messy L2 speech with content-level self-corrections the disfluency model cannot catch. Vercellotti does not set a maximum clause length.
- **"I think" / stance verbs as separate clauses**: Per Vercellotti §13-14, these are valid clauses, optionally coded as "minor" for separate analysis.
- **Nonfinite clauses with complement**: Per Vercellotti §5-6 and Foster et al. (2000), nonfinite verbs with a complement/adjunct are clauses.
- **Coordinated VPs with complement**: Per Vercellotti §1-4, coordinated verb phrases with own complement/adjunct are separate clauses.
