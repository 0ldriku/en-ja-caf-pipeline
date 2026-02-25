# Progress Log

## Status: Complete

## Build Timeline

1. **Source data** — Downloaded 50K EN sentences (NLTK Brown+Gutenberg) + 50K JA sentences (Tatoeba, MeCab-tokenized)
2. **Disfluency injection** — Wrote rule-based scripts injecting 7 L2 disfluency types per language (fillers, repetitions, self-corrections, false starts, partial words, L1-transfer errors)
3. **Label generation** — Created word-level binary labels (0=fluent, 1=disfluent) by comparing disfluent/fluent sentence pairs using Kundu's subsequence alignment method
4. **Model v1 (synthetic only)** — Fine-tuned `xlm-roberta-base` on 40K synthetic EN+JA sentences. Excellent on synthetic test (F1=0.989), but poor on real English disfluencies (DisfluencySpeech F1=0.54) due to missing discourse markers
5. **Error analysis** — DisfluencySpeech evaluation revealed the model missed native English discourse markers (`well`, `you know`, `I mean`) because these weren't in the synthetic training data
6. **Switchboard addition** — Downloaded freely available Switchboard Dialog Act corpus transcriptions (swda, CC BY-NC-SA 3.0), parsed disfluency annotations to extract 48.5K labeled real English sentences
7. **Model v2 (Switchboard + synthetic)** — Retrained on 88.5K combined sentences. DisfluencySpeech F1 improved from 0.54 to 0.73 (+35%), while synthetic performance stayed near-perfect

## Evaluation Results

### Model v2 (recommended) — Switchboard + synthetic EN+JA

| Evaluation Set | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| **Synthetic EN test** (held-out) | 0.984 | 0.989 | 0.987 | 2K fresh sentences, seed=9999 |
| **Synthetic JA test** (held-out) | 0.995 | 0.999 | 0.997 | 2K fresh sentences, seed=9999 |
| **Synthetic combined test** | 0.984 | 0.987 | 0.985 | 1.7K from train/valid split |
| **DisfluencySpeech** (real EN) | 0.882 | 0.619 | 0.727 | 240 real Switchboard-derived utterances |

### Model v1 (synthetic only) — for comparison

| Evaluation Set | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| Synthetic EN test | 0.980 | 0.999 | 0.990 | |
| Synthetic JA test | 0.990 | 1.000 | 0.996 | |
| Synthetic combined test | 0.982 | 0.996 | 0.989 | |
| DisfluencySpeech (real EN) | 0.786 | 0.432 | 0.540 | Missing discourse markers |

### Comparison to Kundu et al. (2022)

Kundu trained on real English Switchboard + synthetic Indian language data. Our v2 follows the same approach (Switchboard + synthetic target language).

| | Kundu (Hindi, real) | Kundu (Marathi, real) | Ours (DisfluencySpeech) | Ours (synthetic EN) | Ours (synthetic JA) |
|---|---|---|---|---|---|
| **F1** | 0.848 | 0.773 | 0.727 | 0.987 | 0.997 |
| **Training** | Switchboard + syn Hindi | Switchboard + syn Marathi | Switchboard + syn EN+JA | same | same |

Note: Kundu's real eval data was 150-300 hand-annotated sentences from YouTube interviews. DisfluencySpeech is 240 re-recorded Switchboard utterances with different annotation conventions, making direct comparison difficult.

### DisfluencySpeech Error Analysis (model v2)

**Most common missed disfluencies (false negatives):**
- `well,` — discourse marker with trailing comma; model often misses
- `you know,` — multi-word discourse marker
- `uh,` / `um,` — sometimes missed when followed by comma

**Most common false positives:**
- Rare; precision is high (0.88)

**Key insight:** The remaining errors are mostly punctuation-related. DisfluencySpeech keeps commas attached to disfluent words (e.g., `well,` not `well`), which can confuse the model since Switchboard training data had punctuation removed.

## Real L2 Transcript Tests (model v1, qualitative)

### English L2 — Repetition-heavy
```
Input:  I think I think the the boy is is um running to the to the park and and he he fell down
Output: I think the boy is running to the park and he fell down
```
Perfect — all repetitions and filler caught.

### English L2 — Full transcript (124 words)
- Flagged 9/124 words (7.3%)
- Caught: `said said`, `could you could you`, `put put huh put`, `took took`, `ha`, `ok`
- All correct, no false positives

### Japanese L2 — Real spoken transcript
```
Input:  えん 来たら あ 犬 犬 を ばず けっと をー 降ります
Output:     来たら    犬 を ばず けっと    降ります
```
Correctly flagged: `えん` (filler), `あ` (filler), first `犬` (repetition), `をー` (elongation).

### Japanese L2 — Synthetic test
```
Input:  えーと 私 は あの 昨日 えー 学校 に に 行き いや 行っ て あの 友達 と と 話し ました
Output:      私 は    昨日    学校 に    行っ て    友達 と    話し ました
```
Perfect — all fillers, repetitions, self-corrections caught.

## Data Statistics

| Dataset | Sentences | Source |
|---|---|---|
| Synthetic EN train | 20,000 | NLTK Brown+Gutenberg + injection |
| Synthetic JA train | 20,000 | Tatoeba + injection |
| Switchboard train | 48,540 | swda (real annotated English) |
| **Combined train (v2)** | **88,540** | All above merged |
| Synthetic valid | 3,334 | EN+JA synthetic |
| Synthetic test | 1,666 | EN+JA synthetic |
| Switchboard valid | 4,451 | swda |
| Switchboard test | 3,763 | swda |

## Known Limitations

- **Binary labels only** — the model says "disfluent" but not what type (filler, repetition, etc.)
- **Punctuation sensitivity** — performance drops if input has attached punctuation (commas, periods); best to strip punctuation before feeding to the model
- **MeCab not needed at inference** — Japanese L2 transcripts are usually already word-segmented; just split on spaces/commas
- **L1-specific patterns** — synthetic data targets Japanese L1 → English L2 and English L1 → Japanese L2 transfer errors; other L1 backgrounds may have different disfluency patterns
- **Source sentence quality** — NLTK/Tatoeba sentences are written/edited text, not spoken language; using OpenSubtitles as source could produce more realistic synthetic disfluencies
