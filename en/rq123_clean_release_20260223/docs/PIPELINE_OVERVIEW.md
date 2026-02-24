# Pipeline Overview — Automatic CAF Measure Extraction from L2 Speech

**Date:** 2025-02-14

This document describes the two core scripts of the automatic CAF pipeline and the design decisions behind each component.

---

## Architecture

```
Audio (.wav)
    │
    ▼
┌─────────────────────────────────────────────┐
│  asr_qwen3_mfa_en.py                       │
│                                             │
│  Step 1: Qwen3-ASR-1.7B transcription      │
│          (with disfluency-aware prompting)  │
│                     ▼                       │
│  Step 2: Filler-augmented MFA alignment     │
│          (beam=100, english_us_arpa)        │
│                     ▼                       │
│  Output: TextGrid (words + phones tiers)    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  textgrid_caf_segmenter_v3.py               │
│                                             │
│  Step 1: Neural disfluency detection        │
│          (xlm-roberta-base, custom-trained) │
│                     ▼                       │
│  Step 2: Clause segmentation                │
│          (spaCy + Vercellotti & Hall 2024)  │
│                     ▼                       │
│  Output: TextGrid (+ clauses tier)          │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  caf_calculator.py                          │
│                                             │
│  9 CAF measures: AR, SR, MLR,               │
│  MCPR, ECPR, PR, MCPD, ECPD, MPD           │
└─────────────────────────────────────────────┘
```

---

## 1. `asr_qwen3_mfa_en.py` — ASR + Forced Alignment

### 1.1 Why Qwen3-ASR?

Automatic speech recognition is the first bottleneck in any fully automatic CAF pipeline. We chose **Qwen3-ASR-1.7B** (Alibaba, 2025) because:

- **State-of-the-art open-source ASR.** At the time of development, Qwen3-ASR achieved competitive word error rates across multiple benchmarks while being fully open-weight and reproducible.
- **Integrated forced aligner.** Qwen3 ships with a companion model (`Qwen3-ForcedAligner-0.6B`) that produces word-level timestamps directly from the ASR output, providing rough initial alignments without an external tool.
- **Disfluency-aware prompting.** The model accepts a `context` parameter. We supply the prompt: *"Transcribe ALL disfluencies exactly as spoken. Include every filled pause, hesitation, repetition, false start, and self-correction."* This encourages the ASR to retain fillers (`uh`, `um`) and repetitions rather than "cleaning" the transcript, which is critical for accurate fluency measurement.

### 1.2 Why MFA on top of Qwen3?

The Qwen3 forced aligner produces usable timestamps, but Montreal Forced Aligner (MFA; McAuliffe et al., 2017) with the `english_us_arpa` acoustic model provides:

- **Phone-level alignment.** MFA outputs both word and phone tiers, enabling syllable counting from vowel nuclei (ARPAbet) rather than text-based estimation.
- **Higher timing precision.** MFA uses a full HMM-GMM pipeline trained on English speech, producing tighter word boundaries than the neural aligner's rough estimates.
- **L2 speech robustness.** With `--beam 100 --retry_beam 400`, MFA avoids alignment drift on long files and heavily accented speech. Default beam settings caused trailing-silence artifacts on ~20 files; the high beam eliminated all such cases.

### 1.3 Filler-augmented alignment

A known problem with forced alignment on L2 speech is that silent pauses between words do not appear in the transcript. If the aligner is forced to map a word sequence onto audio that contains long pauses, it may stretch word boundaries to cover the silence, degrading timing accuracy.

Our solution: **filler injection.** Before running MFA, the script scans for gaps between consecutive ASR words. For any gap ≥ 400 ms, it inserts placeholder filler tokens (`uh`) into the transcript proportionally to gap duration (up to 3 per gap). MFA then aligns these fillers to the silent intervals, anchoring the surrounding real words more tightly. After alignment, the filler tokens are discarded and only the real word timestamps are kept.

| Metric | Value |
|:--|--:|
| Files processed | 190 |
| Total words | 36,011 |
| MFA-aligned words | 35,690 (99.1%) |
| Rough fallback words | 321 (0.9%) |
| Fillers injected | 10,796 |
| ASR gap outliers (>5s, >10 words) | 0 |

### 1.4 Pipeline flow (per file)

1. **Qwen3-ASR** transcribes audio → rough word-level timestamps.
2. **Filler injection** adds placeholder tokens into silent gaps.
3. **MFA** (beam=100) aligns the augmented transcript to audio → precise word + phone timestamps.
4. **Map-back** discards filler tokens, assigns MFA timestamps to original words. Words not matched by MFA fall back to rough Qwen3 timestamps.
5. **Output:** TextGrid with `words` and `phones` tiers; clean TextGrid (fillers removed); JSON with per-word metadata.

---

## 2. `textgrid_caf_segmenter_v3.py` — Clause Segmentation

### 2.1 Clause segmentation rules

Clause boundaries are detected using a custom spaCy pipeline implementing the rules from **Vercellotti & Hall (2024)**. The spaCy model is `en_ud_L1L2e_combined_trf`, a transformer-based model trained on combined L1 and L2 English UD treebanks. The segmenter handles:

- Coordinated VPs (`conj`) as separate clauses
- Parataxis as separate clauses
- Copula constructions (with and without overt copula)
- Existential constructions
- Stance verbs (`think`, `believe`, etc.) as minor clauses when they take a `ccomp`
- Fragment recovery and verbless clause merging

These rules are **identical** across v1, v2, and v3 of the segmenter. What changed in v3 is the preprocessing step.

### 2.2 Neural disfluency detection (v3 key innovation)

Before clause segmentation, disfluent words must be identified and removed. Fillers (`uh`, `um`), repetitions (`the the`), self-corrections, and false starts can cause the dependency parser to produce incorrect clause structures if left in the input.

**Previous approach (v1/v2):** Rule-based regex matching for fillers and a sliding-window heuristic for repetitions. This was brittle — it missed multi-word repetitions, self-corrections, and discourse markers, and could not generalize across languages.

**Current approach (v3):** A fine-tuned **xlm-roberta-base** model that labels each word as fluent (0) or disfluent (1). This model was **custom-trained** for this project, following the methodology of **Kundu, Jyothi, & Bhattacharyya (2022)**, "Zero-shot Disfluency Detection for Indian Languages" (COLING 2022).

#### Training approach (following Kundu et al., 2022)

Kundu et al. demonstrated that disfluency detection can be transferred cross-lingually by training on real annotated English data (Switchboard) combined with synthetic disfluent data in the target language. Their key insight: a multilingual transformer (XLM-RoBERTa) trained on mixed real + synthetic data generalizes to new languages without any real annotated data in those languages.

We adapted this approach for **English and Japanese L2 speech**:

1. **Real English data:** 48,540 sentences from the Switchboard Dialog Act Corpus (Godfrey et al., 1992; Potts, 2012), with disfluency annotations parsed into word-level binary labels.

2. **Synthetic data:** 40,000 sentences (20K English from NLTK Brown+Gutenberg, 20K Japanese from Tatoeba). Seven L2 disfluency types were injected via rule-based scripts:
   - Filled pauses (fillers)
   - Word repetitions
   - Phrase repetitions
   - Self-corrections
   - False starts
   - Partial words
   - L1-transfer errors (language-specific)

3. **Label generation:** Word-level binary labels (0/1) were created by comparing disfluent and fluent sentence pairs using Kundu et al.'s subsequence alignment method.

4. **Model:** `xlm-roberta-base` fine-tuned on the combined 88,540 sentences for token classification (2 labels: fluent/disfluent).

#### Evaluation results

| Evaluation set | Precision | Recall | F1 |
|:--|--:|--:|--:|
| Synthetic EN test (2K held-out) | .984 | .989 | .987 |
| Synthetic JA test (2K held-out) | .995 | .999 | .997 |
| DisfluencySpeech (240 real EN utterances) | .882 | .619 | .727 |

| Comparison | F1 |
|:--|--:|
| Kundu et al. (2022) — Hindi (real eval) | .848 |
| Kundu et al. (2022) — Marathi (real eval) | .773 |
| Our model — DisfluencySpeech (real EN eval) | .727 |
| Our model — synthetic EN | .987 |
| Our model — synthetic JA | .997 |

The lower F1 on DisfluencySpeech compared to Kundu et al.'s Hindi/Marathi results is expected: DisfluencySpeech uses different annotation conventions (punctuation-attached disfluency markers) and contains native English discourse markers (`well`, `you know`) that are ambiguous between fluent and disfluent use. Direct comparison across evaluation sets is not straightforward.

**Critically, the model works for both English and Japanese**, enabling the same pipeline to be applied to Japanese L2 data with minimal modification (only the ASR and spaCy models change; the disfluency detector is shared).

#### Why this matters for CAF

The v2 → v3 upgrade (rule-based → neural disfluency detection) directly affects clause segmentation quality:
- Catches multi-word repetitions and self-corrections that regex missed
- Handles discourse markers appropriately
- Generalizes across L1 backgrounds (important for our multi-L1 corpus)
- Enables cross-linguistic extension (same model for English and Japanese pipelines)

### 2.3 Post-processing

After clause segmentation, two additional steps are applied:

1. **Sentence boundary enforcement.** Pre-computed sentence boundaries (from `wtpsplit`) are used to split any clause that spans multiple sentences. This catches cases where the dependency parser fails to separate run-on structures.

2. **Word-index tracking.** The segmenter tracks which original TextGrid word indices belong to each clause, enabling precise clause time boundaries from word-level timing rather than character offsets.

### 2.4 Output

The segmenter adds a `clauses` tier to the TextGrid, where each interval contains the clause text. Disfluent words are excluded from clause text but their time intervals remain in the `words` tier (they are not deleted from the TextGrid). This preserves the full timing information for pause detection.

---

## 3. `caf_calculator.py` — CAF Measure Computation

Reads the clause-annotated TextGrid and computes 9 CAF measures:

| Category | Measure | Definition |
|:--|:--|:--|
| **Speed** | AR | Articulation rate: syllables / phonation time |
| | SR | Speech rate: syllables / total duration |
| **Breakdown** | MCPR | Mid-clause pause ratio: mid-clause pauses / syllables |
| | ECPR | End-clause pause ratio: end-clause pauses / syllables |
| | PR | Pause ratio: all pauses / syllables |
| | MCPD | Mid-clause pause duration: mean (s) |
| | ECPD | End-clause pause duration: mean (s) |
| | MPD | Mean pause duration (s) |
| **Composite** | MLR | Mean length of run: mean syllables between pauses |

- **Pause threshold:** ≥ 250 ms (configurable)
- **Pause classification:** A pause is ECP if its onset is within 150 ms of a clause offset; otherwise MCP. Pauses not within any clause default to ECP.
- **Syllable counting:** From phone tier vowel nuclei (ARPAbet); text-based fallback when phones unavailable.

---

## 4. Gold Standard Construction — LLM-Assisted Clause Boundary Annotation

To validate the automatic clause segmentation (RQ1) and downstream pause classification (RQ2), a gold standard of human-validated clause boundaries was created using an LLM-assisted annotation pipeline adapted from Morin & Marttinen Larsson (2025).

### 4.1 Sampling

40 files (20 ST1 + 20 ST2) were selected via stratified random sampling across L1 groups and tasks. Files with known quality issues were excluded before sampling:
- HIGH_WER (18 files): Word error rate > 30%
- PREAMBLE (10 files): Manual transcription starts > 5s after ASR
- ASR_GAPS (4 files): Significant ASR alignment gaps > 3s

Selection details: `annotation/selected_files.json` (seed=42).

### 4.2 Rule set

Clause boundary rules were frozen before annotation began (`annotation/FROZEN_RULES.md`), based on Vercellotti & Hall (2024) plus pipeline-specific conventions. 8 rules covering: finite clause boundaries, coordinated VPs, parataxis, copula constructions, existential constructions, stance verbs, fragment handling, and verbless clause merging.

### 4.3 Blind human annotation (10 files)

Two trained coders independently annotated 10 files (boundary-only task):
- **5 pretraining files** — used as few-shot examples for LLM training
- **5 locked test files** — held out for LLM evaluation

| Set | Files |
|:--|:--|
| Pretraining (5) | ALL_001_F_GER_ENG_ST1, ALL_014_M_TUR_ENG_ST2, ALL_044_F_FRA_ENG_ST2, ALL_075_F_CCT_ENG_ST1, ALL_113_M_JPN_ENG_ST1 |
| Locked test (5) | ALL_003_F_RUS_ENG_ST2, ALL_029_M_TUR_ENG_ST1, ALL_034_M_HIN_ENG_ST2, ALL_096_F_KOR_ENG_ST2, ALL_139_M_PBR_ENG_ST1 |

After independent coding, disagreements were adjudicated to produce the gold reference. Inter-rater reliability and adjudication outputs are in `annotation/boundary_agreement_260213/`.

### 4.4 LLM annotation (Claude)

The LLM (Claude/Cascade) was given:
- The frozen rule set
- 5 gold boundary examples from adjudicated pretraining files (few-shot)
- Input: plain transcript text → Output: one clause per line

**Locked test evaluation** (5 files):
- Micro F1 = .929, κ = .914 against human-adjudicated gold
- Evaluation script: `annotation/boundary_agreement_260213/scripts/evaluate_boundary_predictions.py`

**Production** (30 remaining files):
- LLM segmented all 30 files; expert reviewed and accepted
- Output: `annotation/llm_output/production_30/`
- Token validation: all 35 files pass (word count match + sorted word-list diff)

### 4.5 Final gold standard

| Source | Files | Method |
|:--|--:|:--|
| Adjudicated blind | 10 | Two human coders + adjudication |
| LLM production (expert-reviewed) | 30 | Claude few-shot + expert review |
| **Total gold** | **40** | |

Gold segmentations stored in:
- Adjudicated: `annotation/boundary_agreement_260213/final_correct_segments/`
- Production: `annotation/llm_output/production_30/`
- Canonical transcripts: `annotation/transcripts/` (40 files)

### 4.6 Key files

| File | Location |
|:--|:--|
| Selection | `annotation/selected_files.json` |
| Frozen rules | `annotation/FROZEN_RULES.md` |
| Transcripts | `annotation/transcripts/` |
| Blind annotations | `annotation/blind/coder1/`, `annotation/blind/coder2/` |
| Adjudicated gold | `annotation/boundary_agreement_260213/final_correct_segments/` |
| LLM locked test predictions | `annotation/boundary_agreement_260213/your_predictions/locked_test/` |
| LLM production output | `annotation/llm_output/production_30/` |
| Evaluation scripts | `annotation/boundary_agreement_260213/scripts/` |
| Progress log | `annotation/PROGRESS.md` |
| Full procedure | `docs/LLM_CLAUSE_ANNOTATION_MANUAL.md` |

---

## 5. Scripts and models

| Component | Path | Source |
|:--|:--|:--|
| ASR + alignment | `asr_qwen3_mfa_en.py` | Qwen3-ASR-1.7B + MFA english_us_arpa |
| Clause segmenter | `scripts/textgrid_caf_segmenter_v3.py` | spaCy `en_ud_L1L2e_combined_trf` |
| Disfluency model | `disfluency_test/l2_disfluency_detector/model_v2/final/` | Custom xlm-roberta-base (this project) |
| CAF calculator | `scripts/caf_calculator.py` | — |
| Boundary evaluation | `annotation/boundary_agreement_260213/scripts/evaluate_boundary_predictions.py` | — |

---

## 6. References

- Kundu, R., Jyothi, P., & Bhattacharyya, P. (2022). Zero-shot disfluency detection for Indian languages. *Proceedings of COLING 2022*, 4442–4454.
- McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., & Sonderegger, M. (2017). Montreal Forced Aligner: Trainable text-speech alignment using Kaldi. *Proceedings of Interspeech*, 498–502.
- Vercellotti, M. L., & Hall, C. J. (2024). Clausal segmentation of L2 English speech. *Language Learning*.
- Godfrey, J., Holliman, E., & McDaniel, J. (1992). SWITCHBOARD: Telephone speech corpus for research and development. *Proceedings of ICASSP*, 517–520.
- Lea, R., et al. (2024). DisfluencySpeech. HuggingFace datasets.
