# Revision Summary v2 -- Paper Draft Revisions (2026-02-26)

**Manuscript:** "A Shared English--Japanese Pipeline for Automatic Clause-Based Fluency Annotation: Staged Validation and Preliminary Evidence for Cross-Lingual Transportability"

**Revision based on:** Comprehensive peer review (second round), following the first revision tracked in `REVISION_SUMMARY.md`.

---

## Summary of Changes

This revision addresses **6 major issues**, **7 moderate issues**, and **8 new issues** identified in the second-round review. All changes were made to `paper_draftv2.tex` (with `paper_draft.tex` retained as the pre-v2 baseline for diff tracking). The revision is primarily textual (additions, clarifications, justifications) with no changes to data, analyses, or results.

---

## Major Issues Addressed

### Major 1: CAF -> UF Terminology (Global Fix)

**Problem:** The paper used "CAF measures" and "CAF computation" throughout, but only computes fluency measures -- no complexity or accuracy measures. "CAF" is an established tripartite framework in SLA; using it when only F is measured is misleading.

**Changes:**
- Abstract: "CAF computation" -> "UF computation"
- RQ3 description: "CAF measures" -> "UF measures"
- Pipeline diagram: "CAF computation" -> "UF computation"
- Stage 5 heading: renamed from "CAF computation" to "Utterance fluency computation"
- Table 2 caption: "Nine CAF measures" -> "Nine utterance fluency measures"
- Section 2.3.3: "CAF measures" -> "UF measures"
- Section 2.2.3 ("Why explicit rules?"): "CAF measures" -> "fluency measures"
- Section 4.6: "clause-based CAF analyses" -> "clause-based UF analyses"
- Section 4.7: "CAF measures" -> "UF measures"
- Section 4.5 paragraph: "CAF measures" -> "fluency measures"
- Added footnote in Stage 5 explaining that codebase retains "CAF calculator" file names for backward compatibility
- Retained "CAF" only in the footnote referencing script file names

### Major 2: Repair Fluency (RF) Measures Absence

**Problem:** The Introduction motivates RF as a key UF dimension (citing Kormos 2006, Suzuki et al. 2021), and the pipeline includes a disfluency detector, but no RF measures are computed. This gap was never acknowledged.

**Changes:**
- Added paragraph in Stage 5 explaining that the current release computes only speed and breakdown fluency; RF measures are excluded because the disfluency detector has not been validated at the token level for RF quantification
- Added Table 2 caption note: "The current release computes speed and breakdown fluency only; repair fluency measures are not included (see Limitation 11)"
- Added Limitation 11 explicitly acknowledging the RF absence
- Updated limitation count from "Ten" to "Thirteen"

### Major 3: Disfluency Detector Validation Gap Strengthened

**Problem:** The disfluency detector was trained on Switchboard + synthetic data with no held-out evaluation on L2 speech reported. Limitation 6 was too vague.

**Changes:**
- Rewrote Limitation 6 to explicitly state: no P/R/F1 on L2 speech is reported; errors propagate to clause segmentation through two pathways (undetected disfluencies and false positives); Japanese post-rules are acknowledged but not systematically evaluated; quantifying disfluency detection accuracy on L2 speech is flagged as a priority for future validation.

### Major 4: Filler Classifier Cross-Lingual Applicability

**Problem:** The neural filler classifier was trained on English podcast speech (PodcastFillers) and applied to both languages, but cross-lingual generalization was never discussed.

**Changes:**
- Added two sentences in Stage 4 (Section 2.2.4) noting that the same English-trained model is applied to both languages and acknowledging the phonetic differences between English and Japanese fillers
- Added Limitation 13 explicitly flagging this as an unverified assumption and noting the potential for systematic bias in Japanese pause-duration measures

### Major 5: 150 ms ECP Window Justified

**Problem:** The 150 ms ECP classification threshold appeared without citation, justification, or sensitivity analysis.

**Changes:**
- Added justification in Stage 5: the 150 ms window was chosen to accommodate MFA alignment uncertainty (50-100 ms), with a cross-reference to the new Limitation 12
- Added Limitation 12 acknowledging that no formal sensitivity analysis has been conducted and that alternative thresholds could reclassify pauses

### Major 6: Power Analysis Reframed

**Problem:** (a) RQ1/RQ2 power calculations ignored clustering within speakers; (b) RQ3 used r >= .60 field-norm threshold when a validity-appropriate threshold should be higher.

**Changes:**
- Added sentence acknowledging that boundary positions and pause events are nested within speakers; effective sample size is closer to N = 40 than to N = 1,131
- Reframed RQ3: for concurrent validity, the practically relevant threshold is r >= .80 (minimum N ~ 8); retained r >= .60 field-norm threshold as secondary reference

---

## Moderate Issues Addressed

### Moderate 1: Japanese Corpus Description Expanded

**Change:** Expanded Section 2.1.2 to note university collection context, diverse L1 backgrounds, and that demographic details are restricted for privacy. Also noted that gold annotations were produced by trained human coders following the Vercellotti-adapted framework.

### Moderate 2: Quality-Filtered Exclusion Criterion Specified

**Change:** Replaced vague "manual preamble mismatch" with specific explanation: "the manual annotation included task-instruction preamble speech that was absent from the ASR-segmented output, causing a systematic alignment offset that would distort UF computation."

### Moderate 3: WER "Canonical" Clarified

**Change:** Replaced "canonical" with "the manual transcript (human-transcribed words from ALLSSTAR)."

### Moderate 4: Macro Statistics Added for RQ2

**Change:** Added per-file macro statistics to Section 3.2: "mean accuracy = .921 (SD = .052), median = .927, range .789--1.000. Six files achieved perfect pause-location accuracy."

### Moderate 5: Pearson-Spearman Divergences Discussed

**Change:** Added a new paragraph in Section 4.3 analyzing the divergence between Pearson r and Spearman rho for pause-duration measures (Japanese MPD gap = .072; MCPD gap = .053). Notes that this suggests outlier influence and systematic mean-level bias, reinforcing the characterization of these measures as most sensitive. Connects to ICC values being notably lower than Pearson r.

### Moderate 6: ICC Interpretation Citation Added

**Change:** Added citation to Koo & Li (2016) for the ICC interpretive framework (< .50 poor, .50-.75 moderate, .75-.90 good, > .90 excellent). Added reference entry to bibliography.

### Moderate 7: "Three Key Innovations" Heading Renamed

**Change:** Changed Section 4.5 heading from "Three key innovations" to "Technical design contributions" and softened the opening sentence.

---

## New Issues Addressed

### New 1: Pipeline Architecture Diagram Clarified

**Change:** Added clarifying sentence that disfluency detection (Stage 2) is embedded within the clause segmentation step (Stage 3). Added footnote noting the Japanese-specific span-blanking preprocessing step.

### New 2: Vercellotti Strictness Deviation Impact Noted

**Change:** Added parenthetical note to the coordinated VP bullet explaining that the stricter rule "produces fewer clause boundaries than a fully Vercellotti-inclusive implementation, which may slightly affect MLR and clause-count-dependent measures."

### New 3: 250 ms Pause Threshold Justified

**Change:** Added citation to de Jong & Bosker (2013) in Table 2 caption and in Stage 5 text.

### New 4: Filler Injection Formula and MFA Beam Justified

**Changes:**
- Added justification for filler injection parameters: "the gap threshold (350 ms) and filler interval (550 ms) were set based on typical L2 filler durations observed during development"
- Added justification for MFA beam settings: "set to their maximum practical values to ensure alignment convergence on disfluent L2 speech"

### New 5: Japanese nsubj Exclusion Impact Discussed

**Change:** Added paragraph in Section 4.4 noting the cross-lingual asymmetry in complement checking (nsubj excluded in Japanese due to pro-drop), its potential to produce fewer clause boundaries in Japanese, and the implication for cross-lingual comparability of clause-count-dependent measures.

### New 6: URL Placeholders

**Status:** `[URL_TO_BE_INSERTED]` placeholders remain at lines ~96 and ~542 for the author to populate before submission.

### New 7: Date Header Updated

**Change:** Changed from "Draft dated February 26, 2026" to "Revised draft, February 2026."

### New 8: Japanese Romanization

**Status:** Verified that "node" romanization in the paper correctly corresponds to ので in the code. No change needed.

---

## New References Added

- Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine*, 15(2), 155--163.

---

## New Limitations Added (Limitations 11-13)

- **Limitation 11:** RF measures not included in this release despite RF being motivated in the Introduction. The disfluency detector has not been validated at the token level for RF quantification.
- **Limitation 12:** The 150 ms ECP onset window has not been subjected to formal sensitivity analysis. Different thresholds could reclassify pauses.
- **Limitation 13:** The neural filler classifier was trained on English speech. Its cross-lingual application to Japanese has not been empirically verified; Japanese fillers differ phonetically and may be under-detected.

Total limitation count updated from 10 to 13.

---

## Items NOT Changed (Require New Data or Analyses)

The following items from the review would strengthen the paper but require analytical work beyond text revision:

1. **Disfluency detector P/R/F1 on L2 speech subset** -- Requires running the detector on manually annotated files and computing evaluation metrics. Flagged as future priority in Limitation 6.
2. **ECP threshold sensitivity analysis** -- Requires rerunning RQ2 with alternative thresholds (100, 200 ms) and comparing kappa stability. Flagged in Limitation 12.
3. **Per-file MCP count distribution analysis** -- Feasible from existing CSVs. Would quantify the relationship between MCP count and MCPD agreement. Mentioned in Section 4.3 as future direction.
4. **Filler classifier evaluation on Japanese** -- Requires labeled Japanese filler data. Flagged in Limitation 13.
5. **Filler coverage/accuracy analysis** -- Compare filler intervals in auto vs. manual TextGrids for the 40 gold files to quantify net filler detection coverage. Would directly address Limitation 13 by showing whether the multi-layer approach (ASR prompt + MFA injection + neural classifier) achieves adequate filler coverage for Japanese.
6. **URL insertion** -- Awaiting repository publication.

---

## Post-Review Author Corrections (v2.1)

### RF Measures (Limitation 11 + Stage 5 paragraph)

**Correction:** Reframed the reason RF measures are absent. The pipeline CAN compute RF; the actual reason is that the manual reference annotations lack disfluency-level labels, so concurrent validity for RF cannot be computed without creating new annotated data.

**Changes:**
- Stage 5 RF paragraph: rewritten to state the tool supports RF but manual data lacks disfluency annotations
- Limitation 11: rewritten to explain that RF validation is blocked by reference data, not tool capability

### Disfluency Detector Accuracy (Limitation 6)

**Addition:** Inserted actual evaluation metrics from `shared/disfluency_detector/PROGRESS.md` into Limitation 6: synthetic EN F1 = .987, synthetic JA F1 = .997, DisfluencySpeech (real L1 EN) F1 = .727 (P = .882, R = .619). Retained the caveat that no L2 evaluation exists.

### Filler Classifier Multi-Layer Fallback (Section 4.5 + Limitation 13)

**Correction:** Expanded "Dual filler handling" to "Multi-layer filler handling" to acknowledge three complementary layers: (1) ASR disfluency-preserving prompt, (2) MFA filler injection, (3) neural classifier. Noted that even if the classifier underperforms on Japanese, layers 1 and 2 provide complementary coverage.

**Changes:**
- Section 4.5 heading: "Dual filler handling" → "Multi-layer filler handling"
- Section 4.5 paragraph: expanded to describe three layers with explicit note about Japanese coverage
- Limitation 13: added sentence acknowledging mitigation from ASR prompt and MFA injection
- Conclusion paragraph: "dual filler handling" → "multi-layer filler handling"

### Clustering (no change needed)

**Finding:** Matsuura et al. (2025) used the same observation-level kappa approach (Donner & Eliasziw 1987 power calculations assuming independence) without acknowledging clustering. This is the field standard. The existing acknowledgment sentence in the paper is sufficient.

---

## File Changes Summary

| File | Status |
|------|--------|
| `draft/paper_draftv2.tex` | Revised (all changes above + v2.1 corrections) |
| `draft/paper_draft.tex` | Baseline retained for diff tracking |
| `draft/REVISION_SUMMARY_v2.md` | New (this file) |
