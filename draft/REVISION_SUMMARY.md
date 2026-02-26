# Revision Summary

Date: 2026-02-26
Based on: `review_peer_review.md` and `review_critical_thinking.md`

---

## Changes Made

### A. Fix overclaiming about cross-lingual transportability

- [x] **A.1 Title (line ~20):** Changed "Cross-Lingual Concurrent Validity" to "Preliminary Evidence for Cross-Lingual Transportability." Rationale: The title previously implied symmetric cross-lingual validation, but Japanese RQ1/RQ2 are pending.

- [x] **A.2 Abstract (line ~32):** Rewrote to (a) explicitly list all five pipeline stages (ASR, filler-augmented MFA, disfluency detection, clause segmentation, CAF computation), (b) clarify that English has full staged validation while Japanese currently has RQ3 only, (c) note that cross-lingual component-level validation is partial, and (d) differentiate English ("supported by component-level agreement") from Japanese ("component-level confirmation is pending"). Also changed "point estimates are higher" to "point estimates are numerically higher, though cross-study differences ... preclude direct comparison."

- [x] **A.3 Discussion Section 4.4 (cross-lingual transportability):** Changed "supporting cross-lingual transportability" to "providing preliminary evidence for cross-lingual transportability." Added a qualifying paragraph noting that the cross-lingual claim rests on measure-level evidence only for Japanese, and that error cancellation cannot be excluded until Japanese RQ1/RQ2 are complete.

- [x] **A.4 Conclusion (line ~514):** Added "preliminary" qualification to cross-lingual claims. Changed "values above previously reported" to "point estimates numerically above ... descriptive only and cannot establish relative superiority." Added sentence that English and Japanese differ in validation completeness, and the cross-lingual argument will be complete only when Japanese component-level analyses confirm the English pattern.

### B. Address LLM gold standard circularity concern

- [x] **B.5 Gold standard construction section (Section 2.1.3):** Added sentence acknowledging that LLM-assisted annotation for 30 of 40 files may introduce shared systematic biases, and that kappa=.816 should be interpreted as an upper-bound estimate relative to a fully independent human gold standard.

- [x] **B.6 Limitations section (Limitation 3):** Expanded to acknowledge (a) single-author review introduces anchoring bias risk, (b) inter-rater reliability may be inflated relative to independent human gold standard, and (c) English and Japanese gold standards differ in construction method (LLM-assisted vs. independent human annotation), limiting cross-lingual annotation quality comparisons.

### C. Add missing statistical reporting

- [x] **C.7 English RQ3 results text:** Added 95% CI for mean r = .936 via Fisher Z-transformation: [.878, .965]. Added note that confidence intervals for individual measures and ICC values are not reported but should be considered, particularly for task-level cells.

- [x] **C.7 Japanese RQ3 results text:** Added 95% CI for mean r = .953: [.916, .975]. Added qualification that without Japanese RQ1/RQ2, error cancellation contributing to high correlations cannot be excluded.

- [x] **C.7 English RQ3 task-level text:** Added illustrative 95% CI for MCPD ST1 r = .713: approximately [.380, .882]. Added caution about wide intervals at N=19-20 for both correlations and ICC values.

### D. Strengthen hedging on cross-study comparisons

- [x] **D.9 Table 7 caption:** Expanded from single-sentence caveat to explicit statement that "These values are not directly comparable" with enumeration of at least five dimensions of non-equivalence. Changed characterization from "interpreted with caution" to "treated as indicative context rather than evidence of relative system quality."

- [x] **D.10 Comparison text (Section 3.6):** Added "While keeping in mind that cross-study comparison is constrained by methodological differences" framing. Added explicit statement about five dimensions of non-equivalence. Changed conclusion from "corpus, population, and gold-standard differences preclude strong causal claims" to stronger statement that differences "could equally reflect differences in data difficulty or gold-standard construction."

- [x] **D.10 Discussion Section 4.1:** Added "numerically" before "higher" in both comparison sentences. Added "While direct comparison is limited by methodological differences" qualifier.

### E. Address MCPD fragility

- [x] **E.11 Discussion Section 4.3:** Expanded MCPD explanation to (a) explicitly note mid-clause pauses are relatively rare events (sparse counts), (b) hedge "makes this measure inherently more vulnerable" to "may make," (c) acknowledge that per-file MCP count distribution is not reported, (d) note alternative explanation (systematic timing errors) cannot be excluded, and (e) identify quantifying the MCP count--agreement relationship as a future direction.

### F. Add missing limitations

- [x] **F.12a Limitation 3 (expanded):** Added differential gold-standard methods between English (LLM-assisted) and Japanese (independent human annotation) as a limitation for cross-lingual comparisons.

- [x] **F.12b New Limitation 8:** Learner demographic information (L1 distribution, proficiency levels, age, recording conditions) not reported, limiting generalizability assessment.

- [x] **F.12c New Limitation 9:** MFA acoustic model coverage of L2-accented speech is unvalidated; forced alignment errors may propagate to pause duration estimates.

- [x] **F.12d New Limitation 10:** WER not stratified by proficiency level; relationship between ASR quality and CAF measurement validity at different proficiency levels remains unquantified.

- [x] **Limitation count:** Updated from "Seven limitations" to "Ten limitations."

### G. Improve abstract clarity

- [x] **G.13 (combined with A.2):** Abstract now explicitly states all five pipeline stages, differentiates English (full staged validation) from Japanese (RQ3 only), and includes CIs for mean r values.

---

## Review items NOT addressed (with rationale)

1. **Power analysis recalibration (Peer Review Major 4):** The revision instructions did not request changes to the power analysis section. The reviewers suggest recalibrating to r >= .80 or .90 threshold and addressing clustering in RQ1/RQ2. This requires substantive analytical work beyond text revision.

2. **Disfluency detector evaluation on L2 speech (Peer Review Minor 4):** Not included in revision instructions. Would require new experimental evaluation.

3. **Filler classifier cross-lingual applicability (Peer Review Minor 5):** Not included in revision instructions.

4. **150 ms ECP classification window justification (Peer Review Minor 6):** Not included in revision instructions.

5. **WER reference definition (Peer Review Minor 7):** Not included in revision instructions.

6. **Macro statistics for RQ2 (Peer Review Minor 8):** Not included in revision instructions.

7. **CAF vs. UF terminological inconsistency (Peer Review Minor 2):** Not included in revision instructions.

8. **Repair fluency measures absence (Peer Review Minor 12):** Not included in revision instructions.

9. **Section 4.5 heading tone (Peer Review Minor 11):** Not included in revision instructions.

10. **Pearson-Spearman divergence discussion (Critical Thinking 4.2):** Not included in revision instructions.
