# Peer Review

**Manuscript:** "A Shared English--Japanese Pipeline for Automatic Clause-Based Fluency Annotation: Staged Validation and Cross-Lingual Concurrent Validity"

**Review date:** 2026-02-26

---

## 1. Summary Statement

This paper presents a shared English--Japanese NLP pipeline for automatic clause-based fluency annotation of L2 speech. The pipeline comprises five stages (ASR, filler-augmented forced alignment, disfluency detection, clause segmentation, and CAF computation) and is evaluated with a staged design addressing clause-boundary agreement (RQ1), pause-location agreement (RQ2), and concurrent validity of nine CAF measures (RQ3). English component-level results show almost perfect agreement for both clause boundaries (micro F1 = .845, kappa = .816) and pause-location classification (kappa = .840, accuracy = .921). Concurrent validity is strong for English (mean r = .936, N = 39) and Japanese (mean r = .953, N = 40). Three technical innovations are highlighted: filler-augmented forced alignment, dual filler handling, and zero-shot cross-lingual disfluency detection.

The paper has clear strengths. The staged evaluation design -- validating component-level agreement before measure-level validity -- is methodologically sound and addresses a legitimate gap in the L2 fluency automation literature, where end-to-end correlations are often reported without intermediate checks. The explicit operationalization of the Vercellotti & Hall (2024) clause framework in inspectable code is a meaningful contribution to reproducibility. The cross-lingual architecture with principled language-specific adaptations (e.g., te-form chain handling, mora-based normalization) is well-motivated. The writing is generally clear, the statistical reporting is thorough, and the limitations section is commendably transparent about the study's boundaries.

However, the manuscript has several significant weaknesses that require attention before publication. Most critically, the study is incomplete: Japanese RQ1 and RQ2 are explicitly pending, making the cross-lingual claims in the title asymmetric and partially unsupported. The English gold standard relies on an LLM-assisted workflow that introduces methodological concerns about circular validation. The absence of perceived fluency data limits construct validity claims. Additionally, some statistical reporting choices and interpretive claims require tightening.

**Recommendation:** Major revisions. The core pipeline and evaluation design are sound, but the incomplete Japanese validation, gold-standard concerns, and several analytical gaps must be addressed before the cross-lingual claims in the title and abstract are fully warranted.

---

## 2. Major Comments

### Major 1: Incomplete Japanese validation undermines the cross-lingual title claim

The title promises "A Shared English--Japanese Pipeline" with "Cross-Lingual Concurrent Validity," yet Japanese RQ1 (clause-boundary agreement) and RQ2 (pause-location agreement) are explicitly listed as "Pending" (Table 1, lines 87--88). The paper thus provides no component-level evidence that the Japanese pipeline produces valid intermediate outputs. The strong Japanese RQ3 correlations (mean r = .953) could theoretically arise even with moderate boundary agreement if errors cancel in aggregation -- precisely the concern the authors themselves raise when motivating the staged design (Section 4.2: "High measure-level correlations can arise even when intermediate boundary or pause-label agreement is moderate, if errors cancel in aggregation"). Without Japanese RQ1 and RQ2, this possibility cannot be ruled out for the Japanese side.

**Recommendation:** Either (a) complete Japanese RQ1 and RQ2 before submission, which would substantially strengthen the paper, or (b) retitle the paper to accurately reflect the asymmetric evidence (e.g., "An English--Japanese Pipeline... with Full English Validation and Cross-Lingual Concurrent Validity") and explicitly frame the Japanese contribution as partial. The current framing risks implying symmetric cross-lingual validation that does not yet exist.

### Major 2: LLM-assisted gold standard raises circularity and bias concerns

The English gold-standard construction (Section 2.1.3) uses an LLM (Claude) trained on 10 human-adjudicated files to annotate 30 production files, which were then "manually reviewed and accepted as gold by the first author." Several concerns arise:

(a) **Confirmation bias in review.** A single reviewer checking LLM-generated annotations is susceptible to anchoring: the reviewer sees the LLM's output and must decide whether to accept it, rather than producing annotations independently. This is a weaker validation than independent double-coding. The phrase "manually reviewed and accepted as gold" (line 122) does not specify how many boundaries were changed during review, what the review criteria were, or whether the reviewer was blind to the LLM's decisions.

(b) **Potential circularity.** The LLM-validated test performance (micro F1 = .929, kappa = .914 on 5 locked files) establishes that the LLM agrees with human annotators at a high level, but the 30 production files annotated by the LLM and accepted by one reviewer then become the gold standard against which the pipeline is evaluated. If the LLM and the pipeline share systematic biases (e.g., both relying on similar dependency-parse structures), the resulting agreement could be inflated.

(c) **Transparency of the workflow.** The paper cites Morin & Marttinen Larsson (2025) as precedent, but that study concerned grammatical annotation, not clause-boundary annotation for temporal fluency measures. The generalizability of the workflow to this domain should not be assumed.

**Recommendation:** (i) Report the number and proportion of boundaries changed during the first author's manual review of the 30 LLM-annotated files. (ii) Discuss the potential for shared bias between the LLM-assisted gold standard and the pipeline's dependency-parse-based clause segmenter. (iii) Consider computing inter-rater reliability between the LLM annotations and a second independent human coder on a subset of the 30 production files.

### Major 3: No confidence intervals reported for correlation coefficients

The paper reports Pearson r, Spearman rho, ICC(2,1), and MAE for all nine measures across both languages (Tables 3--6), but no confidence intervals are provided for any of these statistics. For sample sizes of N = 39 (English) and N = 40 (Japanese), 95% CIs around correlation coefficients can be substantial. For instance, the weakest English overall value (MCPD r = .821) would have a 95% CI of approximately [.676, .904], and the weakest task-level value (English ST1 MCPD r = .713, N = 19) would have a CI of approximately [.380, .882]. The lower bound of this interval falls below what most researchers would consider strong concurrent validity.

The claim that "all nine Pearson correlations are large (r > .60; Plonsky & Oswald, 2014)" is technically correct for the point estimates, but without CIs, the reader cannot assess whether the population values are reliably above the large-effect threshold.

**Recommendation:** Report 95% confidence intervals for all Pearson r and ICC values in Tables 3--6, or at minimum in the task-level tables where N is smallest. Qualify interpretive claims accordingly.

### Major 4: Power analysis appears to use incorrect benchmarks and conflates unit-of-analysis levels

The sample size considerations (Section 2.4.4) contain two issues:

(a) **RQ1/RQ2 power analysis conflates annotation units with independent observations.** The paper states that N = 1,131 boundary positions and N = 1,902 pause events exceed the minimum samples for kappa. However, these are not independent observations: boundaries and pauses are nested within speakers (40 files), and within-speaker observations are correlated. The effective sample size for kappa power calculations should account for clustering. Standard kappa power formulas (Donner & Eliasziw, 1987) assume independent observations.

(b) **RQ3 power analysis uses Plonsky & Oswald (2014) threshold (r >= .60) as the detectable effect.** The authors compute power to detect r >= .60, concluding that N = 19 is sufficient. However, Plonsky & Oswald's field-specific benchmarks (small = .25, medium = .40, large = .60) are descriptive norms for L2 research effect sizes, not validity thresholds. For concurrent validity of an instrument intended to replace manual annotation, one would typically require much higher correlations (e.g., r >= .80 or .90). The power analysis should be calibrated to the relevant practical threshold for validity, not a general effect-size benchmark.

**Recommendation:** (i) Acknowledge the clustering issue in RQ1/RQ2 power calculations and consider reporting design effects or cluster-adjusted effective sample sizes. (ii) Recalibrate the RQ3 power analysis to a concurrent-validity-appropriate threshold (e.g., r >= .80 or .90) and report whether the study is adequately powered at that level.

### Major 5: Comparison table (Table 7) does not adequately account for methodological differences

Table 7 compares the current pipeline against Chen & Yoon (2012) and Matsuura et al. (2025), and the text notes that "direct comparison should be interpreted with caution." However, the comparison is nonetheless presented prominently and discussed at length in Section 4.1. Several specific methodological differences make the comparison less informative than it appears:

(a) The current study uses a **different gold-standard methodology** (LLM-assisted vs. fully human-annotated in prior work).
(b) The current study uses a **different alignment method** (edit-distance vs. LCS in some prior work), which the authors note is "stricter" -- yet the reported agreement is higher, which could suggest the gold standard is more favorable rather than the pipeline being better.
(c) The **clause definition** differs (Vercellotti & Hall vs. implicit definitions in prior work), meaning the systems are segmenting for different targets.
(d) Matsuura et al. (2025) reported kappa across **proficiency subsets and modalities**, while the current paper reports an overall kappa that aggregates across all speakers.

The comparison risks leaving readers with the impression that the current system is demonstrably superior when the evidence supports only that the point estimates are higher under non-comparable conditions.

**Recommendation:** Either (i) substantially expand the caveats in Table 7's caption and surrounding text to make the non-comparability explicit for each row, or (ii) move the comparison to a discussion-only narrative without a table, which would reduce the visual impression of a direct benchmark comparison.

### Major 6: MCPD validity is weak in English ST1 but insufficiently addressed

English ST1 MCPD shows r = .713 and ICC = .669 (Table 4). An ICC of .669 falls in the "moderate" range and would be classified as "poor to moderate" agreement by Koo & Li (2016) standards. The discussion attributes this to "low frequency of MCPs in fluent speakers' speech" (Section 4.3), but this explanation is speculative without reporting the actual MCP count distribution. If many ST1 speakers have 0--2 MCPs, the MCPD measure is not meaningfully estimable for those speakers, and including them in the correlation inflates noise.

More importantly, the practical implication is under-discussed: if the pipeline cannot reliably estimate MCPD for picture-narrative tasks (or for fluent speakers more generally), this is a substantive limitation for researchers interested in mid-clause pausing, which is precisely the measure most strongly associated with perceived fluency ratings (Suzuki et al., 2021).

**Recommendation:** (i) Report MCP count distributions (median, range, number of files with fewer than 3 MCPs) for both tasks and both languages. (ii) Consider whether files with very few MCPs should be excluded from MCPD correlations or flagged as unreliable. (iii) Discuss the practical implications of MCPD instability for fluency research applications.

---

## 3. Minor Comments

### Minor 1: Abstract omits Japanese RQ1/RQ2 pending status

The abstract mentions "English component-level results and English/Japanese concurrent-validity results are reported, while Japanese component-level analyses (RQ1 and RQ2) are pending completion under the same protocol." This is appropriately transparent, but the abstract's opening claim -- "We developed a shared English--Japanese pipeline" -- front-loads the cross-lingual framing before the reader encounters the asymmetric evidence. Consider reordering to present the evidence status earlier.

### Minor 2: "CAF" measures vs. "UF" measures -- terminological inconsistency

The paper uses "CAF measures" (Complexity, Accuracy, Fluency) in the title of Table 2 and throughout the Methods/Results, but all nine measures are fluency measures (speed, breakdown, repair dimensions). The pipeline does not compute complexity or accuracy measures. The Discussion (Section 4.6) also refers to "CAF analyses." Using "CAF" when only fluency is measured may confuse readers. Consider using "utterance fluency measures" or "temporal fluency measures" consistently.

### Minor 3: Effect-size benchmarks applied asymmetrically

The paper applies Plonsky & Oswald (2014) benchmarks to interpret Pearson r values ("all nine Pearson correlations are large, r > .60") but does not apply comparable benchmarks to kappa values. The Landis & Koch (1977) scale is used for kappa interpretation, but these two scales have different origins and assumptions. This is not incorrect, but the paper should acknowledge that different interpretive frameworks are being applied to different statistics.

### Minor 4: Disfluency detector validation is insufficient

The disfluency detector is described (Section 2.2.2) with training details but no held-out evaluation results on L2 speech. The training data is Switchboard (conversational L1 English) plus synthetic data. The paper acknowledges this in Limitation 6 but does not report even basic metrics (precision, recall, F1) for disfluency detection on L2 English or L2 Japanese speech. Since the disfluency detector feeds directly into clause segmentation (pruning disfluent tokens), its error rate on L2 speech is relevant to interpreting downstream results.

**Recommendation:** Report disfluency detection performance on a subset of the evaluation data (e.g., against manual disfluency annotations on 10 files).

### Minor 5: Filler classifier threshold and cross-lingual applicability

The neural filler classifier (Section 2.2.4) was "trained on the PodcastFillers dataset" with English podcast data. It is applied to both English and Japanese speech. The paper does not discuss whether this English-trained acoustic model generalizes to Japanese fillers, which have different phonetic characteristics (e.g., "eto," "ano" vs. "uh," "um"). If the classifier primarily detects English-type fillers, Japanese filler speech intervals may still be over-counted as pauses.

**Recommendation:** Briefly discuss the cross-lingual applicability of the filler classifier and whether Japanese-specific filler detection was validated.

### Minor 6: 150 ms ECP classification window not justified

The ECP classification rule states: "a pause is labeled ECP if its onset falls within 150 ms of any clause offset" (Section 2.2.5). This 150 ms window is applied without justification or reference. The choice of this threshold directly affects MCP/ECP classification and all downstream pause-location metrics. Was this value empirically optimized, adopted from prior work, or chosen arbitrarily?

**Recommendation:** Justify the 150 ms window with either empirical evidence or a citation. Report sensitivity of RQ2 results to alternative thresholds (e.g., 100 ms, 200 ms).

### Minor 7: WER values reported without definition of the reference

Section 3.1 reports "Mean alignment WER between canonical and ASR clause text was .121 (12.1%)." The term "canonical" is not defined earlier. Is this the manual transcription? The ASR prompt? Clarify.

### Minor 8: Table formatting -- missing macro-level statistics in RQ2

RQ1 reports both micro and macro statistics (micro F1 = .845, macro F1 = .846), but RQ2 reports only micro-level kappa and accuracy. For consistency and to reveal per-file variability in pause-location agreement, macro-level statistics (mean and SD of per-file kappa and accuracy) should be reported for RQ2 as well.

### Minor 9: Japanese corpus description is vague

The Japanese corpus description (Section 2.1.2) states "40 speech files (20 ST1, 20 ST2) from an L2 Japanese speech dataset" without naming the corpus, specifying the L1 backgrounds of speakers, or describing proficiency levels. The English corpus is sourced from ALLSSTAR and described as including "diverse L1 backgrounds." The Japanese description should be comparably detailed.

### Minor 10: Missing reference -- Koo & Li (2016) ICC interpretation guidelines

The paper does not cite ICC interpretation guidelines. While the text states "all ICC values exceed .79, indicating good to excellent absolute agreement" (Section 3.3), the source of this interpretation framework is not cited. Koo & Li (2016) is the standard reference for ICC interpretation and should be cited.

### Minor 11: Section 4.5 heading -- "Three key innovations" reads as promotional

The heading "Three key innovations" in the Discussion is assertive for a Discussion section. Consider "Design contributions" or "Technical design choices" to maintain a more neutral tone appropriate for a journal article.

### Minor 12: Repair fluency (RF) measures are absent

The Introduction (Section 1.1) discusses repair fluency as one of the three dimensions of utterance fluency, but the pipeline computes no RF measures. The nine measures in Table 2 cover speed, composite, and breakdown dimensions only. The absence of RF is not discussed in the Limitations section. This should be acknowledged, especially since the disfluency detector could potentially support RF metrics.

### Minor 13: URL placeholders remain

Two instances of `[URL_TO_BE_INSERTED]` appear (lines 96 and 518). These should be populated before submission.

### Minor 14: "Quality-filtered cohort" exclusion criterion

One English file was excluded for RQ3 due to a "manual preamble mismatch" (Section 2.1.3). This exclusion reduces N from 40 to 39 but is not explained in detail. What constitutes a "preamble mismatch"? Could this issue affect other files at smaller scales? The exclusion criterion should be specified.

---

## 4. Statistical Reporting Check

### Coverage of the nine measures

All nine measures (AR, SR, MLR, MCPR, ECPR, PR, MCPD, ECPD, MPD) are reported with Pearson r, Spearman rho, ICC(2,1), and MAE in Tables 3--6. This is thorough and meets current best-practice standards.

### Confidence intervals

**Not reported.** No 95% CIs are provided for any correlation coefficient or ICC value. This is a significant omission for concurrent validity claims (see Major 3).

### Power analysis

**Partially correct but problematic.** The RQ3 power analysis correctly identifies N = 19 as sufficient for detecting r >= .60 at alpha = .05 and power = .80. However, the use of Plonsky & Oswald (2014) r >= .60 as the threshold for concurrent validity is questionable (see Major 4). The RQ1/RQ2 power analysis does not account for clustering of observations within speakers (see Major 4).

### Effect-size benchmarks

Plonsky & Oswald (2014) benchmarks are cited and applied correctly for the L2 research context (small = .25, medium = .40, large = .60). However, the appropriateness of these benchmarks for concurrent validity (as opposed to association between independent constructs) is debatable. Concurrent validity of an annotation tool typically demands higher thresholds than general field-specific effect-size norms.

### Kappa interpretation

Landis & Koch (1977) interpretation benchmarks are correctly cited and applied. The kappa values are appropriately categorized as "almost perfect" (> .80).

### ICC model specification

ICC(2,1) is correctly specified as a two-way random, single-measures, absolute-agreement model. This is the appropriate ICC variant for concurrent validity between two measurement methods.

### Missing statistical tests

- No formal comparison between ST1 and ST2 agreement levels (e.g., paired test or interaction term). All ST1/ST2 differences are described narratively.
- No formal comparison between English and Japanese concurrent validity. This is acknowledged in Limitation 7.
- No tests for significant difference from benchmark thresholds (e.g., whether r significantly exceeds .60 or .80).

---

## 5. Writing Quality

### Abstract

The abstract is well-structured and informative but long (approximately 200 words). It accurately conveys the key findings and appropriately flags the pending Japanese analyses. The opening sentence effectively motivates the work. One concern: the phrase "filler-augmented MFA" in the pipeline description is jargon that may not be accessible to all readers in an abstract.

### Introduction

The introduction is well-organized across four subsections that progressively narrow from the general construct (utterance fluency) to the specific gaps addressed. The literature coverage is current and relevant. The integration of Vercellotti & Hall (2024) as the clause framework is well-motivated. The three-gap structure (cross-lingual, staged evaluation, clause definition) maps cleanly onto the three RQs.

One weakness: the introduction does not adequately preview why Japanese was selected as the comparison language. A brief motivation for the English--Japanese pair (beyond listing typological contrasts) would strengthen the rationale.

### Methods

The Methods section is detailed and generally well-organized. The five-stage pipeline architecture is clearly described with appropriate technical detail. The gold-standard construction section is transparent about the LLM-assisted workflow, though it should be more detailed about the review process (see Major 2). The evaluation methods for each RQ are well-specified with appropriate metric choices. The sample size considerations section is a good addition, though the calculations need correction (see Major 4).

### Results

The Results section is clearly organized by RQ and presents data in well-formatted tables. The narrative interpretation is generally proportionate to the evidence. The comparison with prior systems (Table 7) is appropriately caveated, though not sufficiently (see Major 5). The pending Japanese RQ1/RQ2 section is honestly flagged.

### Discussion

The Discussion is structured logically and covers error propagation, cross-lingual patterns, and practical implications. Section 4.5 ("Three key innovations") is informative but tonally promotional for a Discussion (see Minor 11). The Future Directions subsection appropriately identifies model-based clause segmentation, external-corpus validation, and perceived fluency ratings as next steps.

### Limitations

The seven-item limitations section is commendably thorough and transparent. It addresses most of the paper's genuine weaknesses, including the pending Japanese analyses, moderate sample sizes, LLM gold-standard concerns, and absence of perceived fluency data. This is a strength of the paper.

### Conclusion

The conclusion effectively summarizes the main findings and contributions without over-stating. The final sentence appropriately identifies the immediate next step (completing Japanese RQ1--RQ2).

---

## 6. Reproducibility

### Pipeline description

The five-stage pipeline is described in sufficient detail for a researcher to understand the processing logic. Specific model names and versions are provided (Qwen3-ASR 1.7B, Qwen3 Forced Aligner 0.6B, MFA with english_us_arpa/japanese_mfa, XLM-RoBERTa-base, spaCy transformer, GiNZA ja_ginza_electra, wtpsplit). The filler-augmented alignment formula is explicitly stated (k = floor((gap - 0.35) / 0.55) + 1, capped at 3). Training parameters for the disfluency detector are reported (3 epochs, lr = 2e-5, batch size 16).

### Code and data availability

The paper states that "all pipeline scripts, trained models, gold annotations, and analysis outputs are publicly available" at a URL that is not yet inserted. If the release bundle is as comprehensive as described, this would be a strong contribution to reproducibility.

### Gaps in reproducibility

1. **Disfluency detector training data composition:** The paper mentions "88.5K training sentences total" from Switchboard and synthetic data, but does not specify the proportion of real vs. synthetic data or the synthetic data generation procedure beyond citing Kundu (2022).

2. **ASR prompt text:** The "disfluency-preserving prompt" for Qwen3-ASR is mentioned but not specified. Since prompt engineering can substantially affect ASR behavior, the exact prompt should be documented.

3. **Clause segmenter rule set:** While the clause-coding logic is described narratively, the full rule set (including edge cases and priority ordering) would need to be examined in the released code. The narrative description in Sections 2.2.3 is necessarily incomplete.

4. **MFA beam settings rationale:** The specific beam values (beam = 100, retry beam = 400) are reported but not justified. Were these empirically tuned?

5. **Gold-standard annotation guidelines:** The training protocol for the two human coders and the adjudication procedure for the 10 blind files are not described in detail.

---

## Summary of Key Issues

| Priority | Issue | Section |
|----------|-------|---------|
| Major | Incomplete Japanese RQ1/RQ2 undermines cross-lingual claims | Throughout |
| Major | LLM-assisted gold standard: circularity and bias risks | 2.1.3 |
| Major | No confidence intervals for correlation coefficients | 3.3--3.4, Tables 3--6 |
| Major | Power analysis uses inappropriate thresholds and ignores clustering | 2.4.4 |
| Major | Cross-study comparison overstated given non-comparable conditions | Table 7, 4.1 |
| Major | MCPD weakness insufficiently characterized | 3.3, 4.3 |
| Minor | CAF vs. UF terminological inconsistency | Throughout |
| Minor | Disfluency detector not validated on L2 speech | 2.2.2 |
| Minor | Filler classifier cross-lingual applicability not discussed | 2.2.4 |
| Minor | 150 ms ECP window unjustified | 2.2.5 |
| Minor | Japanese corpus inadequately described | 2.1.2 |
| Minor | Repair fluency measures absent but not discussed | Table 2, Limitations |
| Minor | Missing macro statistics for RQ2 | 3.2 |
| Minor | "Quality-filtered cohort" exclusion underspecified | 2.1.3 |
