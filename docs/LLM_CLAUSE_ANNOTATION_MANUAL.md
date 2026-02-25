# LLM-Assisted Clause Boundary Annotation Manual

**Project:** CAF Pipeline V2 — RQ1 (Clause Segmentation Accuracy) & RQ2 (Pause Classification)  
**Method:** Morin & Marttinen Larsson (2025) pipeline adapted for clause boundary annotation  
**Date:** 2026-02-06

---

## 1. Overview

This manual describes a step-by-step procedure for creating **human-validated clause boundary annotations** (gold standard `clauses_ref`) for 40 English L2 speech files, following:

- **Annotation rules:** Vercellotti & Hall (2024) broad clause definition
- **LLM pipeline:** Morin & Marttinen Larsson (2025, *Linguistics Vanguard*) iterative prompt engineering method
- **LLM model:** Claude 3.5/3.7 Sonnet (via claude.ai chat interface)

### What This Produces

The output is a set of **clause-segmented transcripts** for 40 files that serve as the gold standard for:

```
RQ1: Auto clause boundaries vs Gold standard (LLM + human review)
     → Report: Cohen's κ, Precision, Recall, F1
RQ2: Auto MCP/ECP vs Gold standard MCP/ECP (using gold clause boundaries)
     → Report: Cohen's κ, Precision, Recall, F1 per class
```

---

## 2. The Morin (2025) Pipeline — Adapted for Our Task

Morin & Marttinen Larsson (2025) used Claude 3.5 Sonnet for binary classification of grammatical constructions. Their pipeline achieved **93% accuracy** on a held-out test set through iterative training with unseen data rounds. We adapt their pipeline for clause boundary segmentation in 7 steps:

```
┌──────────────────────────────────────────────────────────────┐
│  STEP 1: FREEZE RULE SET                                     │
│  Adopt strict OWN-complement CVP rule (matches code).        │
│  No rule changes after this point.                           │
├──────────────────────────────────────────────────────────────┤
│  STEP 2: PREPARE 10 BLIND HUMAN FILES                        │
│  Expert annotates 10 files with NO LLM involvement:          │
│  5 → pretraining examples for Claude                         │
│  5 → locked test set (never shown to Claude during training) │
├──────────────────────────────────────────────────────────────┤
│  STEP 3: PROMPTING + PRETRAINING (one Claude conversation)   │
│  Paste system prompt + 5 gold pretraining examples.          │
│  Claude studies patterns, asks questions.                     │
├──────────────────────────────────────────────────────────────┤
│  STEP 4: SUPERVISED TRAINING                                 │
│  Claude segments unseen transcripts → you correct errors     │
│  → Claude learns from corrections (iterative batches)        │
├──────────────────────────────────────────────────────────────┤
│  STEP 5: UNSUPERVISED VALIDATION (iterative rounds)          │
│  Claude segments unseen transcripts with no feedback →       │
│  you evaluate accuracy → if <90%, give feedback and repeat   │
│  with new unseen files until ≥90% boundary F1                │
├──────────────────────────────────────────────────────────────┤
│  STEP 6: LOCKED TEST                                         │
│  Claude segments the 5 locked blind files (no more tuning).  │
│  Report accuracy. This is the true LLM performance metric.   │
├──────────────────────────────────────────────────────────────┤
│  STEP 7: PRODUCTION                                          │
│  Claude segments remaining 30 files → expert reviews all     │
│  → corrections logged → final gold standard                  │
└──────────────────────────────────────────────────────────────┘
```

**Key design principles from Morin (2025):**
1. Make prompts clear, specific, and contextualized
2. Include examples (few-shot)
3. Use XML tags (`<instructions>`, `<examples>`, `<thinking>`, `<answer>`)
4. Tell Claude to think step-by-step (Chain of Thought prompting)
5. Provide corrective feedback iteratively

---

## 3. Data Preparation

### 3.1 Input Format

Each file is an L2 English speech transcript extracted from ALLSSTAR TextGrids. The transcript is **unpunctuated** and contains **fillers** (um, uh, er, ah).

**Example transcript (ALL_001_F_GER_ENG_ST1):**
```
a black bird um sew a hat and put um this hat on uh his okay again a black 
bird sew a hat and put this hat on top of his hat and while he was walking 
a uh white bird saw the hat which the black bird put on top of his head and 
uh he was looking at the white bird bird was looking strangely at him and so 
the black bird got angry because the white uh bird was laughing so uh the 
blackbird uh got ashamed and left um the hat behind him while um he was 
running so the white bird um happily was sitting um on the grass and um 
caught the hat that the black bird um had left uh the white word bird w 
wanted this head to use it as a nest for um to lay um uh her um her egg 
into the nest but um the blackbird uh w was at the same time observing the 
white one and got angry so um he determined to um to get the hat back and 
suddenly appeared uh behind the whiteboard and got his and took his um hat 
back and put it um on top of his hat again but um surprisingly um uh the 
hat uh fell on the flow uh on the grass and he noticed that um there was a 
young bird on top of his uh hat
```

### 3.2 Expected Output Format

The LLM should produce a **numbered list of clauses** with clause type labels:

```
1. [IND] a black bird sew a hat
2. [CVP] and put this hat on top of his hat
3. [SUB_ADVCL] while he was walking
4. [IND] a white bird saw the hat
5. [SUB_REL] which the black bird put on top of his head
6. [IND] he was looking at
7. [IND] the white bird was looking strangely at him
...
```

**Clause type labels:**
| Label | Meaning | Example |
|-------|---------|---------|
| `IND` | Independent clause | "a black bird sew a hat" |
| `CVP` | Coordinated verb phrase (separate clause) | "and put this hat on his head" |
| `SUB_ADVCL` | Subordinate adverbial clause | "while he was walking" |
| `SUB_CCOMP` | Complement clause | "that there was a young bird" |
| `SUB_REL` | Relative clause | "which the black bird put on his head" |
| `NF_XCOMP` | Nonfinite complement (infinitive/gerund) | "to use it as a nest" |
| `NF_PART` | Nonfinite participle clause | "observing the white one" |
| `MINOR` | Stance verb hedge (I think, I believe) | "I think" |
| `IND_CL` | Copula-less independent clause | "it different from China" |

### 3.3 File Selection

Select 40 files for annotation. Recommend stratifying by:
- Task (ST1/ST2): ~20 each
- L1 background: mix of L1s
- Proficiency: range of WER levels (proxy for proficiency)

---

## 4. The Prompt

### 4.1 Initial System Prompt (Copy-paste into Claude)

```xml
You are a trained linguistic research assistant participating in a study on 
clause segmentation in L2 English speech. You will segment transcripts of 
spontaneous L2 English speech into clauses following the rules of 
Vercellotti & Hall (2024).

<task>
Given an unpunctuated transcript of L2 English speech (with fillers like 
"um", "uh", "er", "ah" present), you will:
1. Identify all clause boundaries
2. Label each clause with its type
3. Output a numbered list of clauses
</task>

<definition>
A CLAUSE is a verbal construction that contains a verb (finite or nonfinite) 
plus at least one additional syntactic element (complement or adjunct).

Following Vercellotti & Hall (2024), the definition of clause includes:
• Finite clauses (subject + finite verb) — ALWAYS a clause
• Coordinated verb phrases — a clause IF the verb has its OWN complement 
  or adjunct (e.g., "took my car and [went home]" = 2 clauses, but 
  "sang and danced" = 1 clause because neither has a complement)
• Nonfinite clauses (infinitive, gerund, participle) — a clause IF the 
  verb has a complement or adjunct (e.g., "[to get] the hat back" = clause, 
  but bare "to go" alone = not a clause)
• Verbal small clauses — a clause IF the verb has a complement 
  (e.g., "see them [open the basket]")
• Copula-less predicates — count as clause in L2 speech 
  (e.g., "it different from China")

NOT a clause:
• Verbless small clauses (e.g., "make them happy" — "happy" has no verb)
• Bare coordinated verbs without complements (e.g., "sang and danced")
• Bare nonfinite verbs without complements (e.g., "running" alone)
</definition>

<clause_types>
Use these labels:
- IND = Independent clause (main clause with finite verb)
- CVP = Coordinated verb phrase counted as separate clause 
        (verb + own complement/adjunct, joined by and/or/but)
- SUB_ADVCL = Subordinate adverbial clause 
              (because, while, when, if, although, so that, etc.)
- SUB_CCOMP = Complement clause (that-clause, reported speech)
- SUB_REL = Relative clause (who, which, that as relative pronoun)
- NF_XCOMP = Nonfinite clause: infinitive or gerund complement 
             (to + verb, verb-ing as complement) WITH own complement/adjunct
- NF_PART = Nonfinite participle clause (verb-ing or verb-ed as modifier)
            WITH own complement/adjunct
- MINOR = Stance verb used as epistemic hedge 
          (e.g., "I think", "I believe", "I guess" when functioning as 
          a hedge rather than a full matrix verb)
- IND_CL = Copula-less independent clause (L2 speech: missing copula)
</clause_types>

<rules>
KEY RULES for segmentation:

1. FILLER HANDLING: Fillers (um, uh, er, ah, oh) are NOT clauses. 
   They belong to whichever clause they appear in. Do not split clauses 
   at fillers. Remove them mentally when analyzing clause structure, but 
   keep them in the output text.

2. COORDINATED VP TEST: When you see "and/or/but" + verb, check:
   Does the second verb have its OWN complement or adjunct?
   - YES → separate clause (CVP): "took the car | and [went to the city]"
   - NO → same clause: "sang and danced" (neither has complement)
   
3. NONFINITE TEST: For infinitives (to + verb) and gerunds (verb-ing):
   Does the nonfinite verb have a complement or adjunct?
   - YES → separate clause: "[to use] it as a nest"
   - NO → not a separate clause: "wanted to go" (bare infinitive)

4. MINOR CLAUSE: "I think", "I believe", "I guess" etc. at sentence start 
   followed by another clause → tag as MINOR. But "I think about X" where 
   "think" has its own complement = regular IND clause.

5. L2 ERRORS: L2 speakers make grammatical errors. Interpret the intended 
   structure, not the surface form. Missing copulas, wrong tense, missing 
   subjects are common.

6. SELF-CORRECTIONS: When speakers correct themselves (false starts), 
   treat each grammatically complete attempt as part of the clause it 
   belongs to, unless it is clearly a separate clause.

7. DISCONTINUOUS MAIN CLAUSE: When a relative clause interrupts between 
   the subject and verb of a main clause (e.g., "this bird [that made 
   fun of him] found that it was a good idea"), the verb fragment absorbs 
   any following complement clause text. Do NOT leave a single-word 
   clause fragment. Example:
   - [IND] and so this bird
   - [SUB_REL] that made fun of this hairy bird
   - [IND] found that it was a very good idea
   - [NF_XCOMP] to use this hat as a nest
   The ccomp text ("that it was a very good idea") stays with "found" 
   rather than being extracted as a separate SUB_CCOMP line.
</rules>

<output_format>
For each transcript, output a numbered list:
<answer>
1. [LABEL] clause text including any fillers
2. [LABEL] clause text including any fillers
3. [LABEL] clause text including any fillers
...
</answer>

Every word in the transcript must appear in exactly one clause.
Clauses should be listed in the order they appear in the speech.
</output_format>

<thinking>
For each clause boundary decision, think step-by-step:
1. Find the verbs in the transcript
2. Determine which verbs are main verbs vs. auxiliaries
3. Check dependency: is this verb coordinated, subordinated, or independent?
4. Apply the "verb + element" test for coordinated/nonfinite verbs
5. Assign clause type label
</thinking>
```

### 4.2 Pretraining Examples (Phase 2)

After the system prompt, provide 5–10 expert-corrected transcript segmentations. Use this format:

```xml
<examples>

<example id="1">
<transcript>
a black bird sew a hat and put this hat on top of his hat and while he was 
walking a white bird saw the hat which the black bird put on top of his head
</transcript>
<segmentation>
1. [IND] a black bird sew a hat
2. [CVP] and put this hat on top of his hat
3. [SUB_ADVCL] and while he was walking
4. [IND] a white bird saw the hat
5. [SUB_REL] which the black bird put on top of his head
</segmentation>
<explanation>
- "sew a hat" = IND (finite verb + complement)
- "put this hat on top of his hat" = CVP (coordinated verb with OWN 
  complement "this hat" and adjunct "on top of his hat")
- "while he was walking" = SUB_ADVCL (subordinator "while" + finite clause)
- "saw the hat" = IND (new independent clause, new subject)
- "which...put on top of his head" = SUB_REL (relative pronoun "which")
</explanation>
</example>

<example id="2">
<transcript>
I think she is really happy and wants to go to the park to play with her 
friends
</transcript>
<segmentation>
1. [MINOR] I think
2. [SUB_CCOMP] she is really happy
3. [CVP] and wants to go to the park
4. [NF_XCOMP] to play with her friends
</segmentation>
<explanation>
- "I think" = MINOR (stance verb as hedge, followed by complement clause)
- "she is really happy" = SUB_CCOMP (complement of "think")
- "wants to go to the park" = CVP (coordinated VP with own complement 
  "to go to the park")
- "to play with her friends" = NF_XCOMP (infinitive with own complement 
  "with her friends")
</explanation>
</example>

<example id="3">
<transcript>
um so uh the boy um he was running very fast and uh fell on the ground 
because um the road was um very slippery
</transcript>
<segmentation>
1. [IND] um so uh the boy um he was running very fast
2. [CVP] and uh fell on the ground
3. [SUB_ADVCL] because um the road was um very slippery
</segmentation>
<explanation>
- Fillers (um, uh) kept inside clauses, not used for splitting
- "was running very fast" = IND (finite verb + adjunct)
- "fell on the ground" = CVP (coordinated VP with own adjunct "on the ground")
- "because the road was very slippery" = SUB_ADVCL (subordinator "because")
</explanation>
</example>

<example id="4">
<transcript>
it different from China so I I need to uh learn more about American culture
</transcript>
<segmentation>
1. [IND_CL] it different from China
2. [IND] so I I need to learn more about American culture
</segmentation>
<explanation>
- "it different from China" = IND_CL (copula-less: missing "is", but clearly 
  a predicate construction in L2 speech)
- "I need to learn more about American culture" = IND (finite clause; 
  "to learn" is bare infinitive complement of "need", not a separate clause 
  because it's an obligatory complement — but see note below)
- Note: "to learn more about American culture" COULD be a separate NF_XCOMP 
  if treated as having its own complement "about American culture". 
  Either analysis is defensible. Be consistent.
</explanation>
</example>

<example id="5">
<transcript>
and he noticed that um there was a young bird on top of his uh hat
</transcript>
<segmentation>
1. [IND] and he noticed
2. [SUB_CCOMP] that um there was a young bird on top of his uh hat
</segmentation>
<explanation>
- "he noticed" = IND (finite verb; "that" introduces complement clause)
- "that there was a young bird..." = SUB_CCOMP (complement clause with "that")
</explanation>
</example>

</examples>
```

**IMPORTANT:** Before using these examples, you (the expert) should **manually verify** that they are correct according to Vercellotti rules. Correct any errors in the auto pipeline output before feeding them to Claude.

---

## 5. Step-by-Step Procedure

### Step 1: Freeze Rule Set (Before Anything Else)

1. **Adopt strict OWN-complement CVP rule** — matches `textgrid_caf_segmenter_v2.py`
   - Coordinated VP = separate clause ONLY if the verb has its **own** complement/adjunct
   - "encourage and accompany me" = **1 clause** (shared complement "me")
   - "took my car and went home" = **2 clauses** (each verb has own complement)
2. **No rule changes after this point.** All annotations use this frozen rule set.

### Step 2: Prepare 10 Blind Human Files (Before Claude)

1. **Select 40 files** for annotation (stratified by task/L1)
2. **Extract transcripts** from the TextGrid word tiers (concatenate words, keep fillers)
3. **Expert-annotate 10 files manually** (blind, no LLM):
   - 5 files → pretraining examples for Claude (Step 3)
   - 5 files → locked test set for Step 6 (never shown to Claude during training)
4. **Export auto pipeline clause output** for the same 40 files (for comparison later)

### Step 3: Prompting + Pretraining (One Claude Conversation)

1. **Start a new Claude conversation** (Claude 3.5 Sonnet or newer)
2. **Paste the system prompt** from Section 4.1
3. **Paste the pretraining examples** from Section 4.2 (your 5 expert-annotated files)
4. **Ask Claude to study the examples:**

```
<instructions>
Study the 5 examples above carefully. Think about how each clause boundary 
was decided. Think step-by-step about whether you would segment the data 
the same way. If you have questions or comments about any segmentation 
decision, ask them now.
</instructions>
```

5. **Answer Claude's questions** and clarify any ambiguities

### Step 4: Supervised Training (In Claude)

1. **Give Claude unseen transcripts** to segment (ones you have NOT annotated):

```
<instructions>
Now segment the following transcript into clauses. Work through it 
step-by-step using <thinking></thinking> for your reasoning, then provide 
your segmentation in <answer></answer> format.
</instructions>

<transcript>
[paste transcript here]
</transcript>
```

2. **Review Claude's output** against your own analysis
3. **Provide corrective feedback** for any errors:

```
Your segmentation had the following errors:
- Clause 3: You split "sang and danced" into two clauses, but neither verb 
  has its own complement. This should be one clause.
- Clause 7: You missed the subordinate clause "because he was tired". 
  The word "because" introduces a new SUB_ADVCL clause.
Please acknowledge these corrections and explain what you learned.
```

4. **Repeat** with more unseen transcripts until Claude's accuracy improves visibly

### Step 5: Unsupervised Validation (Iterative Rounds)

1. **Give Claude unseen transcripts** (files not used in Steps 2–4)
2. **Claude segments them without any feedback**
3. **Compare** Claude's output to your own analysis
4. **Calculate boundary F1:**
   - Count total clause boundaries in your annotation
   - Count how many Claude matched (±1 word tolerance)
   - Compute Precision, Recall, F1

5. **If F1 ≥ 90%:** proceed to Step 6
6. **If F1 < 90%:** provide feedback on errors, then repeat Step 5 with **new unseen files** (Morin-style: each round always uses unseen data)

### Step 6: Locked Test (No More Tuning)

1. **Give Claude the 5 locked blind files** (from Step 2 — never seen, never trained on)
2. **Claude segments them without feedback**
3. **Compare** Claude's output to your blind annotations
4. **Report:** This is the true LLM performance metric (Precision, Recall, F1)
5. **No further tuning allowed** after this step

### Step 7: Production (In Claude)

1. **Feed Claude the remaining 30 transcripts** (in batches of 5–10)
2. **For each batch:**

```
<instructions>
Segment the following transcripts into clauses following the Vercellotti & 
Hall (2024) rules we have practiced. For each transcript, provide your 
segmentation in <answer></answer> format. Think step-by-step but you may 
keep your thinking brief.
</instructions>

<transcript id="ALL_025_M_TUR_ENG_ST1">
[paste transcript]
</transcript>

<transcript id="ALL_030_F_CMN_ENG_ST2">
[paste transcript]
</transcript>
```

3. **Expert reviews ALL output** — correct any errors
4. **Log corrections** (for reporting LLM accuracy and correction rate)

---

## 6. Post-Processing

### 6.1 Converting LLM Output to clauses_ref TextGrid

After collecting all segmented transcripts, convert them back to TextGrid format:

1. Each clause's text → match to word timestamps in the original TextGrid
2. First word of clause → clause start time
3. Last word of clause → clause end time
4. Create `clauses_ref` interval tier with these boundaries

### 6.2 Comparison with Auto Pipeline

```
For each of the 40 files:
  - clauses_auto: from auto pipeline (spaCy segmenter)
  - clauses_ref:  from LLM + human review (gold standard)
  
Compute:
  - Boundary matching (±0.15s tolerance)
  - Cohen's κ
  - Precision, Recall, F1
  - Per-clause-type accuracy
```

---

## 7. Metrics to Report in Paper

### 7.1 LLM Performance (from Step 6: Locked Test)

| Metric | Value |
|--------|-------|
| LLM F1 on locked blind test set (5 files) | [X]% |
| Number of boundaries in test set | [N] |
| Boundaries correctly identified | [M] |
| Precision | [P] |
| Recall | [R] |
| F1 | [F] |

### 7.2 Correction Rate (from Step 7: Production)

| Metric | Value |
|--------|-------|
| Total clause boundaries (30 production files) | [N] |
| Boundaries corrected by expert | [C] |
| Correction rate | [C/N]% |
| Types of corrections (added/removed/moved/relabeled) | [breakdown] |

### 7.3 Final RQ1 Results (Clause Boundary Agreement)

| Metric | Value |
|--------|-------|
| Auto vs Gold standard: Cohen's κ | [κ] |
| Auto vs Gold standard: Precision | [P] |
| Auto vs Gold standard: Recall | [R] |
| Auto vs Gold standard: F1 | [F] |

### 7.4 Final RQ2 Results (Pause Classification Agreement)

| Metric | Value |
|--------|-------|
| MCP: Auto vs Gold standard: Cohen's κ | [κ] |
| MCP: Precision / Recall / F1 | [P] / [R] / [F] |
| ECP: Auto vs Gold standard: Cohen's κ | [κ] |
| ECP: Precision / Recall / F1 | [P] / [R] / [F] |

---

## 8. What to Write in the Paper

### Methods Section

> Clause boundaries were annotated following the broader clause definition of Vercellotti & Hall (2024), which includes finite clauses, coordinated verb phrases with complements, nonfinite clauses with complements, and verbal small clauses. 
>
> Annotations were generated using an LLM-assisted pipeline adapted from Morin & Marttinen Larsson (2025). Specifically, Claude [version] was trained through iterative prompt engineering with expert-annotated examples of L2 English speech transcripts. The training followed a seven-step process: (1) rule set frozen to strict OWN-complement coordinated VP rule matching the automated segmenter, (2) 10 files annotated blind by the first author (5 pretraining, 5 locked test), (3) prompt design with Vercellotti & Hall (2024) clause coding rules and 5 gold examples, (4) supervised training with corrective feedback, (5) iterative unsupervised validation on unseen data until $\geq$90\% boundary F1, (6) locked test on 5 held-out files achieving [X]\% F1, and (7) production annotation of the remaining 30 files with expert review.
>
> To address potential anchoring bias (cf. Li et al., 2023), [N] files were annotated independently by the first author without LLM assistance prior to LLM training, serving as a blind calibration set for evaluating true LLM accuracy. For the remaining [N] files, the first author reviewed and corrected all LLM-generated clause boundaries. [X]% of boundaries required correction.
>
> The LLM model, version, complete prompt, and all training transcripts are available in the supplementary materials.

### Reproducibility

Log and archive the following:
- [ ] Claude model version (e.g., Claude 3.5 Sonnet, November 2024)
- [ ] Complete conversation transcript (export from claude.ai)
- [ ] All input transcripts (40 files)
- [ ] All LLM output (raw, before correction)
- [ ] All expert corrections (diff from LLM output)
- [ ] Final gold standard annotations (clauses_ref)
- [ ] Evaluation metrics

---

## 9. Practical Tips

### 9.1 Conversation Management

- **One conversation per project** — Claude's training is conversation-scoped (Anthropic, 2024b). Starting a new conversation loses all training.
- **Use Claude Projects** — Attach the Vercellotti paper and methodology doc as project knowledge.
- **Batch size:** 5–10 transcripts per message (too many may degrade quality).

### 9.2 Common Edge Cases in L2 Speech

| Case | Rule |
|------|------|
| "I sang and danced" | 1 clause (no complements on either verb) |
| "I took my car and went home" | 2 clauses (each verb has own complement) |
| "she would be patient and comfort me" | 2 clauses (shared modal; "comfort me" has complement) |
| "encourage and accompany me" | 1 clause (shared complement "me" — strict OWN-complement rule) |
| "I hope to meet her" | 2 clauses ("meet her" = NF_XCOMP with complement) |
| "I want to go" | 1 clause (bare infinitive, no complement on "go") |
| "it different from China" | 1 clause: IND_CL (copula-less) |
| "I think she is happy" | 2 clauses: MINOR + SUB_CCOMP |
| "I think about the problem" | 1 clause: IND (think has its own complement) |
| "there's someone driving his car" | 2 clauses (NF_PART: "driving his car" has complement) |
| "he determined to get the hat back" | 2 clauses (NF_XCOMP: "to get the hat back" has complement) |
| "make them happy" | 1 clause ("happy" = verbless small clause → NOT a separate clause) |
| "this bird that laughed at him found that it was fun" | 3 clauses: IND ("this bird") + SUB_REL ("that laughed at him") + IND ("found that it was fun") — ccomp stays with verb to avoid 1-word fragment |

### 9.3 Ambiguous Cases — Be Consistent

Some segmentation decisions are genuinely ambiguous. Document your decisions and apply them consistently:

1. **"want/need/try to V"**: Is the infinitive a separate clause?
   - Decision: Only if the infinitive verb has its own complement/adjunct beyond the obligatory link to the matrix verb
   - "need to learn more about culture" → debatable; pick one rule and stick with it

2. **Self-corrections**: "I went to... I went to the store"
   - Decision: If the false start is incomplete, merge with the completed clause

3. **Fillers between clauses**: "so um uh the bird..."
   - Decision: Attach fillers to the following clause

---

## 10. References

- Morin, C. & Marttinen Larsson, M. (2025). Large corpora and large language models: a replicable method for automating grammatical annotation. *Linguistics Vanguard, 11*(1), 501–510. https://doi.org/10.1515/lingvan-2024-0228
- Vercellotti, M. L., & Hall, S. (2024). Coding all clauses in L2 data: A call for consistency. *Research Methods in Applied Linguistics, 3*, 100132. https://doi.org/10.1016/j.rmal.2024.100132
- Anthropic. (2024a). Claude 3.5 Sonnet [large language model]. https://claude.ai/
- Anthropic. (2024b). Collaborate with Claude on projects. https://www.anthropic.com/news/projects
- Li, M., et al. (2023). CoAnnotating: Uncertainty-Guided Work Allocation between Human and Large Language Models for Data Annotation. *EMNLP 2023*, pp. 1487–1505.
- Gilardi, F., Alizadeh, M., & Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *PNAS*. https://doi.org/10.1073/pnas.2305016120

---

## Appendix A: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│  CLAUSE = Verb + Complement/Adjunct                         │
│                                                             │
│  ✅ Count as clause:                                        │
│    • Finite clause (subject + finite verb)                  │
│    • Coordinated VP with OWN complement                     │
│    • Nonfinite clause with complement                       │
│    • Verbal small clause with complement                    │
│    • Copula-less predicate (L2 speech)                      │
│                                                             │
│  ❌ NOT a clause:                                           │
│    • Bare verb (no complement): "sang and danced"           │
│    • Verbless small clause: "make them happy"               │
│    • Filler: um, uh, er, ah                                 │
│                                                             │
│  LABELS: IND CVP SUB_ADVCL SUB_CCOMP SUB_REL               │
│          NF_XCOMP NF_PART MINOR IND_CL                     │
└─────────────────────────────────────────────────────────────┘
```

## Appendix B: Checklist

- [ ] 40 files selected (stratified by task/L1)
- [ ] 10 files expert-annotated blind (5 pretraining + 5 test)
- [ ] Transcripts extracted from TextGrids
- [ ] Claude conversation started with system prompt
- [ ] Pretraining examples provided (5 corrected transcripts)
- [ ] Supervised training completed (5+ transcripts with feedback)
- [ ] Validation accuracy ≥ 90% on held-out set
- [ ] All 40 files segmented by Claude
- [ ] Expert reviewed/corrected all Claude output
- [ ] Corrections logged (count, type)
- [ ] Converted to clauses_ref TextGrids
- [ ] Compared clauses_auto vs clauses_ref
- [ ] Metrics computed (κ, F1, precision, recall)
- [ ] Conversation transcript archived
- [ ] All materials ready for supplementary submission
