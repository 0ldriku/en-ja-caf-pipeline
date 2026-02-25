# L2 Disfluency Detector

Binary (fluent/disfluent) word-level disfluency detection for English and Japanese L2 speech transcripts. Based on Kundu et al. (2022) "Zero-shot Disfluency Detection for Indian Languages" (COLING 2022).

Canonical shared runtime path used by EN/JA pipeline scripts:

- `shared/disfluency_detector/model_v2/final` (preferred)
- `en/disfluency_test/l2_disfluency_detector/model_v2/final` (legacy fallback)

## What It Does

Given a disfluent spoken transcript, the model labels each word as **fluent** (keep) or **disfluent** (remove). Example:

```
Input:  えん 来たら あ 犬 犬 を ばず けっと をー 降ります
Labels:  1    0     1  1  0  0  0    0    1    0
Output:      来たら    犬 を ばず けっと    降ります
```

The model outputs **binary labels only** (0/1). It does not classify disfluency type (filler, repetition, etc.).

## How It Was Built

### Step 1: Synthetic Data Generation

Clean sentences were collected, then L2 disfluency patterns were injected programmatically:

- **English source:** 50K sentences from NLTK (Brown + Gutenberg corpora)
- **Japanese source:** 50K sentences from Tatoeba, tokenized with MeCab/fugashi
- **Injection:** 7 disfluency types per language (see table below)
- **Output:** 20K train + 2.5K valid sentences per language = **40K synthetic total**

### Step 2: Switchboard Real English Data

To improve performance on real English disfluencies (discourse markers, repairs), we added the Switchboard Dialog Act corpus transcriptions:

- **Source:** [swda repo](https://github.com/cgpotts/swda) (CC BY-NC-SA 3.0, free)
- **Processing:** Parsed disfluency annotations (`{D ...}`, `{F ...}`, `[reparandum + repair]`, etc.) to extract word-level labels
- **Output:** **48.5K labeled English sentences** with real disfluencies (discourse markers, filled pauses, repairs, partial words)
- **Split:** Standard Switchboard split following Wang et al. (2021) / Kundu et al. (2022)

### Step 3: Model Training

- **Model:** `xlm-roberta-base` (278M params) fine-tuned for token classification (2 labels)
- **Training data (v2):** 88.5K sentences (48.5K Switchboard + 20K synthetic EN + 20K synthetic JA)
- **Validation/test:** Synthetic EN+JA only (for fair comparison)
- **Hyperparameters:** 3 epochs, batch size 16, learning rate 2e-5, AdamW
- **Training time:** ~1 hour on single GPU (RTX 3060)

### Two Model Versions

| | Model v1 (`model/final`) | Model v2 (`model_v2/final`) |
|---|---|---|
| **Training data** | 40K synthetic EN+JA only | 88.5K (Switchboard + synthetic EN+JA) |
| **Approach** | Pure zero-shot (no real data) | Switchboard + zero-shot synthetic |
| **Synthetic EN F1** | 0.990 | 0.987 |
| **Synthetic JA F1** | 0.996 | 0.997 |
| **DisfluencySpeech F1** | 0.54 | **0.73** |
| **Use case** | L2-only disfluencies | L2 + native English disfluencies |

**Recommended: model v2** for general use.

## L2 Disfluency Types Injected

### English L2 (common for Japanese L1 speakers)
| Type | Example | Code |
|------|---------|------|
| **Filled pause** | "I *um* went to the store" | FP |
| **Word repetition** | "I *went went* to the store" | WR |
| **Phrase repetition** | "I went to *I went to* the store" | PR |
| **Self-correction** | "I *walked* no I mean *went* to the store" | SC |
| **False start/restart** | "The thing *uh* I went to the store" | FS |
| **Partial word** | "I *wal-* walked to the store" | PW |
| **Article/prep correction** | "I went *at* uh *to* the store" | LC |

### Japanese L2 (common for English L1 speakers)
| Type | Example | Code |
|------|---------|------|
| **Filled pause** | "えーと 学校 に 行きました" | FP |
| **Word repetition** | "学校 *学校* に 行きました" | WR |
| **Phrase repetition** | "学校 に *学校 に* 行きました" | PR |
| **Self-correction** | "学校 *で* あ *に* 行きました" | SC |
| **False start** | "がっこ 学校 に 行きました" | FS |
| **Partial word** | "*い* 行きました" | PW |
| **Particle correction** | "学校 *を* えーと *に* 行きました" | PC |

## Project Structure

```
l2_disfluency_detector/
  README.md                 ← this file
  PROGRESS.md               ← detailed results and evaluation log

  scripts/
    download_source_data.py   ← download clean EN (NLTK) + JA (Tatoeba) sentences
    inject_english.py         ← inject 7 L2 disfluency types into English
    inject_japanese.py        ← inject 7 L2 disfluency types into Japanese
    prepare_labels.py         ← generate word-level 0/1 labels from .dis/.flu pairs
    download_switchboard.py   ← download swda, parse annotations, combine with synthetic
    train.py                  ← fine-tune xlm-roberta-base (Hugging Face Trainer)
    test_batch.py             ← batch predict on real L2 transcripts (outputs JSON)
    test_single.py            ← quick test on a single sentence
    eval_english.py           ← evaluate on fresh synthetic EN data (2K sentences)
    eval_japanese.py          ← evaluate on fresh synthetic JA data (2K sentences)
    eval_disfluencyspeech.py  ← evaluate on DisfluencySpeech dataset (real English)
    eval_disfluencyspeech_detail.py ← detailed error analysis on DisfluencySpeech

  data/
    source/                   ← clean source sentences (EN + JA)
    synthetic/en/             ← synthetic EN disfluent .dis + .flu files
    synthetic/ja/             ← synthetic JA disfluent .dis + .flu files
    labeled/                  ← combined synthetic train/valid/test .dis + .labels
    switchboard/              ← Switchboard-only .dis + .labels (48.5K train)
    combined/                 ← Switchboard + synthetic merged (88.5K train, for v2)

  model/final/                ← model v1 (synthetic only)
  model_v2/final/             ← model v2 (Switchboard + synthetic) ← RECOMMENDED
  results/                    ← evaluation outputs (JSON)
```

## Usage

### Reproduce from scratch

```bash
# 1. Download clean source sentences (EN + JA)
python scripts/download_source_data.py

# 2. Generate synthetic disfluent data
python scripts/inject_english.py
python scripts/inject_japanese.py

# 3. Prepare word-level labels (Kundu format)
python scripts/prepare_labels.py

# 4. Download Switchboard and combine with synthetic data
python scripts/download_switchboard.py

# 5. Train model v2 (Switchboard + synthetic)
python scripts/train.py -d data/combined -o model_v2

# 6. Evaluate
python scripts/eval_english.py       # fresh synthetic EN
python scripts/eval_japanese.py       # fresh synthetic JA
python scripts/eval_disfluencyspeech.py  # real English (DisfluencySpeech)
```

### Use the trained model on your own data

The model expects **space-separated words** as input. How you get those depends on the language:

- **English:** Already has spaces between words. Just split on spaces.
- **Japanese:** No spaces between words. You **must** preprocess with MeCab (via `fugashi`) to split into words first, then remove punctuation.

#### English input

```python
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = "model_v2/final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, num_labels=2)
model.eval()

# English: just split on spaces
tokens = "I think I think the the boy is is um running".split()

inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
with torch.no_grad():
    preds = torch.argmax(model(**inputs).logits, dim=2)[0].tolist()

word_ids = inputs.word_ids()
word_preds = {}
for idx, wid in enumerate(word_ids):
    if wid is not None:
        word_preds.setdefault(wid, []).append(preds[idx])

fluent_words = []
for i, tok in enumerate(tokens):
    votes = word_preds.get(i, [0])
    if not (sum(votes) > len(votes) / 2):
        fluent_words.append(tok)

print("Cleaned:", " ".join(fluent_words))
# Output: I think the boy is running
```

#### Japanese input (requires MeCab preprocessing)

Japanese text has no spaces, so you must tokenize with MeCab first. Install `fugashi` and `unidic-lite`:

```bash
pip install fugashi unidic-lite
```

```python
import torch, re
import fugashi
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = "model_v2/final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, num_labels=2)
model.eval()

# Step 1: Remove punctuation
raw = "えー、朝、えーケンとマリ、は、えー、ピクニック、の準備を、していました"
nopunct = re.sub(r'[、。？！「」（）,.\u3000]+', '', raw)

# Step 2: MeCab tokenize into words
tagger = fugashi.Tagger()
tokens = [w.surface for w in tagger(nopunct) if w.surface.strip()]
# tokens = ['えー', '朝', 'えー', 'ケン', 'と', 'マリ', 'は', 'えー', 'ピクニック', 'の', '準備', 'を', 'し', 'て', 'い', 'まし', 'た']

# Step 3: Feed to model
inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
with torch.no_grad():
    preds = torch.argmax(model(**inputs).logits, dim=2)[0].tolist()

word_ids = inputs.word_ids()
word_preds = {}
for idx, wid in enumerate(word_ids):
    if wid is not None:
        word_preds.setdefault(wid, []).append(preds[idx])

fluent_words = []
for i, tok in enumerate(tokens):
    votes = word_preds.get(i, [0])
    if not (sum(votes) > len(votes) / 2):
        fluent_words.append(tok)

print("Cleaned:", "".join(fluent_words))
# Output: 朝ケンとマリはピクニックの準備をしていました
```

## Dependencies

```
torch>=2.0
transformers>=4.30
datasets>=2.14
scikit-learn>=1.3
fugashi>=1.3          # Japanese word segmentation (required for Japanese inference)
unidic-lite           # MeCab dictionary (required for Japanese inference)
nltk>=3.8             # English source sentences (for data generation only)
```

Note: For English inference, only `torch` and `transformers` are required. For Japanese inference, `fugashi` and `unidic-lite` are also needed to split text into words.

## References

- Kundu, R., Jyothi, P., & Bhattacharyya, P. (2022). "Zero-shot Disfluency Detection for Indian Languages." *Proceedings of COLING 2022*, pp. 4442-4454. [Paper](https://aclanthology.org/2022.coling-1.392/) | [Code](https://github.com/RKKUNDU/zero-shot-disfluency-detection)
- Godfrey, J., Holliman, E., & McDaniel, J. (1992). "SWITCHBOARD: Telephone speech corpus for research and development." *ICASSP 1992*.
- Potts, C. (2012). Switchboard Dialog Act Corpus. [swda repo](https://github.com/cgpotts/swda) (CC BY-NC-SA 3.0)
- Shriberg, E. (1994). "Preliminaries to a theory of speech disfluencies." PhD thesis, UC Berkeley.
- Lea, R., et al. (2024). "DisfluencySpeech." [HuggingFace](https://huggingface.co/datasets/amaai-lab/DisfluencySpeech)
