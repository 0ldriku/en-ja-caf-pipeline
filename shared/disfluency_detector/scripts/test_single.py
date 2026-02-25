"""Quick test: run model on a single sentence with MeCab preprocessing."""
import sys, os, io, re, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from transformers import AutoTokenizer, AutoModelForTokenClassification
import fugashi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL = os.path.join(PROJECT_DIR, "model_v2", "final")

# Input
raw = "ＩＤお願いします ＣＣＨ１２ えー、朝、えーケンとマリ、は、えー、ピクニック、の準備を、していました あの、えー、犬は、えーそばに、行って、えずっと、あー、ケンとマリさんの、えーことを見ています えー、あ、ケンとマリは、えー、サンドイッチを、作り、まし、た後、ずっと地図を見てます んーピクニックの、あー、あ、場所、を考え、ています えーピク、ニックいについてのことを、はね、しあいます あー、でも、そばに、いる、あー犬はひそかに、えーバスケットに、入っちゃった んー、ケン、ケンさんとマリさんは、えー、公園、に、着き、ましたが、着きました後、えー犬は、急にあーバスケットの中、えー、えー、はね、えー、はね、は跳ねました あ、あー、あー犬は、ずっと、えー、バスケットの中、で、います えー、さんどいちとりんごは全部、えー犬に食べられちゃ、食べられました"
print(f"Raw: {raw}\n")

# Step 1: Remove punctuation
nopunct = re.sub(r'[、。？！「」（）,.\u3000]+', '', raw)

# Step 2: MeCab tokenize
tagger = fugashi.Tagger()
tokens = [w.surface for w in tagger(nopunct)]
# Remove any whitespace-only tokens
tokens = [t for t in tokens if t.strip()]
print(f"MeCab tokens ({len(tokens)}): {' '.join(tokens)}\n")

# Step 3: Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=2)
model.eval()

# Step 4: Predict
inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits
preds = torch.argmax(logits, dim=2)[0].tolist()
word_ids = inputs.word_ids()

# Majority vote per word
word_preds = {}
for idx, wid in enumerate(word_ids):
    if wid is not None:
        word_preds.setdefault(wid, []).append(preds[idx])

# Step 5: Display results
print("Results:")
fluent = []
for i, tok in enumerate(tokens):
    votes = word_preds.get(i, [0])
    label = 1 if sum(votes) > len(votes) / 2 else 0
    tag = "DISFL" if label == 1 else "ok"
    marker = "  <--" if label == 1 else ""
    print(f"  {tok:12s} {tag}{marker}")
    if label == 0:
        fluent.append(tok)

print(f"\nCleaned: {''.join(fluent)}")
