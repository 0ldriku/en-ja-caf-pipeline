"""
Batch test disfluency detection on real L2 English and Japanese transcripts.
Uses the trained model to predict disfluent tokens and produce cleaned text.
Follows Kundu et al. test-interactive.py prediction logic.
"""
import sys
import io
import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import Dataset

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "model", "final")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

label_list = ['is_fluent', 'is_disfluent']


def load_model(model_dir):
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=len(label_list))
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(model, data_collator=data_collator, processing_class=tokenizer)
    return tokenizer, trainer


def predict_sentence(sentence, tokenizer, trainer):
    """Predict disfluency labels for a sentence using Kundu's logic."""
    words = sentence.strip().split()
    if not words:
        return words, [], ""

    def test_tokenizer_fn(examples):
        return tokenizer(examples["disfluent"], truncation=True, is_split_into_words=True)

    test_dict = {'disfluent': [words]}
    test_dataset = Dataset.from_dict(test_dict)
    test_dataset = test_dataset.map(test_tokenizer_fn, batched=True)
    test_dataset = test_dataset.remove_columns(['disfluent'])

    prediction, _, _ = trainer.predict(test_dataset)
    prediction = np.argmax(prediction, axis=2)

    tokenized_input = tokenizer.tokenize(words, is_split_into_words=True)
    word_ids = tokenizer(words, is_split_into_words=True).word_ids()[1:-1]

    previous_word_idx = None
    disfluent_count = 0
    fluent_count = 0
    fluent_sentence = []
    word_labels = []

    for idx, predicted_label in enumerate(prediction[0][1:1 + len(tokenized_input)]):
        if word_ids[idx] != previous_word_idx:
            if previous_word_idx is not None:
                is_fluent = fluent_count >= disfluent_count
                word_labels.append(0 if is_fluent else 1)
                if is_fluent:
                    fluent_sentence.append(words[previous_word_idx])
            fluent_count, disfluent_count = 0, 0

        if predicted_label == 0:
            fluent_count += 1
        else:
            disfluent_count += 1
        previous_word_idx = word_ids[idx]

    if previous_word_idx is not None:
        is_fluent = fluent_count >= disfluent_count
        word_labels.append(0 if is_fluent else 1)
        if is_fluent:
            fluent_sentence.append(words[previous_word_idx])

    return words, word_labels, " ".join(fluent_sentence)


# ============================================================
# Test sentences
# ============================================================

TEST_CASES = {
    # English L2 examples (from user's actual transcripts)
    "en_short": "then michael get hats and walking along and then michael take put put his child into the hat at then",

    "en_full": "ok one day arisa found the hat on the road and then arisa took this hat on hat on ha ha head then arisa is walking along and then arisa met with michael and michael said said that oh this hat is mine so could you could you put me back with his so arisa gabe this hat to michael and then michael get hats and walking along and then michael put put huh put his child into the hat at then alisa found again lisa again found the michael and lisa took took the hat from michael and put put it on ha head and then michael got angry but in lisa took off ha ha hat then the michael chart appears again",

    # Japanese L2 example (from user's actual transcripts)
    "ja_test": "彼らは いい ビクニク で 壊して こ 壊して 彼らう こわられて 彼らは あーん とても がっかり しました",

    # Additional test sentences
    "en_test2": "I think I think the the boy is is um running to the to the park and and he he fell down",

    "ja_test2": "えーと 私 は あの 昨日 えー 学校 に に 行き いや 行っ て あの 友達 と と 話し ました",
}


if __name__ == "__main__":
    tokenizer, trainer = load_model(MODEL_DIR)

    print("\n" + "=" * 80)
    print("L2 DISFLUENCY DETECTION RESULTS")
    print(f"Model: xlm-roberta-base trained on EN+JA synthetic data")
    print("=" * 80)

    all_results = {}
    for name, text in TEST_CASES.items():
        words, labels, clean = predict_sentence(text, tokenizer, trainer)

        print(f"\n{'─' * 80}")
        print(f"TEST: {name}")
        print(f"{'─' * 80}")

        disfluent_words = []
        for i, (w, l) in enumerate(zip(words, labels)):
            marker = "*** DISFL" if l == 1 else ""
            print(f"  {i:<4} {w:<15} {marker}")
            if l == 1:
                disfluent_words.append(w)

        n_disfl = len(disfluent_words)
        n_total = len(words)
        pct = 100 * n_disfl / max(n_total, 1)

        print(f"\n  Total: {n_total} | Disfluent: {n_disfl} ({pct:.1f}%)")
        print(f"  Flagged: {disfluent_words}")
        print(f"  Original: {text}")
        print(f"  Cleaned:  {clean}")

        all_results[name] = {
            "original": text,
            "cleaned": clean,
            "words": words,
            "labels": labels,
            "disfluent_words": disfluent_words,
            "stats": {"total": n_total, "disfluent": n_disfl, "pct": round(pct, 1)}
        }

    # Save
    out_path = os.path.join(RESULTS_DIR, "l2_detection_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
