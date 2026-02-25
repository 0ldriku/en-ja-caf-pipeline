"""
Train xlm-roberta-base for token-level disfluency detection.
Adapted from Kundu et al. (2022) train.py

Trains on combined English + Japanese synthetic disfluency data.
Uses Hugging Face Trainer API with GPU support.
"""
import os
import sys
import numpy as np
import torch
import argparse
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed,
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

set_seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

parser = argparse.ArgumentParser(description='Train disfluency detection model')
parser.add_argument('--data-dir', '-d', default=os.path.join(PROJECT_DIR, "data", "labeled"),
                    type=str, help='Labeled data directory')
parser.add_argument('--model', '-m', default='xlm-roberta-base', type=str, help='Model checkpoint')
parser.add_argument('--epochs', '-e', default=3, type=int, help='Number of epochs')
parser.add_argument('--batch-size', '-b', default=16, type=int, help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float, help='Learning rate')
parser.add_argument('--output-dir', '-o', default=os.path.join(PROJECT_DIR, "model"),
                    type=str, help='Output model directory')
args = parser.parse_args()

model_checkpoint = args.model
batch_size = args.batch_size
label_list = ['is_fluent', 'is_disfluent']
label_all_tokens = True

print(f"Model: {model_checkpoint}")
print(f"Data: {args.data_dir}")
print(f"Epochs: {args.epochs}, Batch size: {batch_size}, LR: {args.lr}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")


def get_dataset(data_dir):
    splits = {}
    for split in ['train', 'valid', 'test']:
        dis_path = os.path.join(data_dir, f"{split}.dis")
        lab_path = os.path.join(data_dir, f"{split}.labels")

        if not os.path.exists(dis_path) or not os.path.exists(lab_path):
            print(f"WARNING: {split} files not found, skipping.")
            continue

        data = {'labels': [], 'disfluent': []}
        with open(dis_path, 'r', encoding='utf-8') as f_dis, \
             open(lab_path, 'r', encoding='utf-8') as f_lab:
            for dis_line, lab_line in zip(f_dis, f_lab):
                words = dis_line.strip().split()
                labels = list(map(int, lab_line.strip().split()))
                if len(words) == len(labels) and len(words) > 0:
                    data['disfluent'].append(words)
                    data['labels'].append(labels)

        splits[split] = Dataset.from_dict(data)
        print(f"  {split}: {len(data['disfluent'])} sentences")

    return DatasetDict(splits)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["disfluent"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        p for prediction, label in zip(predictions, labels)
        for (p, l) in zip(prediction, label) if l != -100
    ]
    true_labels = [
        l for prediction, label in zip(predictions, labels)
        for (p, l) in zip(prediction, label) if l != -100
    ]

    results = precision_recall_fscore_support(true_labels, true_predictions, zero_division=0)
    return {
        'accuracy': accuracy_score(true_labels, true_predictions),
        'precision0': torch.tensor(results[0])[0],
        'precision1': torch.tensor(results[0])[1],
        'recall0': torch.tensor(results[1])[0],
        'recall1': torch.tensor(results[1])[1],
        'f1score0': torch.tensor(results[2])[0],
        'f1score1': torch.tensor(results[2])[1],
    }


# Load data
print("\nLoading data...")
datasets = get_dataset(args.data_dir)

# Tokenize
print("\nTokenizing...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

# Model
print("\nLoading model...")
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

# Training args
checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
training_args = TrainingArguments(
    checkpoint_dir,
    eval_strategy="steps",
    learning_rate=args.lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    eval_steps=1000,
    logging_steps=200,
    save_steps=1000,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1score1",
    save_only_model=True,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)
trainer.train()
trainer.evaluate()

# Test
if "test" in tokenized_datasets:
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        p for prediction, label in zip(predictions, labels)
        for (p, l) in zip(prediction, label) if l != -100
    ]
    true_labels = [
        l for prediction, label in zip(predictions, labels)
        for (p, l) in zip(prediction, label) if l != -100
    ]

    results = precision_recall_fscore_support(true_labels, true_predictions, zero_division=0)
    print({
        'precision': results[0],
        'recall': results[1],
        'f1score': results[2]
    })
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, true_predictions, normalize='all'))
    print("\n" + classification_report(true_labels, true_predictions,
                                       target_names=label_list, zero_division=0))

# Save final model
final_dir = os.path.join(args.output_dir, "final")
print(f"\nSaving final model to {final_dir}...")
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)
print("Done!")
