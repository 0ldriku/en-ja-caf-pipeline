"""
Prepare labeled data for disfluency detection training.
Adapted from Kundu et al. (2022) prepare-labeled-dataset.py

Takes .dis (disfluent) + .flu (fluent) pairs and generates .dis + .labels files
where labels are 0 (fluent) or 1 (disfluent) at the word level.

Combines English and Japanese data into a single training set.
"""
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SYNTHETIC_DIR = os.path.join(PROJECT_DIR, "data", "synthetic")
LABELED_DIR = os.path.join(PROJECT_DIR, "data", "labeled")
os.makedirs(LABELED_DIR, exist_ok=True)


def isSubSequence(str1, str2):
    """Check if str1 is a subsequence of str2 at word level."""
    m, n = len(str1), len(str2)
    j, i = 0, 0
    while j < m and i < n:
        if str1[j] == str2[i]:
            j += 1
        i += 1
    return j == m


def generate_labels(dis_path, flu_path):
    """Generate labels from disfluent/fluent pairs. Returns (dis_lines, label_lines)."""
    with open(dis_path, 'r', encoding='utf-8') as f:
        dis_lines = f.readlines()
    with open(flu_path, 'r', encoding='utf-8') as f:
        flu_lines = f.readlines()

    out_dis = []
    out_labels = []
    skipped = 0

    for dis_line, flu_line in zip(dis_lines, flu_lines):
        dis_line = dis_line.strip()
        flu_line = flu_line.strip()

        if flu_line == "None":
            flu_line = ""

        dis_words = dis_line.split()
        flu_words = flu_line.split()

        if not isSubSequence(flu_words, dis_words):
            skipped += 1
            continue

        # Label: compare from end (Kundu's approach)
        i = len(dis_words) - 1
        j = len(flu_words) - 1
        labels = [1] * len(dis_words)

        while i >= 0:
            if j >= 0 and dis_words[i] == flu_words[j]:
                labels[i] = 0
                j -= 1
            i -= 1

        if j != -1:
            skipped += 1
            continue

        out_dis.append(dis_line)
        out_labels.append(" ".join(map(str, labels)))

    return out_dis, out_labels, skipped


if __name__ == "__main__":
    print("=" * 60)
    print("Preparing labeled data (Kundu format)")
    print("=" * 60)

    all_train_dis = []
    all_train_labels = []
    all_valid_dis = []
    all_valid_labels = []

    for lang in ["en", "ja"]:
        lang_dir = os.path.join(SYNTHETIC_DIR, lang)
        if not os.path.exists(lang_dir):
            print(f"  [{lang}] No synthetic data found, skipping.")
            continue

        for split in ["train", "valid"]:
            dis_path = os.path.join(lang_dir, f"{split}.dis")
            flu_path = os.path.join(lang_dir, f"{split}.flu")

            if not os.path.exists(dis_path) or not os.path.exists(flu_path):
                print(f"  [{lang}/{split}] Files not found, skipping.")
                continue

            dis_lines, label_lines, skipped = generate_labels(dis_path, flu_path)
            print(f"  [{lang}/{split}] {len(dis_lines)} labeled (skipped {skipped})")

            if split == "train":
                all_train_dis.extend(dis_lines)
                all_train_labels.extend(label_lines)
            else:
                all_valid_dis.extend(dis_lines)
                all_valid_labels.extend(label_lines)

    # Split valid into valid + test
    valid_count = len(all_valid_dis)
    test_split = valid_count // 3  # ~1/3 for test
    test_dis = all_valid_dis[:test_split]
    test_labels = all_valid_labels[:test_split]
    valid_dis = all_valid_dis[test_split:]
    valid_labels = all_valid_labels[test_split:]

    # Write combined files
    for split, dis, labels in [
        ("train", all_train_dis, all_train_labels),
        ("valid", valid_dis, valid_labels),
        ("test", test_dis, test_labels),
    ]:
        with open(os.path.join(LABELED_DIR, f"{split}.dis"), "w", encoding="utf-8") as f:
            for line in dis:
                f.write(line + "\n")
        with open(os.path.join(LABELED_DIR, f"{split}.labels"), "w", encoding="utf-8") as f:
            for line in labels:
                f.write(line + "\n")
        print(f"  {split}: {len(dis)} sentences")

    total = len(all_train_dis) + len(valid_dis) + len(test_dis)
    print(f"\nTotal: {total} labeled sentences")
    print(f"Output: {LABELED_DIR}")
    print("Done!")
