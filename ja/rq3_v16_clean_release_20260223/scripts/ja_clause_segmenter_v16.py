"""
Japanese TextGrid Clause Segmentation V2

Japanese clause segmentation using GiNZA + UD dependencies with Vercellotti-style rules.
Processes TextGrid files to add clause tier for CAF analysis.

Key differences from English version:
- Uses GiNZA (default: ja_ginza_electra) for Japanese NLP
- Japanese-specific clause rules (te-form chains, tari-form, advcl with subordinators)
- Japanese filler handling (ãˆãƒ¼, ã‚ã®, etc.)
- Mora counting instead of syllable counting
- Neural disfluency detection (xlm-roberta-base, shared EN/JA model)

Rules: See ja/annotation/FROZEN_RULES.md for the full frozen rule set.

Usage:
    python ja_clause_segmenter.py -i textgrids/ -o clauses/
    python ja_clause_segmenter.py -i input.TextGrid -o output.TextGrid

Changelog:
    V2.1 (2026-02-14): Aligned with FROZEN_RULES.md
        - Added ãŸã‚Š-form detection (CHAIN_TE label, same verb+element rule as ã¦-form)
        - Added csubj handling (SUB_CSUBJ)
        - Note: ãŸã‚ã«/ã‚ˆã†ã«/ã¾ã§/å‰ã«/å¾Œã§/æ™‚ are in FROZEN_RULES but GiNZA
          does not parse them as mark children. They are caught by the advcl
          fallback path (verb+element â†’ SUB_ADVCL) when they have complements.
"""

import os
import sys
import re
import unicodedata
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass

try:
    import spacy
except ImportError:
    print("Please install spacy: pip install spacy")
    sys.exit(1)

try:
    from praatio import textgrid
    from praatio.utilities.constants import Interval
except ImportError:
    print("Please install praatio: pip install praatio")
    sys.exit(1)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForTokenClassification = None


# ==============================================================================
# DisfluencyDetector - Neural disfluency detection (xlm-roberta-base)
# Same model works for both English and Japanese.
# ==============================================================================

class DisfluencyDetector:
    """
    Detects disfluent words/morphemes using a fine-tuned xlm-roberta-base model.
    Labels each token as fluent (0) or disfluent (1).
    Works for both English and Japanese (multilingual model).
    """

    # Candidate default model paths (searched upward from script location)
    MODEL_PATH_CANDIDATES = (
        os.path.join("shared", "disfluency_detector", "model_v2", "final"),
        os.path.join("en", "disfluency_test", "l2_disfluency_detector", "model_v2", "final"),
    )

    def __init__(self, model_path: str = None):
        if torch is None:
            raise ImportError(
                "torch and transformers are required for DisfluencyDetector.\n"
                "Install: pip install torch transformers"
            )

        if model_path is None:
            # Auto-discover model by walking upward from script location.
            # This keeps the script runnable both in-repo and in copied release folders.
            script_dir = os.path.dirname(os.path.abspath(__file__))
            search_dir = script_dir
            found = None
            for _ in range(8):
                for rel in self.MODEL_PATH_CANDIDATES:
                    candidate = os.path.join(search_dir, rel)
                    if os.path.exists(candidate):
                        found = candidate
                        break
                if found is not None:
                    break
                parent = os.path.dirname(search_dir)
                if parent == search_dir:
                    break
                search_dir = parent

            if found is not None:
                model_path = found
            else:
                # Keep previous fallback behavior for error reporting.
                project_root = os.path.dirname(script_dir)
                model_path = os.path.join(project_root, self.MODEL_PATH_CANDIDATES[0])

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Disfluency model not found at: {model_path}\n"
                f"Expected one of:\n"
                f"  - <repo_root>/shared/disfluency_detector/model_v2/final\n"
                f"  - <repo_root>/en/disfluency_test/l2_disfluency_detector/model_v2/final"
            )

        print(f"Loading disfluency model from: {model_path}")
        self._tokenizer_fast = False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self._tokenizer_fast = bool(getattr(self.tokenizer, "is_fast", False))
        except Exception as fast_err:
            print(f"Fast tokenizer load failed ({fast_err}). Trying compatible fast conversion.")
            try:
                # Some older tokenizers stacks can't parse newer tokenizer.json.
                # For XLM-R family, rebuilding fast tokenizer from sentencepiece
                # avoids that parse path and keeps word_ids() support.
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=True, tokenizer_file=None
                )
                self._tokenizer_fast = bool(getattr(self.tokenizer, "is_fast", False))
            except Exception as fast_conv_err:
                print(f"Compatible fast conversion failed ({fast_conv_err}). Falling back to slow tokenizer.")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    self._tokenizer_fast = bool(getattr(self.tokenizer, "is_fast", False))
                except Exception as slow_err:
                    raise RuntimeError(
                        "Failed to load disfluency tokenizer (fast parse, fast conversion, and slow all failed).\n"
                        "If using slow tokenizer, ensure sentencepiece is installed:\n"
                        "  pip install sentencepiece\n"
                        "If converting to fast tokenizer fails with protobuf errors, install protobuf 3.20.x.\n"
                        f"Fast parse error: {fast_err}\n"
                        f"Fast conversion error: {fast_conv_err}\n"
                        f"Slow error: {slow_err}"
                    ) from slow_err
        self.model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=2)
        self.model.eval()
        mode = "fast" if self._tokenizer_fast else "slow"
        print(f"Disfluency model loaded. tokenizer_mode={mode}")

    def detect(self, words: List[str]) -> List[int]:
        """
        Predict disfluency labels for a list of words/morphemes.

        Args:
            words: List of tokens (morphemes for Japanese, words for English)

        Returns:
            List of labels: 0 = fluent, 1 = disfluent
        """
        if not words:
            return []

        if self._tokenizer_fast:
            inputs = self.tokenizer(
                words, is_split_into_words=True,
                return_tensors="pt", truncation=True
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
            preds = torch.argmax(logits, dim=2)[0].tolist()
            word_ids = inputs.word_ids()

            # Majority vote per word
            word_preds: Dict[int, List[int]] = {}
            for idx, wid in enumerate(word_ids):
                if wid is not None:
                    word_preds.setdefault(wid, []).append(preds[idx])

            labels = []
            for i in range(len(words)):
                votes = word_preds.get(i, [0])
                labels.append(1 if sum(votes) > len(votes) / 2 else 0)
        else:
            # Slow tokenizers don't provide word_ids(); score per word independently.
            labels = []
            special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
            for w in words:
                inputs = self.tokenizer(w, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                pred_ids = torch.argmax(logits, dim=2)[0].tolist()
                input_ids = inputs["input_ids"][0].tolist()
                token_preds = [p for tid, p in zip(input_ids, pred_ids) if tid not in special_ids]
                if not token_preds:
                    labels.append(0)
                else:
                    labels.append(1 if sum(token_preds) > len(token_preds) / 2 else 0)

        # Rule-based post-processing: catch what the neural model misses
        labels = self._postprocess_labels(words, labels)
        return labels

    @staticmethod
    def _postprocess_labels(words: List[str], labels: List[int]) -> List[int]:
        """Rule-based fixes for common disfluency patterns the model misses.
        
        1. Consecutive identical tokens â†’ mark earlier as disfluent
        2. Near-identical tokens (one has ãƒ¼ elongation) â†’ mark earlier as disfluent
        3. Triple+ repetition â†’ keep only the last occurrence
        """
        labels = list(labels)  # copy
        n = len(words)
        
        # Pass 1: consecutive identical tokens (e.g., è­¦å¯Ÿ è­¦å¯Ÿ)
        for i in range(n - 1):
            if labels[i] == 0 and labels[i + 1] == 0:
                w1 = words[i].replace('ãƒ¼', '')
                w2 = words[i + 1].replace('ãƒ¼', '')
                if w1 == w2 and len(w1) >= 2:
                    labels[i] = 1  # mark the first as disfluent
        
        # Pass 2: triple+ repetition â€” keep only the last
        i = 0
        while i < n - 2:
            base = words[i].replace('ãƒ¼', '')
            if len(base) >= 2:
                j = i + 1
                while j < n and words[j].replace('ãƒ¼', '') == base:
                    j += 1
                if j - i >= 3:  # 3+ consecutive identical
                    for k in range(i, j - 1):
                        labels[k] = 1  # keep only last
                    i = j
                    continue
            i += 1
        
        # Pass 3: elongated variant before correction
        # e.g., word ending with ãƒ¼ where next word is same without ãƒ¼
        for i in range(n - 1):
            if labels[i] == 0 and words[i].endswith('ãƒ¼') and len(words[i]) >= 2:
                base = words[i].rstrip('ãƒ¼')
                if words[i + 1].startswith(base):
                    labels[i] = 1
        
        # Pass 4: split-repetition detection
        # GiNZA may split a repeated word: è­¦å¯Ÿè­¦å¯Ÿ â†’ è­¦å¯Ÿ + è­¦ + å¯Ÿ
        # Detect when concat of subsequent tokens matches a previous token
        for i in range(n - 1):
            if labels[i] == 0 and len(words[i]) >= 2:
                # Try to match words[i] with concat of words[i+1], words[i+1:i+2], etc.
                concat = ''
                for j in range(i + 1, min(i + 6, n)):  # look ahead up to 5 tokens
                    concat += words[j]
                    if concat == words[i]:
                        # Split repetition found â€” mark the split parts as DIS
                        for k in range(i + 1, j + 1):
                            labels[k] = 1
                        break
                    if len(concat) > len(words[i]):
                        break
                
                # Also check reverse: words[i] + words[i+1] + ... == words[j]
                if i + 1 < n and labels[i] == 0:
                    concat = words[i]
                    for j in range(i + 1, min(i + 6, n)):  # look ahead up to 5 tokens
                        if concat == words[j] and len(concat) >= 2:
                            # words[i:j] are the split version, words[j] is the whole
                            # Mark the split parts as DIS
                            for k in range(i, j):
                                labels[k] = 1
                            break
                        concat += words[j]
                        if len(concat) > 10:
                            break
        
        # Pass 5: same-script prefix false start
        # e.g., ãƒã‚¹ before ãƒã‚¹ã‚±ãƒƒãƒˆ â€” word[i] is a prefix of word[i+1]
        # and word[i+1] is significantly longer (>=1.5x)
        for i in range(n - 1):
            if labels[i] == 0 and labels[i + 1] == 0:
                w1 = words[i]
                w2 = words[i + 1]
                if (len(w1) >= 2 and len(w2) >= len(w1) + 2
                        and w2.startswith(w1)):
                    labels[i] = 1

        # Pass 6: bridge auxiliaries between repeated content words.
        # Example: ã‚¸ãƒ£ãƒ³ãƒ—(DIS) ã— ã¾ã™ ã‚¸ãƒ£ãƒ³ãƒ—(FLU) -> mark ã—/ã¾ã™ as DIS.
        bridge_ok = {"ã—", "ã¾ã™", "ã¾ã—", "ãŸ", "ã ", "ã§ã™", "ã§ã—"}
        for i in range(n - 2):
            if labels[i] != 1:
                continue
            left = words[i].replace('ãƒ¼', '')
            if len(left) < 2:
                continue
            for j in range(i + 2, min(i + 6, n)):
                right = words[j].replace('ãƒ¼', '')
                if labels[j] == 0 and right == left:
                    between = words[i + 1:j]
                    if between and all(w in bridge_ok for w in between):
                        for k in range(i + 1, j):
                            labels[k] = 1
                    break
        

        # v15 Patch D: Protect まで particle from destruction
        for i_p in range(n - 1):
            if labels[i_p] == 1 and words[i_p] == "ま" and words[i_p + 1] == "で":
                labels[i_p] = 0

        return labels

    def clean(self, words: List[str]) -> Tuple[List[str], List[int], Dict]:
        """
        Remove disfluent words from a word list.

        Returns:
            (clean_words, removed_indices, info_dict)
        """
        labels = self.detect(words)
        clean_words = []
        removed_indices = []
        removed_words = []
        for i, (word, label) in enumerate(zip(words, labels)):
            if label == 1:
                removed_indices.append(i)
                removed_words.append(word)
            else:
                clean_words.append(word)

        info = {
            'original_length': len(words),
            'processed_length': len(clean_words),
            'disfluencies_removed': len(removed_indices),
            'removed_words': removed_words,
        }
        return clean_words, removed_indices, info


# ==============================================================================
# Japanese Clause Data Structure
# ==============================================================================

@dataclass(frozen=True)
class Clause:
    sent_index: int
    clause_type: str
    head_i: int
    token_indices: Tuple[int, ...]
    text: str


# ==============================================================================
# Japanese Clause Segmenter (GiNZA + Vercellotti-style rules)
# ==============================================================================

class JapaneseClauseSegmenter:
    """
    Rule layer over a Japanese UD dependency parse (GiNZA / spaCy).
    
    Implements Vercellotti & Hall (2024) inspired rules for Japanese:
    - Independent clause: sentence ROOT predicate (VERB/ADJ/NOUN)
    - Subordinate clause heads (advcl/ccomp/acl/xcomp) are clauses
    - Te-form chains are promoted to clause status only with overt complement/adjunct
    - Stance verbs like æ€ã† with ccomp child are tagged as MINOR
    """

    def __init__(
        self,
        nlp,
        debug: bool = False,
        element_deps: Optional[Set[str]] = None,
        minor_matrix_lemmas: Optional[Set[str]] = None,
        allow_dep_as_element: bool = False,
        disfluency_detector: Optional["DisfluencyDetector"] = None,
        apply_methodology_rules: bool = True,
    ):
        self.nlp = nlp
        self.debug = debug
        self._current_filename = None
        # Optional neural disfluency detector. When set, clause text will
        # exclude detected disfluent words (fillers, repetitions, false starts).
        self.disfluency_detector = disfluency_detector
        self.apply_methodology_rules = apply_methodology_rules

        # "Verb + element" deps used to promote borderline predicates to clause status
        self._element_deps = element_deps or {"obj", "iobj", "obl", "advmod", "nmod"}
        self._allow_dep_as_element = allow_dep_as_element

        # Stance list for "minor clause" tagging
        self._minor_matrix_lemmas = minor_matrix_lemmas or {"æ€ã†", "è€ƒãˆã‚‹", "æ„Ÿã˜ã‚‹"}

        # Common fillers / backchannels in Japanese speech transcripts
        self._filler_text = {
            "ãˆãƒ¼", "ãˆ", "ã‚ãƒ¼", "ã‚", "ã‚“ãƒ¼", "ã†ãƒ¼ã‚“", "ãˆã£ã¨", "ãˆãˆã¨",
            "ã¾ãƒ¼", "ã¾ã‚", "ãã®ãƒ¼", "ãã®", "ã‚ã®ãƒ¼", "ã‚ã®", "ã­ãƒ¼", "ã­",
            "ã†ã‚“", "ãµãƒ¼ã‚“", "ã¸ãƒ¼", "ã¯ã„", "ãˆãˆ",
        }
        # Normalize and broaden filler/backchannel matching.
        self._filler_text.update({"ã‚“", "ã†ã†ã‚“"})
        self._filler_text_norm = {self._normalize_surface_text(x) for x in self._filler_text}
        # Plain demonstratives are ambiguous in learner speech ("ãã®çŠ¬" vs filler "ãã®").
        self._ambiguous_filler_norm = {"ãã®", "ã‚ã®"}
        self._filler_char_only_re = re.compile(r"^[ã‚“ãƒ¼]+$")
        self._filler_backchannel_re = re.compile(
            r"^(?:"
            r"ã†+ã‚“+|"
            r"ãˆ+ãƒ¼*|"
            r"ã‚+ãƒ¼*|"
            r"ãˆã£?ã¨|"
            r"ã‚ã®ãƒ¼?|"
            r"ãã®ãƒ¼?|"
            r"ã¾+ãƒ¼?|"
            r"ã­+ãƒ¼?|"
            r"ã¯ã„+"
            r")$"
        )
        # Cached surface-profile analysis used for boundary cleanup.
        self._surface_profile_cache: Dict[str, Dict[str, bool]] = {}
        self._function_pos = {"AUX", "ADP", "PART", "SCONJ", "CCONJ", "PUNCT", "SYM"}
        self._function_deps = {"aux", "cop", "mark", "case", "punct", "cc", "fixed"}
        self._content_pos = {"VERB", "ADJ", "NOUN", "PROPN", "ADV", "NUM", "PRON"}

    @staticmethod
    def _normalize_surface_text(text: str) -> str:
        t = (text or "").strip().lower()
        t = t.replace("ã€€", "").replace(" ", "")
        t = t.replace("ã€œ", "ãƒ¼").replace("ï½ž", "ãƒ¼")
        t = re.sub(r"^[ã€ã€‚,.;:!?ï¼ï¼Ÿ]+|[ã€ã€‚,.;:!?ï¼ï¼Ÿ]+$", "", t)
        return t

    def _is_filler_surface(self, text: str) -> bool:
        t = self._normalize_surface_text(text)
        if not t:
            return False
        if t in self._filler_text_norm:
            return True
        if self._filler_char_only_re.fullmatch(t):
            return True
        if self._filler_backchannel_re.fullmatch(t):
            return True
        return False

    def _is_ambiguous_filler_surface(self, text: str) -> bool:
        t = self._normalize_surface_text(text)
        return t in self._ambiguous_filler_norm

    def _is_unambiguous_filler_surface(self, text: str) -> bool:
        t = self._normalize_surface_text(text)
        if not t or t in self._ambiguous_filler_norm:
            return False
        return self._is_filler_surface(t)

    def _is_filler_token(self, tok) -> bool:
        """Check if token is a filler/backchannel."""
        if self._is_unambiguous_filler_surface(tok.text):
            return True
        if self._is_ambiguous_filler_surface(tok.text):
            if tok.dep_ == "discourse":
                return True
            if tok.pos_ in {"INTJ", "SYM"}:
                return True
            if "ãƒ•ã‚£ãƒ©ãƒ¼" in (tok.tag_ or ""):
                return True
        if "ãƒ•ã‚£ãƒ©ãƒ¼" in (tok.tag_ or ""):
            return True
        if tok.dep_ == "discourse" and tok.pos_ in {"INTJ", "SYM", "ADJ"}:
            return True
        return False

    def _surface_profile(self, text: str) -> Dict[str, bool]:
        """Return a cached coarse profile for a surface word."""
        norm = self._normalize_surface_text(text)
        if not norm:
            return {"has_content": False, "all_function": True}
        cached = self._surface_profile_cache.get(norm)
        if cached is not None:
            return cached
        doc = self.nlp(norm)
        tokens = [t for t in doc if t.text.strip()]
        if not tokens:
            profile = {"has_content": False, "all_function": True}
        else:
            has_content = any(t.pos_ in self._content_pos for t in tokens)
            all_function = all(
                (t.pos_ in self._function_pos) or (t.dep_ in self._function_deps)
                for t in tokens
            )
            profile = {"has_content": has_content, "all_function": all_function}
        self._surface_profile_cache[norm] = profile
        return profile

    def _is_function_like_surface(self, text: str) -> bool:
        """True when a surface form is function-like (aux/particle/etc)."""
        if self._is_unambiguous_filler_surface(text):
            return True
        profile = self._surface_profile(text)
        return profile["all_function"] and not profile["has_content"]

    def _is_content_surface(self, text: str) -> bool:
        """True when a surface form contains lexical content."""
        norm = self._normalize_surface_text(text)
        if not norm or self._is_unambiguous_filler_surface(norm):
            return False
        profile = self._surface_profile(norm)
        if profile["has_content"]:
            return True
        return not profile["all_function"]

    def _is_predicate(self, tok) -> bool:
        """Check if token can be a predicate (ROOT-level)."""
        if self._is_filler_token(tok):
            return False
        return tok.pos_ in {"VERB", "ADJ", "NOUN"}

    def _is_subordinate_predicate(self, tok) -> bool:
        """
        For embedded/subordinate clause heads, be stricter than ROOT:
        - allow VERB/ADJ freely
        - allow NOUN only for ROOT-like copula-less predicates (handled elsewhere)
        """
        if self._is_filler_token(tok):
            return False
        if tok.pos_ == "VERB":
            # Reject noun-like tokens misparsed as VERB, but keep
            # sahen-style verbal predicates (e.g., ã³ã£ãã‚Š + ã— + ãŸ).
            tag = tok.tag_ or ""
            noun_like = ("åè©ž" in tag) or ("å›ºæœ‰åè©ž" in tag)
            if noun_like:
                aux_children = [ch for ch in tok.children if ch.pos_ == "AUX"]
                sahen_like = ("ã‚µå¤‰" in tag) or ("sahen" in tag.lower())
                if not (aux_children or sahen_like):
                    if self.debug:
                        print(f"Rejecting misparsed VERB (tag={tag}): {tok.text}")
                    return False
            # Reject if it looks like a proper noun (katakana name)
            if tok.text and all(c in "ã‚¢ã‚¤ã‚¦ã‚¨ã‚ªã‚«ã‚­ã‚¯ã‚±ã‚³ã‚µã‚·ã‚¹ã‚»ã‚½ã‚¿ãƒãƒ„ãƒ†ãƒˆãƒŠãƒ‹ãƒŒãƒãƒŽãƒãƒ’ãƒ•ãƒ˜ãƒ›ãƒžãƒŸãƒ ãƒ¡ãƒ¢ãƒ¤ãƒ¦ãƒ¨ãƒ©ãƒªãƒ«ãƒ¬ãƒ­ãƒ¯ãƒ²ãƒ³ã‚¡ã‚£ã‚¥ã‚§ã‚©ãƒƒãƒ£ãƒ¥ãƒ§ãƒ¼" for c in tok.text):
                if len(tok.text) >= 2:  # Katakana names are usually 2+ chars
                    if self.debug:
                        print(f"Rejecting katakana name misparsed as VERB: {tok.text}")
                    return False
            return True
        if tok.pos_ == "ADJ":
            # Restrict to true predicative adjectives (i-adjectives)
            if "å½¢å®¹è©ž" in (tok.tag_ or ""):
                return True
            # Allow adjective-like predicates when supported by auxiliary/copula
            cop_like_text = {"ã ", "ã§ã™", "ã ã£ãŸ", "ã§ã—ãŸ", "ã˜ã‚ƒ", "ãªã„", "ãª"}
            for ch in tok.children:
                if ch.pos_ == "AUX" and ch.text in cop_like_text:
                    return True
            return False
        return False

    def _has_tari_mark(self, tok) -> bool:
        """Check if token has tari-form marking (ãŸã‚Š).
        GiNZA may parse ãŸã‚Š as a mark child or as part of the verb form."""
        for child in tok.children:
            if child.dep_ == "mark" and child.text == "ãŸã‚Š":
                return True
        # Also check if the verb text itself ends in ãŸã‚Š
        if tok.text.endswith("ãŸã‚Š") or tok.text.endswith("ã ã‚Š"):
            return True
        return False

    def _has_te_mark(self, tok) -> bool:
        """Check if token has te-form marking (ã¦/ã§).
        Returns False for ã¦ã‹ã‚‰ (sequential 'after doing'), which is a subordinator."""
        for child in tok.children:
            if child.dep_ == "mark" and child.text in {"ã¦", "ã§"}:
                # Check for ã¦ã‹ã‚‰: ã¦ + ã‹ã‚‰ as sibling â†’ subordinator, not te-chain
                has_kara = any(
                    sib.text == "ã‹ã‚‰" and sib.dep_ in ("case", "mark")
                    for sib in tok.children if sib is not child
                )
                if has_kara:
                    return False  # Let _has_explicit_subordinator handle ã¦ã‹ã‚‰
                return True
        return False

    def _has_explicit_subordinator(self, tok) -> bool:
        """Detect common overt subordinators."""
        for child in tok.children:
            # Standard mark-dependency subordinators
            if child.dep_ == "mark":
                if child.text in {"ã‹ã‚‰", "ãªãŒã‚‰", "ã¤ã¤", "ã‘ã©", "ã‘ã‚Œã©", "ãŒ", "ãªã‚‰", "ã°", "ã¨", "ã—"}:
                    return True
                # ã®ã§ / ã®ã« often come as mark 'ã®' + fixed 'ã§'/'ã«'
                if child.text == "ã®":
                    for g in child.children:
                        if g.dep_ == "fixed" and g.text in {"ã§", "ã«"}:
                            return True
                # ã¦ã‹ã‚‰: ã¦(mark) + ã‹ã‚‰(case) â†’ subordinator "after doing"
                if child.text in {"ã¦", "ã§"}:
                    has_kara = any(
                        sib.text == "ã‹ã‚‰" and sib.dep_ in ("case", "mark")
                        for sib in tok.children if sib is not child
                    )
                    if has_kara:
                        return True
            # GiNZA sometimes parses ãªã‚‰ as AUX (not mark)
            if child.dep_ == "aux" and child.text == "ãªã‚‰":
                return True
        return False

    def _has_complement_or_adjunct(self, tok) -> bool:
        """
        Strict "verb + element" heuristic.
        NOTE: nsubj is NOT counted as the "extra element" (shared/implied subjects).
        """
        for child in tok.children:
            if self._is_filler_token(child):
                continue
            if child.dep_ in self._element_deps:
                return True
            if self._allow_dep_as_element and child.dep_ == "dep":
                if child.pos_ in {"NOUN", "PROPN", "ADV", "ADJ", "VERB"}:
                    return True
        return False

    def _token_depth(self, tok) -> int:
        """Compute syntactic depth of token."""
        depth = 0
        cur = tok
        while cur.head is not cur:
            depth += 1
            cur = cur.head
            if depth > 200:
                break
        return depth

    def _find_clause_heads(self, sent) -> List[Tuple[object, str]]:
        """Return list of (head_token, clause_type) for a single sentence span."""
        heads: List[Tuple[object, str]] = []

        for tok in sent:
            if not self._is_predicate(tok):
                continue

            dep = tok.dep_

            if dep == "ROOT":
                ctype = "IND"
                # Tag stance/matrix verbs as MINOR when they take a ccomp
                if tok.pos_ == "VERB" and tok.lemma_ in self._minor_matrix_lemmas and any(
                    ch.dep_ == "ccomp" for ch in tok.children
                ):
                    ctype = "MINOR"
                heads.append((tok, ctype))
                continue
            
            # Handle broken parses: VERB with dep="dep" pointing to NOUN ROOT
            # This often happens in unpunctuated speech where GiNZA misparses
            if dep == "dep" and tok.pos_ == "VERB":
                # Check if this verb has its own subject - it's likely a real clause
                has_own_subject = any(ch.dep_ == "nsubj" for ch in tok.children)
                has_content = self._has_complement_or_adjunct(tok) or has_own_subject
                if has_content:
                    heads.append((tok, "IND"))
                    if self.debug:
                        print(f"Promoting dep=dep VERB as IND: {tok.text}")
                continue

            if dep in {"acl"}:
                # Relative/adnominal clause
                if self._is_subordinate_predicate(tok):
                    heads.append((tok, "SUB_REL"))
                continue

            if dep in {"ccomp"}:
                # Content/complement clause
                if self._is_subordinate_predicate(tok) or tok.pos_ == "NOUN":
                    heads.append((tok, "SUB_CCOMP"))
                continue

            if dep in {"xcomp"}:
                # Japanese xcomp is rarer
                if self._is_subordinate_predicate(tok):
                    heads.append((tok, "SUB_XCOMP"))
                continue

            if dep == "csubj":
                if self._is_subordinate_predicate(tok):
                    heads.append((tok, "SUB_CSUBJ"))
                continue

            if dep == "advcl":
                if not self._is_subordinate_predicate(tok):
                    continue
                # Check for explicit subordinator FIRST (ã‹ã‚‰, ã‘ã©, ã®ã§, etc.)
                # These take priority over te-form marking
                if self._has_explicit_subordinator(tok):
                    heads.append((tok, "SUB_ADVCL"))
                elif self._has_te_mark(tok):
                    # Strict Vercellotti-style promotion for te-chains
                    if self._has_complement_or_adjunct(tok):
                        heads.append((tok, "CHAIN_TE"))
                    else:
                        if self.debug:
                            print(f"SKIP te-chain (no element): {tok.text} / {sent.text}")
                elif self._has_tari_mark(tok):
                    # Tari-form: same verb+element rule as te-form (FROZEN_RULES Rule 5)
                    if self._has_complement_or_adjunct(tok):
                        heads.append((tok, "CHAIN_TE"))
                    else:
                        if self.debug:
                            print(f"SKIP tari-chain (no element): {tok.text} / {sent.text}")
                else:
                    # For other advcl: verb+element required
                    if self._has_complement_or_adjunct(tok):
                        heads.append((tok, "SUB_ADVCL"))
                    else:
                        if self.debug:
                            print(f"SKIP advcl (no mark/element): {tok.text} / {sent.text}")
                continue

            if dep == "conj":
                # Coordinated predicates: promote only if it has its own element
                if self._is_subordinate_predicate(tok) and self._has_complement_or_adjunct(tok):
                    heads.append((tok, "COORD"))
                continue

            if dep == "parataxis":
                if self._is_subordinate_predicate(tok):
                    heads.append((tok, "PARA"))
                continue

        # Deduplicate
        seen = set()
        out: List[Tuple[object, str]] = []
        for tok, ctype in heads:
            if tok.i in seen:
                continue
            seen.add(tok.i)
            out.append((tok, ctype))
        return out

    def _collect_clause_tokens(
        self,
        head_tok,
        sent,
        excluded: Set[int],
        clause_head_indices: Set[int],
    ) -> Set[int]:
        """Collect a token set for this clause, staying within the sentence."""
        sent_start = sent.start
        sent_end = sent.end

        collected: Set[int] = set()

        def in_sent(i: int) -> bool:
            return sent_start <= i < sent_end

        def collect(tok, depth: int = 0) -> None:
            if depth > 60:
                return
            if tok.i in excluded or not in_sent(tok.i):
                return

            collected.add(tok.i)

            for child in tok.children:
                if child.i in excluded or not in_sent(child.i):
                    continue
                # Avoid swallowing other clause heads
                if child.i in clause_head_indices and child.i != head_tok.i:
                    continue
                # For content clauses, exclude quotative particle ã¨
                if head_tok.dep_ == "ccomp" and child.dep_ == "case" and child.text == "ã¨":
                    continue
                # CRITICAL: Don't swallow children that have their own nsubj
                # This indicates a separate clause structure
                if child.dep_ == "dep" and child.pos_ == "VERB":
                    child_has_nsubj = any(gc.dep_ == "nsubj" for gc in child.children)
                    if child_has_nsubj:
                        if self.debug:
                            print(f"Stopping at VERB with own nsubj: {child.text}")
                        continue
                collect(child, depth + 1)

        collect(head_tok)

        # Add "matrix glue" tokens for minor/matrix verbs
        if head_tok.dep_ == "ROOT" and head_tok.lemma_ in self._minor_matrix_lemmas:
            for ch in head_tok.children:
                if ch.dep_ == "ccomp":
                    for g in ch.children:
                        if g.dep_ == "case" and g.text == "ã¨" and in_sent(g.i) and g.i not in excluded:
                            collected.add(g.i)

        return collected

    def _is_functionish_token(self, tok) -> bool:
        """Heuristic: token is function-like (boundary glue, not lexical core)."""
        if self._is_filler_token(tok):
            return True
        if tok.pos_ in self._function_pos or tok.dep_ in self._function_deps:
            return True
        return self._is_function_like_surface(tok.text)

    def _prune_disconnected_function_runs(self, tokens: Set[int], head_tok) -> Set[int]:
        """
        Drop disconnected function-only islands that can leak into a clause span.
        Example: sentence-initial `ã§` attached to ROOT while real predicate is later.
        """
        if not tokens:
            return tokens
        idxs = sorted(tokens)
        runs: List[List[int]] = []
        cur = [idxs[0]]
        for i in idxs[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
        runs.append(cur)

        head_run = None
        for r in runs:
            if head_tok.i in r:
                head_run = r
                break
        if head_run is None:
            return tokens

        keep = set(head_run)
        for r in runs:
            if r is head_run:
                continue
            # Keep non-function runs (defensive), drop function-only glue runs.
            if not all(self._is_functionish_token(head_tok.doc[i]) for i in r):
                keep.update(r)
        return keep

    def segment_sentence(self, sent, sent_index: int) -> List[Clause]:
        """Segment one spaCy sentence span into non-overlapping clause strings."""
        clause_heads = self._find_clause_heads(sent)
        if not clause_heads:
            return []

        clause_head_indices = {tok.i for tok, _ in clause_heads}

        # Two-pass: take deeper heads first
        scored = []
        for tok, ctype in clause_heads:
            scored.append((self._token_depth(tok), tok.i, tok, ctype))
        scored.sort(key=lambda x: (-x[0], x[1]))

        excluded: Set[int] = set()
        clauses: List[Clause] = []

        for _depth, _i, head_tok, ctype in scored:
            tokens = self._collect_clause_tokens(head_tok, sent, excluded, clause_head_indices)
            tokens = self._prune_disconnected_function_runs(tokens, head_tok)
            if not tokens:
                continue
            excluded.update(tokens)

            idxs = tuple(sorted(tokens))
            text = "".join(head_tok.doc[i].text_with_ws for i in idxs).strip()

            clauses.append(
                Clause(
                    sent_index=sent_index,
                    clause_type=ctype,
                    head_i=head_tok.i,
                    token_indices=idxs,
                    text=text,
                )
            )

        # Return in surface order
        clauses.sort(key=lambda c: c.token_indices[0])
        return clauses

    def _validate_clauses(self, clauses: List[Tuple]) -> List[Tuple]:
        """
        Post-process clauses to filter invalid ones and fix issues.
        
        Vercellotti rules applied:
        1. Clause must have predicate (verb/adj/noun) + element
        2. Te-form chains need own complement/adjunct
        3. Filter fragments (<4 chars)
        4. Truncate very long clauses (>60 chars) at space boundaries
        """
        validated = []
        
        for start_idx, end_idx, ctype, ctext, indices in clauses:
            # Clean text for length check
            clean = ctext.replace(' ', '').replace('ã€€', '')
            
            # Skip fragments (< 3 chars after removing spaces)
            # BUT: keep MINOR clauses and clauses with clear predicates even if short
            if len(clean) < 3:
                if self.debug:
                    print(f"[DEBUG] Filtered fragment: '{ctext}'")
                continue
            
            # For 3-char clauses, only keep if it has a clear predicate (verb/adj)
            if len(clean) < 4 and ctype not in ['MINOR', 'IND']:
                doc = self.nlp(ctext)
                has_clear_pred = any(t.pos_ in ('VERB', 'ADJ') for t in doc)
                if not has_clear_pred:
                    if self.debug:
                        print(f"[DEBUG] Filtered short non-predicate: '{ctext}'")
                    continue
            
            # Skip if just particles/punctuation/fillers (but keep MINOR)
            skip_chars = 'ã‚’ã«ãŒã§ã¨ã¯ã‚‚ã¸ã‹ã‚‰ã¾ã§ã‚ˆã‚Šã‚„ã®ã‹ãªã­ã‚ˆã‚ã•ãžã£ã¦ã‘ã©ã‘ã‚Œã©ã°ãŸã‚‰ã€ã€‚ãƒ¼ã£'
            if all(c in skip_chars for c in clean) and ctype != 'MINOR':
                if self.debug:
                    print(f"[DEBUG] Filtered particles-only: '{ctext}'")
                continue
            
            # Skip clauses that are entirely meta-speech
            if self._is_meta_clause(clean):
                if self.debug:
                    print(f"[DEBUG] Filtered meta clause: '{ctext}'")
                continue
            
            # Strip trailing meta from clause text (e.g., ...ã—ã¾ã—ãŸä»¥ä¸Šã§ã™)
            stripped = self._META_TAIL.sub('', ctext)
            if stripped != ctext and len(stripped.replace(' ', '').replace('ã€€', '')) >= 3:
                ctext = stripped.strip()
                clean = ctext.replace(' ', '').replace('ã€€', '')
            
            # Parse to check for predicate
            doc = self.nlp(ctext)
            # Reject misparsed VERB (pos=VERB but tag=åè©ž*) â€” common with disfluent L2
            has_verb = any(t.pos_ == 'VERB' and 'åè©ž' not in (t.tag_ or '') for t in doc)
            has_adj = any(t.pos_ == 'ADJ' for t in doc)
            has_aux = any(t.pos_ == 'AUX' and t.lemma_ not in ('ã ', 'ã§ã™') for t in doc)
            has_noun_pred = any(t.pos_ == 'NOUN' and t.dep_ == 'ROOT' for t in doc)
            has_predicate = has_verb or has_adj or has_aux or has_noun_pred
            
            # For CHAIN_TE, check it has complement/adjunct (Vercellotti rule)
            # No length threshold: bare te-form is never a clause regardless of length
            # NOTE: nsubj excluded â€” matches _has_complement_or_adjunct (shared subjects don't count)
            if ctype == 'CHAIN_TE':
                has_element = any(t.dep_ in ('obj', 'iobj', 'obl', 'advmod', 'nmod') for t in doc)
                if not has_element:
                    if self.debug:
                        print(f"[DEBUG] Filtered bare te-form: '{ctext}'")
                    continue
            
            # SUB_REL bare-adjective filter: a single adnominal adjective (å¹¸ã›ãª, ã‹ã‚ã„ãã†ãª)
            # is just a modifier, not a clause per Vercellotti's "predicate + element" rule.
            if ctype == 'SUB_REL' and (has_adj and not has_verb):
                has_element = any(t.dep_ in ('obj', 'iobj', 'obl', 'advmod', 'nmod') for t in doc)
                if not has_element and len(clean) < 15:
                    if self.debug:
                        print(f"[DEBUG] Filtered bare-adjective SUB_REL: '{ctext}'")
                    continue
            
            # Skip clauses without any predicate (violates Vercellotti)
            if not has_predicate and len(clean) < 20:
                if self.debug:
                    print(f"[DEBUG] Filtered no-predicate: '{ctext}'")
                continue
            
            # Sanity check: NOUN-ROOT predicate with case-marked arguments
            # is likely a fragment (broken verb), not a copula-less predicate.
            # Real copula-less: ä»Šæ—¥ã¯ä¼‘ã¿ (topic + noun) â€” no obj/obl markers.
            # Fragment: ãƒ”ã‚¯ãƒ‹ãƒƒã‚¯ã‚’ã¯ã˜ (obj + truncated verb) â€” has ã‚’.
            if has_noun_pred and not has_verb and not has_adj:
                # Only flag as fragment if there are true object/goal markers,
                # NOT topic/obl with ã¯ (which is legitimate for copula-less: ä»Šæ—¥ã¯ä¼‘ã¿ã§ã™)
                has_obj_case = any(
                    t.dep_ in ('obj', 'iobj') or
                    (t.dep_ == 'case' and t.text == 'ã‚’' and t.head.dep_ != 'ROOT')
                    for t in doc
                )
                if has_obj_case:
                    if self.debug:
                        print(f"[DEBUG] Filtered noun-ROOT with obj/obl (likely fragment): '{ctext}'")
                    continue
            
            # Truncate very long clauses at space boundaries (max 60 chars)
            if len(clean) > 60:
                # Find last space before 60 chars
                parts = ctext.split(' ')
                truncated = []
                current_len = 0
                for part in parts:
                    part_clean = part.replace('ã€€', '')
                    if current_len + len(part_clean) > 55:
                        break
                    truncated.append(part)
                    current_len += len(part_clean)
                
                if truncated:
                    ctext = ' '.join(truncated)
                    if self.debug:
                        print(f"[DEBUG] Truncated long clause to: '{ctext[:40]}...'")
            
            validated.append((start_idx, end_idx, ctype, ctext, indices))
        
        # Post-pass: split long clauses at ã¦/ã§-form or ã¯-topic boundaries
        final = []
        for start_idx, end_idx, ctype, ctext, indices in validated:
            clean = ctext.replace(' ', '').replace('ã€€', '')
            if len(clean) > 25:
                splits = self._split_long_clause(ctext, ctype, indices)
                if splits:
                    final.extend(splits)
                    continue
            final.append((start_idx, end_idx, ctype, ctext, indices))
        
        return final
    
    def _split_long_clause(self, ctext: str, ctype: str, indices: list):
        """Split a long clause at ã¦/ã§-form verb or ã¯-topic change.
        
        Priority:
          1. ã¦/ã§-form verb boundary (Vercellotti CHAIN_TE rule)
          2. ã¯-topic change (subject change)
        
        Returns list of sub-clause tuples, or None if no split needed.
        """
        doc = self.nlp(ctext)
        
        # --- Strategy 1: Split at ã¦/ã§-form boundary ---
        # Look for ã¦/ã§ SCONJ markers (dep=mark) â€” reliably detected even
        # when GiNZA misparses the verb in broken L2 text.
        # Also check verbs ending in ã¦/ã§ directly.
        te_split = None
        for tok in doc:
            split_after = None
            
            # Pattern A: ã¦/ã§ as separate SCONJ token with dep=mark
            if tok.text in ('ã¦', 'ã§') and tok.pos_ == 'SCONJ' and tok.dep_ == 'mark':
                split_after = tok.i + 1
            
            # Pattern B: verb ending in ã¦/ã§ (e.g., é£Ÿã¹ã¦ as one token)
            elif tok.pos_ == 'VERB' and tok.text.endswith(('ã¦', 'ã§')):
                split_after = tok.i + 1
            
            if split_after is not None:
                left_len = split_after
                right_len = len(doc) - split_after
                if left_len >= 3 and right_len >= 3:
                    te_split = split_after
                    break  # take the first valid ã¦/ã§ split
        
        if te_split:
            left_text = ''.join(t.text_with_ws for t in doc[:te_split]).strip()
            right_text = ''.join(t.text_with_ws for t in doc[te_split:]).strip()
            if left_text and right_text:
                left_indices = indices[:te_split] if indices else []
                right_indices = indices[te_split:] if indices else []
                return [
                    (0, te_split, 'CHAIN_TE', left_text, left_indices),
                    (te_split, len(doc), ctype, right_text, right_indices),
                ]
        
        # --- Strategy 2: Split at ã¯-topic change ---
        wa_positions = []
        for tok in doc:
            if tok.text == 'ã¯' and tok.dep_ == 'case':
                head = tok.head
                if head.pos_ in ('NOUN', 'PRON', 'PROPN'):
                    wa_positions.append((tok.i, head.i))
        
        if len(wa_positions) >= 2:
            _, second_head_i = wa_positions[1]
            split_at = second_head_i
            for child in doc[second_head_i].subtree:
                if child.i < split_at:
                    split_at = child.i
            
            if split_at >= 3 and len(doc) - split_at >= 3:
                left_text = ''.join(t.text_with_ws for t in doc[:split_at]).strip()
                right_text = ''.join(t.text_with_ws for t in doc[split_at:]).strip()
                if left_text and right_text:
                    left_indices = indices[:split_at] if indices else []
                    right_indices = indices[split_at:] if indices else []
                    return [
                        (0, split_at, ctype, left_text, left_indices),
                        (split_at, len(doc), 'IND', right_text, right_indices),
                    ]
        
        return None

    def segment_utterance(self, text: str, sent_index: int = 0) -> List[Clause]:
        """Segment a single utterance as one sentence span."""
        doc = self.nlp(text)
        sent = doc[:]  # force one-sentence view
        return self.segment_sentence(sent, sent_index=sent_index)

    def _resplit_long_sentences(self, doc, sents) -> list:
        """Re-split long sentences at sentence-final verb forms.
        
        When text has no punctuation (common after disfluency removal),
        GiNZA may fail to detect sentence boundaries. This finds obvious
        sentence-final forms (ã¾ã—ãŸ, ã§ã™, etc.) and splits there.
        
        GiNZA tokenization:
          ã¾ã—ãŸ â†’ ã¾ã—(AUX,lemma=ã¾ã™) + ãŸ(AUX,lemma=ãŸ)
          ã¾ã›ã‚“ â†’ ã¾ã›(AUX,lemma=ã¾ã™) + ã‚“(AUX,lemma=ã¬)
          ã§ã™   â†’ ã§ã™(AUX,lemma=ã§ã™)
          ã§ã—ãŸ â†’ ã§ã—(AUX,lemma=ã§ã™) + ãŸ(AUX,lemma=ãŸ)
        """
        # Strong connectives: always block a split
        # NOTE: ã§ is NOT here â€” after ã¾ã—ãŸ/ã§ã™, ã§ is almost always
        #       conjunction "and then", not connective particle
        CONNECTIVE = {'ã¦', 'ã‘ã©', 'ã‘ã‚Œã©', 'ã®ã§', 'ã‹ã‚‰',
                      'ãŸã‚Š', 'ã—', 'ã°', 'ã¨', 'ã®ã«', 'ãªãŒã‚‰', 'ã‚‚ã®ã®',
                      'ã¦ã‚‚', 'ã§ã‚‚', 'ã¨ã“ã‚', 'ã®ã¯'}
        # ãŒ is special: block split only if the remainder is short
        # (ã¾ã—ãŸãŒ + short subordinate = one clause;
        #  ã¾ã—ãŸãŒ + long independent = two clauses)
        
        new_sents = []
        for sent in sents:
            sent_text = sent.text.replace(' ', '').replace('\u3000', '')
            if len(sent_text) <= 20:
                new_sents.append(sent)
                continue
            
            split_points = []
            for tok in sent:
                # ----------------------------------------------------------
                # Pattern 1: AUX lemma=ã¾ã™ (ã¾ã—ãŸ / ã¾ã›ã‚“ / ã¾ã™)
                # ----------------------------------------------------------
                if tok.pos_ == 'AUX' and tok.lemma_ == 'ã¾ã™':
                    split_after = tok.i
                    peek = tok.i + 1
                    # Include ãŸ/ã  (ã¾ã—ãŸ) or ã‚“ (ã¾ã›ã‚“)
                    if peek < len(doc) and doc[peek].pos_ == 'AUX':
                        if doc[peek].lemma_ in ('ãŸ', 'ã ', 'ã¬'):
                            split_after = peek
                    
                    check_idx = split_after + 1
                    if check_idx >= len(doc) or check_idx >= sent.end:
                        continue
                    next_tok = doc[check_idx]
                    if next_tok.text in CONNECTIVE:
                        continue
                    # ãŒ: only block if remainder is short (subordinate clause)
                    if next_tok.text == 'ãŒ':
                        remainder_len = sent.end - check_idx
                        if remainder_len < 8:
                            continue
                    left_len = split_after + 1 - sent.start
                    right_len = sent.end - (split_after + 1)
                    if left_len >= 3 and right_len >= 3:
                        split_points.append(split_after + 1)
                
                # ----------------------------------------------------------
                # Pattern 2: AUX lemma=ã§ã™ (ã§ã™ / ã§ã—ãŸ)
                # ----------------------------------------------------------
                if tok.pos_ == 'AUX' and tok.lemma_ == 'ã§ã™':
                    split_after = tok.i
                    peek = tok.i + 1
                    if peek < len(doc) and doc[peek].pos_ == 'AUX' and doc[peek].lemma_ in ('ãŸ', 'ã '):
                        split_after = peek
                    
                    check_idx = split_after + 1
                    if check_idx >= len(doc) or check_idx >= sent.end:
                        continue
                    next_tok = doc[check_idx]
                    if next_tok.text in CONNECTIVE:
                        continue
                    if next_tok.text == 'ãŒ':
                        remainder_len = sent.end - check_idx
                        if remainder_len < 8:
                            continue
                    left_len = split_after + 1 - sent.start
                    right_len = sent.end - (split_after + 1)
                    if left_len >= 3 and right_len >= 3:
                        split_points.append(split_after + 1)
                
                # ----------------------------------------------------------
                # Pattern 3: Plain past ãŸ/ã  (not after ã¾ã™/ã§ã™)
                # Only split when followed by clear new-sentence start
                # ----------------------------------------------------------
                if tok.pos_ == 'AUX' and tok.lemma_ in ('ãŸ', 'ã ') and tok.text in ('ãŸ', 'ã '):
                    prev_idx = tok.i - 1
                    if prev_idx >= 0 and doc[prev_idx].pos_ == 'AUX' and doc[prev_idx].lemma_ in ('ã¾ã™', 'ã§ã™'):
                        continue
                    
                    check_idx = tok.i + 1
                    if check_idx >= len(doc) or check_idx >= sent.end:
                        continue
                    next_tok = doc[check_idx]
                    if next_tok.text in CONNECTIVE or next_tok.text == 'ãŒ':
                        continue
                    if next_tok.pos_ in ('NOUN', 'PRON', 'DET', 'CCONJ', 'SCONJ', 'ADV'):
                        left_len = tok.i + 1 - sent.start
                        right_len = sent.end - (tok.i + 1)
                        if left_len >= 4 and right_len >= 4:
                            split_points.append(tok.i + 1)
            

                # v15 Pattern 4: Split at subordinator boundary
                if tok.pos_ in ("SCONJ", "CCONJ") and tok.dep_ in ("mark", "cc", "case"):
                    _sub_surfaces = {
                        "から", "ので", "けど", "けれど",
                        "けれども", "たら", "ば",
                        "のに", "ながら",
                    }
                    if tok.text in _sub_surfaces:
                        split_after_sub = tok.i + 1
                        if split_after_sub < len(doc) and split_after_sub < sent.end:
                            left_len_s = split_after_sub - sent.start
                            right_len_s = sent.end - split_after_sub
                            if left_len_s >= 5 and right_len_s >= 5:
                                left_has_pred = any(
                                    doc[k].pos_ in ("VERB", "AUX", "ADJ")
                                    for k in range(sent.start, split_after_sub)
                                )
                                if left_has_pred:
                                    split_points.append(split_after_sub)

            if not split_points:
                new_sents.append(sent)
                continue
            
            # Create sub-spans
            prev = sent.start
            for sp in split_points:
                if sp > prev:
                    new_sents.append(doc[prev:sp])
                prev = sp
            if prev < sent.end:
                new_sents.append(doc[prev:sent.end])
        
        return new_sents

    def segment_text(self, text: str) -> List[Clause]:
        """Segment text using doc.sents, with re-splitting for long unpunctuated sentences."""
        doc = self.nlp(text)
        sents = list(doc.sents)

        # Re-split any overly long sentences at sentence-final forms
        sents = self._resplit_long_sentences(doc, sents)
        
        out: List[Clause] = []
        for si, sent in enumerate(sents):
            out.extend(self.segment_sentence(sent, sent_index=si))
        return out

    # Participant ID pattern (ï¼£ï¼£ï¼¨03, ï¼ªï¼ªï¼¥07, etc.)
    _PARTICIPANT_ID = re.compile(
        r'(ï¼£ï¼£ï¼¨|ï¼£ï¼£ï¼´|ï¼ªï¼ªï¼¥|ï¼ªï¼ªï¼¡)(ã®)?[ï¼-ï¼™0-9A-Zï½-ï½š]+'
        r'[\sã€€]*(ç•ª)?[\sã€€]*(ã§ãƒ¼ã™|ã§ã™)?[\sã€€]*'
    )
    _META_END = re.compile(
        r'('
        r'ä»¥ä¸Š(ã§ã™|ã§ãƒ¼ã™|ã§ã£ã™)?'
        r'|ã‚ã‚ŠãŒã¨(ã†(ã”ã–ã„ã¾ã™|ã”ã–ã„ã¾ã—ãŸ)?)?'
        r'|ã¯ã„(çµ‚ã‚ã‚Š|ã„ã„ã§ã™(ã‹|ã‚ˆ)?)?'
        r'|çµ‚ã‚ã‚Š(ã§ã™(ã‹)?|ã¾ã—ãŸ)?'
        r'|ã‚ªãƒƒã‚±ãƒ¼'
        r'|ã‚ã‹ã‚Šã¾ã—ãŸ(å’³)?'
        r'|ã¯ã„ã‚ã‹ã‚Š'
        r'|ã„ã„ã§ã™(ã‹|ã‚ˆ)?'
        r'|ç¬‘'
        r')+[\sã€€]*$'
    )

    # Clause-level meta-speech patterns (entire clause is meta)
    _META_CLAUSE = re.compile(
        r'^[\sã€€]*(ã¾|ã¯ã„|ãˆãƒ¼|ã‚ãƒ¼)*[\sã€€]*('
        r'ä»¥ä¸Š(ã§ã™|ã§ãƒ¼ã™|ã§ã£ã™)?(ã¯ã„)?'
        r'|çµ‚ã‚ã‚Š(ã§ã™|ã¾ã—ãŸ)?(ã¯ã„|ã‹)?'
        r'|ã‚ã‚ŠãŒã¨(ã†(ã”ã–ã„ã¾ã™|ã”ã–ã„ã¾ã—ãŸ)?)?'
        r'|(ã¯ã„)?ã„ã„ã§ã™(ã‹|ã‚ˆ)?'
        r'|ã§ã”ã–ã„ã¾ã™'
        r'|ã‚ªãƒƒã‚±ãƒ¼'
        r'|ã“ã‚Œã‹ã‚‰.{0,6}(ãŠé¡˜ã„|ã­ãŒã—)'
        r'|ã‚ˆã‚ã—ããŠé¡˜ã„'
        r'|ã¯ã„ã‚ã‚ŠãŒ'
        r'|è©±ã—ã¦.{0,8}(ãã ã•ã„|è¨€ã£ã¦)'
        r'|ã‚ã‹ã‚Šã¾ã—ãŸ'
        r'|ãŠé¡˜ã„ã—ã¾ã™'
        r'|çµ‚ã‚ã‚Šã§ã™(ã¯ã¯ãƒ¼ã„|ã¯ãƒ¼ã„)?ã‚ã‚ŠãŒ.{0,3}'
        r'|æ•™ãˆã¦ãã ã•ã„'
        r')[\sã€€]*(ç¬‘|ã¯ã„|ã‹)*[\sã€€]*$'
    )

    # Trailing meta that can be stripped from the END of any clause
    _META_TAIL = re.compile(
        r'(ã¯ã„)?(ä»¥ä¸Šã§ã™|ä»¥ä¸Š|çµ‚ã‚ã‚Š(ã§ã™)?|ã‚ã‚ŠãŒã¨ã†)[\sã€€]*(ç¬‘|ã¯ã„)*[\sã€€]*$'
    )

    @classmethod
    def _is_meta_clause(cls, clean_text: str) -> bool:
        """Check if a clause is entirely meta-speech (no narrative content)."""
        return bool(cls._META_CLAUSE.match(clean_text))

    def _strip_meta_speech(self, text: str) -> str:
        """Remove interview meta-speech from start and end of text.
        
        Strategy: The narrative always starts after the participant ID.
        Strip everything from start through the last ID pattern.
        Strip closing meta from the end.
        """
        # Find participant ID â€” strip everything up to and including it
        # Only look in the first 40% of text to avoid stripping narrative content
        search_limit = min(len(text), max(60, len(text) * 2 // 5))
        search_text = text[:search_limit]
        matches = list(self._PARTICIPANT_ID.finditer(search_text))
        if matches:
            last_match = matches[-1]
            # Also skip ã¯ã„ã©ã†ãž / ã©ã†ãž after ID
            remainder = text[last_match.end():]
            for prefix in ['ã¯ã„ã©ã†ãž', 'ã©ã†ãž']:
                if remainder.startswith(prefix):
                    remainder = remainder[len(prefix):]
            text = remainder
        
        # Strip opening meta (greetings, interviewer speech)
        for prefix in ['ã¯ã„ã“ã‚Œã‹ã‚‰ã‚ˆã‚ã—ããŠé¡˜ã„ã—ã¾ã™', 'ã¯ã„ã“ã‚Œã‹ã‚‰ã‚ˆã‚ã—ããŠé¡˜ã„',
                       'ã“ã‚Œã‹ã‚‰ã‚ˆã‚ã—ããŠé¡˜ã„ã—ã¾ã™', 'ã“ã‚Œã‹ã‚‰ãˆã­ãŒã—ã¾ã™',
                       'ã“ã‚Œã‹ã‚‰è©±ã—ã¾ã™', 'ã¯ã„ã§ã”ã–ã„ã¾ã™', 'ã§ã”ã–ã„ã¾ã™',
                       'ã¯ã„ï¼©ï¼¤ã‚’æ•™ãˆã¦ãã ã•ã„', 'ï¼©ï¼¤ã‚’æ•™ãˆã¦ãã ã•ã„']:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        
        # Strip closing meta
        text = self._META_END.sub('', text)
        return text.strip()

    def segment(self, text: str, verbose: bool = True,
                 remove_fillers: bool = False) -> List[Tuple[int, int, str, str, List[int]]]:
        """
        Segment text into clauses.
        
        Args:
            text: Input text
            verbose: Print clause info
            remove_fillers: If True, remove fillers before parsing for cleaner results.
        
        Returns list compatible with English version: (start_idx, end_idx, clause_type, clause_text, token_indices)
        """
        # Strip meta-speech (interviewer questions, ID announcements, closings)
        text = self._strip_meta_speech(text)
        
        # Remove fillers for cleaner parsing
        if remove_fillers:
            try:
                from ja_filler_handler import remove_fillers_simple
                text = remove_fillers_simple(text)
                if self.debug:
                    print(f"[DEBUG] Removed fillers from text")
            except ImportError:
                pass
        
        clauses = self.segment_text(text)
        
        # Convert Clause objects to tuples
        result = []
        for c in clauses:
            start_idx = min(c.token_indices) if c.token_indices else 0
            end_idx = max(c.token_indices) + 1 if c.token_indices else 0
            result.append((start_idx, end_idx, c.clause_type, c.text, list(c.token_indices)))
        
        # Post-process: validate and filter clauses
        result = self._validate_clauses(result)
        
        if verbose:
            print(f"{'TYPE':<20} | {'CLAUSE TEXT'}")
            print("-" * 80)
            for start, end, ctype, ctext, indices in result:
                clean = ctext[:60] + "..." if len(ctext) > 60 else ctext
                print(f"{ctype:<20} | {clean}")
            print(f"\nTotal clauses: {len(result)}")
        
        return result


# ==============================================================================
# TextGrid Handler (adapted for Japanese)
# ==============================================================================

@dataclass
class WordInterval:
    start: float
    end: float
    text: str
    is_pause: bool = False
    mora_count: int = 0  # Japanese uses mora instead of syllables

@dataclass
class PauseInterval:
    start: float
    end: float
    duration: float
    location: str = "unknown"


class TextGridHandler:
    """Handles TextGrid files for Japanese speech."""

    PAUSE_MARKERS = {'', 'sp', 'sil', '<sil>', 'pause', '#', '...',
                     '<p>', '<pause>', 'breath', '<breath>', 'silB', 'silE'}

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.tg = textgrid.openTextgrid(filepath, includeEmptyIntervals=True)
        self.words: List[WordInterval] = []
        self.phones: List[Tuple[float, float, str]] = []
        self.total_mora = 0
        self._extract_tiers()

    def _extract_tiers(self):
        tier_names = self.tg.tierNames

        word_tier_name = None
        for name in ['words', 'word', 'Word', 'Words', 'å˜èªž', 'transcription']:
            if name in tier_names:
                word_tier_name = name
                break
        if not word_tier_name:
            word_tier_name = tier_names[0]

        phone_tier_name = None
        for name in ['phones', 'phone', 'Phone', 'Phones', 'éŸ³ç´ ']:
            if name in tier_names:
                phone_tier_name = name
                break

        if phone_tier_name:
            phone_tier = self.tg.getTier(phone_tier_name)
            for entry in phone_tier.entries:
                self.phones.append((entry.start, entry.end, entry.label.strip()))

        word_tier = self.tg.getTier(word_tier_name)
        for entry in word_tier.entries:
            text = entry.label.strip()
            is_pause = text.lower() in self.PAUSE_MARKERS or text == ''

            mora = 0
            if not is_pause:
                mora = self._count_mora(text)

            self.words.append(WordInterval(
                start=entry.start,
                end=entry.end,
                text=text,
                is_pause=is_pause,
                mora_count=mora
            ))

            if not is_pause:
                self.total_mora += mora

    def _count_mora(self, text: str) -> int:
        """
        Count mora in Japanese text.
        Simple heuristic: count characters, treating small kana as part of previous mora.
        """
        # Small kana that don't count as separate mora
        small_kana = set("ããƒã…ã‡ã‰ã‚ƒã‚…ã‚‡ã‚Žã‚¡ã‚£ã‚¥ã‚§ã‚©ãƒ£ãƒ¥ãƒ§ãƒ®ã£ãƒƒãƒ¼")
        
        count = 0
        for char in text:
            if char not in small_kana and not char.isspace():
                count += 1
        
        return max(count, 1) if text else 0

    def get_transcript(self) -> str:
        return ''.join(w.text for w in self.words if not w.is_pause and w.text)

    def get_word_list(self) -> List[WordInterval]:
        return [w for w in self.words if not w.is_pause and w.text]

    def get_pauses(self, min_duration: float = 0.25) -> List[PauseInterval]:
        pauses = []
        for w in self.words:
            if w.is_pause:
                duration = w.end - w.start
                if duration >= min_duration:
                    pauses.append(PauseInterval(
                        start=w.start,
                        end=w.end,
                        duration=duration
                    ))
        return pauses

    def add_clause_tier(self, clauses: List[Dict], tier_name: str = "clauses"):
        """Add a clause tier with all clauses as non-overlapping intervals."""
        from praatio.data_classes.interval_tier import IntervalTier

        min_time = self.tg.minTimestamp
        max_time = self.tg.maxTimestamp

        if not clauses:
            intervals = [Interval(min_time, max_time, "")]
            new_tier = IntervalTier(tier_name, intervals, min_time, max_time)
            self.tg.addTier(new_tier)
            return []

        # Sort by start_time
        sorted_clauses = sorted(clauses, key=lambda c: (c['start_time'], c['end_time']))

        # Build non-overlapping intervals
        final_clauses = []
        last_end = min_time

        for clause in sorted_clauses:
            start = max(clause['start_time'], last_end)
            end = clause['end_time']

            if end > start + 0.01:
                final_clauses.append({
                    'start': start,
                    'end': end,
                    'label': clause['text'],
                    'type': clause['type'],
                    'word_indices': clause.get('word_indices', None)
                })
                last_end = end

        # Build TextGrid intervals
        intervals = []
        created_intervals = []
        current_time = min_time

        for clause in final_clauses:
            start, end = clause['start'], clause['end']
            label = clause.get('label', '')
            
            if start > current_time + 0.001:
                intervals.append(Interval(current_time, start, ""))
            
            intervals.append(Interval(start, end, label))
            if label.strip():
                created_intervals.append({
                    'label': label,
                    'type': clause['type'],
                    'start': start,
                    'end': end
                })
            current_time = end

        if current_time < max_time - 0.001:
            intervals.append(Interval(current_time, max_time, ""))

        if not intervals:
            intervals = [Interval(min_time, max_time, "")]

        if tier_name in self.tg.tierNames:
            self.tg.removeTier(tier_name)

        new_tier = IntervalTier(tier_name, intervals, min_time, max_time)
        self.tg.addTier(new_tier)
        
        return created_intervals

    def save(self, output_path: str):
        self.tg.save(output_path, format="short_textgrid", includeBlankSpaces=True)


# ==============================================================================
# Clause Aligner (Japanese version)
# ==============================================================================

class ClauseAligner:
    """Align clause boundaries to TextGrid word times for Japanese."""

    def __init__(self, tg_handler: TextGridHandler, segmenter: JapaneseClauseSegmenter):
        self.tg = tg_handler
        self.segmenter = segmenter

    @staticmethod
    def _is_disfluent_idx(word_idx: int, disfluency_labels: List[int] = None) -> bool:
        return disfluency_labels is not None and word_idx < len(disfluency_labels) and disfluency_labels[word_idx] == 1

    def _is_boundary_noise_idx(
        self,
        word_idx: int,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        if self._is_disfluent_idx(word_idx, disfluency_labels):
            return True
        text = tg_words[word_idx].text
        if not self.segmenter._is_unambiguous_filler_surface(text):
            return False
        # Standalone "ã‚“" can be a real morpheme (e.g., negative endings), so
        # only trim it when the disfluency model already marked it.
        if self.segmenter._normalize_surface_text(text) == "ã‚“":
            return False
        return True

    def _is_trailing_discourse_idx(
        self,
        word_idx: int,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        """
        Conservative trailing-only discourse/filler detection.
        This handles boundary leftovers like "...ãªã‚“ã‹ / ...ã‚ã® / ...ãˆã£ã¨" without
        globally deleting ambiguous forms inside clause bodies.
        """
        if self._is_boundary_noise_idx(word_idx, tg_words, disfluency_labels):
            return True

        text = tg_words[word_idx].text
        norm = self.segmenter._normalize_surface_text(text)
        if not norm:
            return False

        trailing_discourse = {
            "\u306a\u3093\u304b",  # なんか
            "\u3042\u306e",        # あの
            "\u305d\u306e",        # その
            "\u3048\u3063\u3068",  # えっと
            "\u3048\u3048\u3068",  # ええと
            "\u3046\u30fc\u3093",  # うーん
            "\u3046\u3093",        # うん
            "\u307e\u3042",        # まあ
            "\u307e",              # ま
            "\u3048\u30fc",        # えー
            "\u3042\u30fc",        # あー
        }
        if norm in trailing_discourse:
            return True

        # Parse fallback for very short function/discourse surfaces.
        doc = self.segmenter.nlp(text)
        toks = [t for t in doc if t.text.strip()]
        if not toks or len(toks) > 2:
            return False
        last = toks[-1]
        if last.dep_ == "discourse":
            return True
        if last.pos_ in {"INTJ", "PART", "SCONJ", "CCONJ"}:
            return True
        return False

    def _is_movable_function_idx(
        self,
        word_idx: int,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        if self._is_boundary_noise_idx(word_idx, tg_words, disfluency_labels):
            return False
        text = tg_words[word_idx].text
        if not text or not text.strip():
            return False
        return self.segmenter._is_function_like_surface(text)
    def _has_content_tokens(
        self,
        indices: List[int],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        for wi in indices:
            if self._is_disfluent_idx(wi, disfluency_labels):
                continue
            text = tg_words[wi].text
            if not text or not text.strip():
                continue
            if self.segmenter._is_unambiguous_filler_surface(text):
                continue
            if self.segmenter._is_content_surface(text):
                return True
        return False

    def _rebuild_clause_payload(self, clause: Dict, tg_words: List[WordInterval], disfluency_labels: List[int] = None) -> Dict:
        """Recompute timing/text/mora after word-index edits."""
        idxs = sorted(clause.get('word_indices', []))
        if not idxs:
            clause['text'] = ''
            clause['mora'] = 0
            return clause

        clause_words = [tg_words[i] for i in idxs]
        if disfluency_labels is not None:
            displayed_words = [tg_words[i] for i in idxs
                               if disfluency_labels[i] == 0
                               and not self.segmenter._is_unambiguous_filler_surface(tg_words[i].text)]
        else:
            displayed_words = [w for w in clause_words
                               if not self.segmenter._is_unambiguous_filler_surface(w.text)]

        clause['word_indices'] = idxs
        clause['start_time'] = clause_words[0].start
        clause['end_time'] = clause_words[-1].end
        clause['text'] = ''.join(w.text for w in displayed_words)
        clause['mora'] = sum(w.mora_count for w in displayed_words)
        return clause

    def _repair_boundary_hangovers(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """
        Repair boundary hangovers where a clause starts with function-only tokens
        (AUX/PART/etc.) that should attach to the previous clause.
        """
        if len(clauses) < 2:
            return clauses

        max_leading_move = 3
        repaired = [clauses[0]]

        for cur in clauses[1:]:
            prev = repaired[-1]

            prev_idxs = sorted(prev.get('word_indices', []))
            cur_idxs = sorted(cur.get('word_indices', []))
            if not prev_idxs or not cur_idxs:
                repaired.append(cur)
                continue

            # Only repair when boundary is contiguous/tiny gap.
            gap = float(cur.get('start_time', 0.0)) - float(prev.get('end_time', 0.0))
            if gap > 0.25:
                repaired.append(cur)
                continue

            leading = []
            for wi in cur_idxs:
                if len(leading) >= max_leading_move:
                    break
                if self._is_movable_function_idx(wi, tg_words, disfluency_labels):
                    leading.append(wi)
                else:
                    break

            if not leading:
                repaired.append(cur)
                continue

            leading_set = set(leading)
            cur_remaining = [wi for wi in cur_idxs if wi not in leading_set]
            if not cur_remaining:
                repaired.append(cur)
                continue
            if not self._has_content_tokens(cur_remaining, tg_words, disfluency_labels):
                repaired.append(cur)
                continue

            prev_augmented = sorted(set(prev_idxs + leading))
            if not self._has_content_tokens(prev_augmented, tg_words, disfluency_labels):
                repaired.append(cur)
                continue

            prev['word_indices'] = prev_augmented
            cur['word_indices'] = cur_remaining
            prev = self._rebuild_clause_payload(prev, tg_words, disfluency_labels)
            cur = self._rebuild_clause_payload(cur, tg_words, disfluency_labels)

            repaired[-1] = prev
            if cur.get('word_indices'):
                repaired.append(cur)

        return repaired

    def _is_relocatable_tail_surface(self, clause_text: str, tail_surface: str) -> bool:
        """
        Decide whether a clause-final token is boundary glue that can move to the next clause.
        This is parse-driven and avoids utterance-specific lexical patching.
        """
        if not clause_text or not tail_surface:
            return False
        tail_norm = self.segmenter._normalize_surface_text(tail_surface)
        if not tail_norm:
            return False

        doc = self.segmenter.nlp(clause_text)
        toks = [t for t in doc if t.text.strip()]
        if not toks:
            return False

        last_tok = None
        for tok in reversed(toks):
            if self.segmenter._normalize_surface_text(tok.text) == tail_norm:
                last_tok = tok
                break
        if last_tok is None:
            return False

        if self.segmenter._is_unambiguous_filler_surface(last_tok.text):
            return True
        if last_tok.dep_ in {"discourse", "mark", "cc"}:
            return True
        if last_tok.pos_ in {"PART", "SCONJ", "CCONJ", "INTJ"}:
            return True
        if last_tok.pos_ == "ADV" and last_tok.dep_ in {"advmod", "discourse"}:
            return True
        return False

    def _repair_trailing_function_hangovers(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """
        Repair the inverse boundary artifact of _repair_boundary_hangovers:
        clause-final function/discourse tokens that should attach to the next clause.
        """
        if len(clauses) < 2:
            return clauses

        max_trailing_move = 2
        repaired = [clauses[0]]

        for cur in clauses[1:]:
            prev = repaired[-1]

            prev_idxs = sorted(prev.get("word_indices", []))
            cur_idxs = sorted(cur.get("word_indices", []))
            if not prev_idxs or not cur_idxs:
                repaired.append(cur)
                continue

            gap = float(cur.get("start_time", 0.0)) - float(prev.get("end_time", 0.0))
            if gap > 0.8:
                repaired.append(cur)
                continue

            trailing: List[int] = []
            for wi in reversed(prev_idxs):
                if len(trailing) >= max_trailing_move:
                    break
                if not self._is_movable_function_idx(wi, tg_words, disfluency_labels):
                    break
                if not self._is_relocatable_tail_surface(prev.get("text", ""), tg_words[wi].text):
                    break
                trailing.append(wi)

            if not trailing:
                repaired.append(cur)
                continue

            trailing_set = set(trailing)
            prev_remaining = [wi for wi in prev_idxs if wi not in trailing_set]
            if not prev_remaining:
                repaired.append(cur)
                continue
            if not self._has_content_tokens(prev_remaining, tg_words, disfluency_labels):
                repaired.append(cur)
                continue

            cur_augmented = sorted(set(cur_idxs + trailing))
            if not self._has_content_tokens(cur_augmented, tg_words, disfluency_labels):
                repaired.append(cur)
                continue

            prev["word_indices"] = prev_remaining
            cur["word_indices"] = cur_augmented
            prev = self._rebuild_clause_payload(prev, tg_words, disfluency_labels)
            cur = self._rebuild_clause_payload(cur, tg_words, disfluency_labels)

            repaired[-1] = prev
            repaired.append(cur)

        return repaired

    def _trim_boundary_noise(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """Trim leading/trailing filler or disfluent words from each clause."""
        trimmed = []
        for clause in clauses:
            idxs = sorted(clause.get('word_indices', []))
            if not idxs:
                continue
            left = 0
            right = len(idxs) - 1
            while left <= right and self._is_boundary_noise_idx(idxs[left], tg_words, disfluency_labels):
                left += 1
            while right >= left and self._is_trailing_discourse_idx(idxs[right], tg_words, disfluency_labels):
                right -= 1
            if left > right:
                continue
            new_idxs = idxs[left:right + 1]
            if not self._has_content_tokens(new_idxs, tg_words, disfluency_labels):
                continue
            clause['word_indices'] = new_idxs
            clause = self._rebuild_clause_payload(clause, tg_words, disfluency_labels)
            if clause.get('text', '').strip():
                trimmed.append(clause)
        return trimmed

    def _build_sentence_word_groups(
        self,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
        pause_split_sec: float = 0.6,
    ) -> List[List[int]]:
        """
        Build sentence-like word groups in two stages:
        1) Split by long pauses in timing.
        2) Split each pause chunk by model sentence boundaries (doc.sents).
        """
        fluent_indices = [
            i for i in range(len(tg_words))
            if disfluency_labels is None or disfluency_labels[i] == 0
        ]
        if not fluent_indices:
            return []

        pause_groups: List[List[int]] = [[fluent_indices[0]]]
        for wi in fluent_indices[1:]:
            prev = pause_groups[-1][-1]
            gap = tg_words[wi].start - tg_words[prev].end
            if gap >= pause_split_sec:
                pause_groups.append([wi])
            else:
                pause_groups[-1].append(wi)

        sentence_groups: List[List[int]] = []
        for group in pause_groups:
            sentence_groups.extend(self._split_group_by_model_sentences(group, tg_words))
        return sentence_groups

    def _split_group_by_model_sentences(
        self,
        group_indices: List[int],
        tg_words: List[WordInterval],
    ) -> List[List[int]]:
        """Split one timing chunk by parser sentence boundaries when available."""
        if len(group_indices) <= 1:
            return [group_indices]

        text = ''.join(tg_words[i].text for i in group_indices)
        if not text:
            return [group_indices]

        doc = self.segmenter.nlp(text)
        sents = list(doc.sents)
        if len(sents) <= 1:
            return [group_indices]

        char_to_local_word: Dict[int, int] = {}
        char_pos = 0
        for local_idx, global_idx in enumerate(group_indices):
            for _ in tg_words[global_idx].text:
                char_to_local_word[char_pos] = local_idx
                char_pos += 1

        split_points: List[int] = []
        for sent in sents[:-1]:
            boundary_char = sent.end_char
            local_cut: Optional[int] = None

            probe = boundary_char
            while probe < char_pos:
                if probe in char_to_local_word:
                    local_cut = char_to_local_word[probe]
                    break
                probe += 1

            if local_cut is None:
                probe = boundary_char - 1
                while probe >= 0:
                    if probe in char_to_local_word:
                        local_cut = char_to_local_word[probe] + 1
                        break
                    probe -= 1

            if local_cut is None:
                continue
            if 0 < local_cut < len(group_indices):
                split_points.append(local_cut)

        if not split_points:
            return [group_indices]

        split_points = sorted(set(split_points))
        out: List[List[int]] = []
        prev = 0
        for cut in split_points:
            if cut > prev:
                out.append(group_indices[prev:cut])
            prev = cut
        if prev < len(group_indices):
            out.append(group_indices[prev:])
        return [g for g in out if g]

    def _align_group_clauses(
        self,
        group_indices: List[int],
        tg_words: List[WordInterval],
        claimed_word_indices: Set[int],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """Run clause segmentation+alignment for one sentence-like word group."""
        group_transcript = ''.join(
            tg_words[i].text for i in group_indices
            if disfluency_labels is None or disfluency_labels[i] == 0
        )
        if not group_transcript:
            return []

        clauses = self.segmenter.segment(group_transcript, verbose=False)
        if not clauses:
            return []

        doc = self.segmenter.nlp(group_transcript)
        char_to_word_idx: Dict[int, int] = {}
        char_pos = 0
        for wi in group_indices:
            if disfluency_labels is not None and disfluency_labels[wi] == 1:
                continue
            for _ in tg_words[wi].text:
                char_to_word_idx[char_pos] = wi
                char_pos += 1

        aligned: List[Dict] = []
        for _start_idx, _end_idx, clause_type, _clause_text, token_indices in clauses:
            if not token_indices:
                continue

            word_indices = set()
            for tok_idx in token_indices:
                if tok_idx >= len(doc):
                    continue
                tok = doc[tok_idx]
                tok_start = tok.idx
                tok_end = tok.idx + len(tok.text)
                for c_pos in range(tok_start, tok_end):
                    wi = char_to_word_idx.get(c_pos)
                    if wi is None:
                        continue
                    if wi in claimed_word_indices:
                        continue
                    word_indices.add(wi)

            if not word_indices:
                continue

            word_indices = sorted(word_indices)
            claimed_word_indices.update(word_indices)

            # Keep disfluent anchor words inside this span for timing continuity.
            if disfluency_labels is not None and len(word_indices) >= 2:
                min_wi = word_indices[0]
                max_wi = word_indices[-1]
                for wi in range(min_wi, max_wi + 1):
                    if wi in claimed_word_indices:
                        continue
                    if disfluency_labels[wi] == 1:
                        word_indices.append(wi)
                        claimed_word_indices.add(wi)
                word_indices = sorted(word_indices)

            clause_words = [tg_words[i] for i in word_indices]
            if disfluency_labels is not None:
                displayed_words = [tg_words[i] for i in word_indices
                                   if disfluency_labels[i] == 0
                                   and not self.segmenter._is_unambiguous_filler_surface(tg_words[i].text)]
            else:
                displayed_words = [w for w in clause_words
                                   if not self.segmenter._is_unambiguous_filler_surface(w.text)]

            final_text = ''.join(w.text for w in displayed_words)
            clause_mora = sum(w.mora_count for w in displayed_words)
            aligned.append({
                'start_time': clause_words[0].start,
                'end_time': clause_words[-1].end,
                'text': final_text,
                'type': clause_type,
                'word_indices': word_indices,
                'mora': clause_mora,
            })
        return aligned

    def _merge_boundary_continuations(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """
        Generic continuation recovery without lexical hard-coding.

        Pattern A:
          short subordinate-like span with predicate but no element
          + immediate next span -> merge forward.

        Pattern B:
          non-predicative host span
          + immediate short predicate fragment -> merge forward.
        """
        if len(clauses) < 2:
            return clauses

        subordinate_like = {"SUB_REL", "SUB_ADVCL", "SUB_CCOMP", "SUB_XCOMP", "SUB_CSUBJ", "CHAIN_TE"}
        merged: List[Dict] = []
        i = 0
        while i < len(clauses):
            cur = clauses[i]
            if i + 1 >= len(clauses):
                merged.append(cur)
                break

            nxt = clauses[i + 1]
            gap = float(nxt.get('start_time', 0.0)) - float(cur.get('end_time', 0.0))

            cur_text = (cur.get('text') or '').strip()
            nxt_text = (nxt.get('text') or '').strip()
            if not cur_text or not nxt_text:
                merged.append(cur)
                i += 1
                continue

            cur_feats = _analyze_clause_surface(self.segmenter, cur_text)
            nxt_feats = _analyze_clause_surface(self.segmenter, nxt_text)

            pattern_a = (
                gap <= 0.35
                and
                cur.get('type') in subordinate_like
                and len(cur_text) <= 12
                and (
                    (cur.get('type') == 'SUB_REL' and not cur_feats.get('has_nominal_token', False))
                    or (
                        cur_feats.get('has_pred_verb_adj', False)
                        and not cur_feats.get('has_element', False)
                    )
                )
            )
            pattern_b = (
                gap <= 1.2
                and
                not cur_feats.get('has_pred_verb_adj', False)
                and cur_feats.get('has_element', False)
                and len(nxt_text) <= 10
                and nxt_feats.get('has_pred_verb_adj', False)
                and not nxt_feats.get('has_element', False)
                and not nxt_feats.get('has_nominal_token', False)
            )
            pattern_c = (
                gap <= 0.8
                and cur.get('type') == 'SUB_REL'
                and len(cur_text) <= 12
                and nxt.get('type') in {'fragment', 'IND', 'SUB_REL'}
                and not nxt_feats.get('has_pred_verb_adj', False)
                and nxt_feats.get('has_pred_or_nominal', False)
            )
            # Pattern D:
            # short non-predicative host + predicative continuation.
            # This catches noisy splits like:
            #   "ãã®æ™‚ã¯..." + "ãªã£ãŸ..."
            #   "ãƒã‚¹ã‚±ãƒƒãƒˆã¨é£Ÿã¹ç‰©ã‚‚..." + "æŒã¡ã¾ã—ãŸ"
            ends_with_connector = bool(
                re.search(
                    r"(?:\u306f|\u304c|\u3092|\u306b|\u3078|\u3067|\u3068|\u306e|\u3082|\u304b\u3089|\u307e\u3067|\u3063\u3066|\u306a\u3093\u304b|\u3042\u306e|\u305d\u306e|\u3048\u3063\u3068|\u3048\u3048\u3068)$",
                    cur_text,
                )
            )
            pattern_d = (
                cur.get('type') in {'IND', 'fragment', 'SUB_REL'}
                and len(cur_text) <= 24
                and not cur_feats.get('has_pred_verb_adj', False)
                and nxt.get('type') in {'IND', 'SUB_REL', 'SUB_ADVCL', 'SUB_CCOMP', 'CHAIN_TE', 'fragment'}
                and nxt_feats.get('has_pred_verb_adj', False)
                and (
                    cur_feats.get('has_element', False)
                    or cur_feats.get('has_case_tail', False)
                    or ends_with_connector
                    or len(cur_text) <= 4
                    or cur_feats.get('all_particles', False)
                )
            )

            if not (pattern_a or pattern_b or pattern_c or pattern_d):
                merged.append(cur)
                i += 1
                continue

            merged_clause = dict(cur)
            merged_clause['word_indices'] = sorted(set(
                list(cur.get('word_indices', [])) + list(nxt.get('word_indices', []))
            ))
            merged_clause = self._rebuild_clause_payload(merged_clause, tg_words, disfluency_labels)

            if pattern_b and merged_clause.get('type') == 'fragment':
                merged_clause['type'] = nxt.get('type', 'fragment')
            if pattern_d and not cur_feats.get('has_pred_verb_adj', False):
                merged_clause['type'] = nxt.get('type', merged_clause.get('type', 'IND'))

            if not merged_clause.get('text', '').strip():
                merged.append(cur)
                i += 1
                continue

            merged.append(merged_clause)
            i += 2

        return merged

    def _find_runon_split_char(self, text: str) -> Optional[int]:
        """
        Detect a robust internal split point for over-merged IND clauses.
        Uses marker-based boundaries and requires predicate evidence on both sides.
        """
        text = (text or "").strip()
        if len(text) < 20:
            return None

        doc = self.segmenter.nlp(text)
        toks = [t for t in doc if t.text.strip()]
        if len(toks) < 6:
            return None

        predicate_count = sum(
            1
            for t in toks
            if t.pos_ in {"VERB", "ADJ"}
            or (t.pos_ == "AUX" and t.lemma_ in {"だ", "です", "ます", "た"})
        )
        if predicate_count < 2:
            return None

        strong_markers = {
            "\u305f\u3089",  # たら
            "\u304b\u3089",  # から
            "\u306e\u3067",  # ので
            "\u306e\u306b",  # のに
            "\u3051\u3069",  # けど
            "\u3051\u308c\u3069",  # けれど
            "\u3051\u308c\u3069\u3082",  # けれども
            "\u3063\u3066",  # って
            "\u3068\u304d",  # とき
            "\u6642",        # 時
            "\u3046\u3061\u306b",  # うちに
            "\u9593\u306b",  # 間に
            "\u306a\u304c\u3089",  # ながら
            "\u3064\u3064",  # つつ
        }
        weak_markers = {"\u3057"}  # し

        def _has_predicate_like(span_text: str) -> bool:
            span_doc = self.segmenter.nlp(span_text)
            return any(
                t.pos_ in {"VERB", "ADJ"}
                or (t.pos_ == "AUX" and t.lemma_ in {"だ", "です", "ます", "た"})
                for t in span_doc
                if t.text.strip()
            )

        candidates: List[Tuple[float, int]] = []
        text_len = len(text)

        for tok in toks[1:-1]:
            surf = self.segmenter._normalize_surface_text(tok.text)
            if not surf:
                continue
            split_specs: List[Tuple[int, float]] = []

            # Marker-driven split after this token.
            if (
                surf in strong_markers
                or surf in weak_markers
                or (tok.dep_ in {"mark", "cc"} and surf not in {"\u3066", "\u3067"})
            ):
                split_specs.append((tok.idx + len(tok.text), 2.0 if surf in strong_markers else 1.0))

            # Topic restart: split at start of noun phrase before a second "は".
            if tok.text == "\u306f" and tok.dep_ == "case" and tok.head.pos_ in {"NOUN", "PRON", "PROPN"}:
                split_specs.append((tok.head.idx, 1.8))

            # Sentence-final polite endings often indicate a clause boundary.
            if tok.pos_ == "AUX" and tok.lemma_ in {"ます", "です"}:
                split_specs.append((tok.idx + len(tok.text), 1.2))
            if tok.pos_ == "AUX" and tok.lemma_ in {"た", "だ"}:
                split_specs.append((tok.idx + len(tok.text), 0.9))

            for split_char, base_score in split_specs:
                if split_char < 6 or split_char > text_len - 6:
                    continue

                left = text[:split_char].strip()
                right = text[split_char:].strip()
                if len(left) < 6 or len(right) < 6:
                    continue

                # v16-fix: protect auxiliary constructions (てしまう, ている, etc.)
                _aux_starts = ("しまい", "しまう", "しまっ", "しまいました",
                               "いる", "いた", "いました", "います",
                               "おく", "おき", "おいた",
                               "くる", "きた", "きました",
                               "いく", "いき", "いった")
                if any(right.startswith(a) for a in _aux_starts):
                    continue
                # v16-fix: protect te-form stems (left ending in っ/し/き/り/び/み/ん + split before て/で)
                if left and left[-1] in "っしきりびみんえ" and right and right[0] in "てで":
                    continue
                # v16-fix: don't split at te-form chain boundary (left ends with て/で)
                # This would break a CHAIN_TE continuation (V1-て V2 patterns).
                if left and left[-1] in "てで":
                    continue
                if not (_has_predicate_like(left) and _has_predicate_like(right)):
                    continue

                score = base_score
                if any(ch in right[:12] for ch in ("\u306f", "\u304c")):
                    score += 1.0
                ratio = split_char / max(1, text_len)
                score += 1.0 - abs(0.5 - ratio)
                candidates.append((score, split_char))

        if not candidates:
            return None
        best_by_char: Dict[int, float] = {}
        for score, split_char in candidates:
            best_by_char[split_char] = max(score, best_by_char.get(split_char, -1e9))
        best = sorted(best_by_char.items(), key=lambda x: x[1], reverse=True)[0][0]
        return best


    def _repair_te_form_boundaries(
        self,
        clauses: List[Dict],
        tg_words,  # List[WordInterval]
        disfluency_labels=None,
    ):
        """
        Move leading て/で from clause start to previous clause end.

        In Japanese, て/で is morphologically part of the preceding verb
        (e.g., 行って = 行っ + て). When the clause segmenter splits at
        the wrong boundary, て/で ends up at the start of the next clause.

        This dedicated repair step catches cases that _repair_boundary_hangovers
        misses (because standalone て may be parsed as noun 手 by GiNZA).
        """
        if len(clauses) < 2:
            return clauses

        # て and で as te-form markers
        TE_MARKERS = {"\u3066", "\u3067"}  # て, で

        repaired = [clauses[0]]
        for cur in clauses[1:]:
            prev = repaired[-1]

            prev_idxs = sorted(prev.get('word_indices', []))
            cur_idxs = sorted(cur.get('word_indices', []))
            if not prev_idxs or not cur_idxs:
                repaired.append(cur)
                continue

            # Find leading て/で in current clause (skip disfluent/empty words)
            te_indices_to_move = []
            found_te = False
            for wi in cur_idxs:
                if disfluency_labels is not None and wi < len(disfluency_labels) and disfluency_labels[wi] == 1:
                    continue  # skip disfluent words
                text = tg_words[wi].text
                if not text or not text.strip():
                    continue
                norm = text.strip()
                if norm in TE_MARKERS and not found_te:
                    te_indices_to_move.append(wi)
                    found_te = True
                else:
                    break  # stop at first non-te content word

            if not te_indices_to_move:
                repaired.append(cur)
                continue

            # Only move if gap is small (contiguous speech)
            gap = float(cur.get('start_time', 0.0)) - float(prev.get('end_time', 0.0))
            if gap > 0.5:
                repaired.append(cur)
                continue

            # Ensure current clause has content remaining after removal
            cur_remaining = [wi for wi in cur_idxs if wi not in set(te_indices_to_move)]
            if not cur_remaining:
                repaired.append(cur)
                continue

            # Move て/で to previous clause
            prev['word_indices'] = sorted(set(prev_idxs + te_indices_to_move))
            cur['word_indices'] = cur_remaining
            prev = self._rebuild_clause_payload(prev, tg_words, disfluency_labels)
            cur = self._rebuild_clause_payload(cur, tg_words, disfluency_labels)

            repaired[-1] = prev
            if cur.get('word_indices'):
                repaired.append(cur)

        return repaired

    def _repair_inflection_splits(
        self,
        clauses: List[Dict],
        tg_words,
        disfluency_labels=None,
    ):
        """
        Repair verb inflection morphology split across clause boundaries.

        MFA sometimes tokenises ました as まし|た or ません as ませ|ん,
        causing the clause segmenter to place a boundary mid-inflection.
        This moves the orphaned inflection suffix word(s) back to the
        previous clause.

        Also absorbs immediately following conjunctive particles (が, けど,
        けれど) so that e.g. ましたが stays as one unit.
        """
        if len(clauses) < 2:
            return clauses

        # Incomplete inflection stems that need a suffix from next clause
        _INFLECTION_STEMS = {"まし", "ませ", "でし"}
        # Valid suffixes that complete the inflection
        _INFLECTION_SUFFIXES = {"た", "ん", "て"}
        # Conjunctive particles that may follow the suffix
        _CONJ_PARTICLES = {"が", "けど", "けれど", "けれども"}
        # Polite suffix words that should attach to previous verb stem
        _POLITE_SUFFIX_WORDS = {"まし", "ます", "ませ"}
        # Completions needed after polite suffix words
        _POLITE_COMPLETIONS = {"まし": {"た", "て"}, "ませ": {"ん"}}

        repaired = [clauses[0]]
        for cur in clauses[1:]:
            prev = repaired[-1]

            prev_idxs = sorted(prev.get('word_indices', []))
            cur_idxs = sorted(cur.get('word_indices', []))
            if not prev_idxs or not cur_idxs:
                repaired.append(cur)
                continue

            # Check gap
            gap = float(cur.get('start_time', 0.0)) - float(prev.get('end_time', 0.0))
            if gap > 0.5:
                repaired.append(cur)
                continue

            # Get last non-disfluent word of previous clause
            prev_last_text = None
            for wi in reversed(prev_idxs):
                if disfluency_labels is not None and wi < len(disfluency_labels) and disfluency_labels[wi] == 1:
                    continue
                text = tg_words[wi].text.strip()
                if text:
                    prev_last_text = text
                    break

            if prev_last_text is None:
                repaired.append(cur)
                continue

            # Get first non-disfluent word(s) of current clause
            first_words = []
            for wi in cur_idxs:
                if disfluency_labels is not None and wi < len(disfluency_labels) and disfluency_labels[wi] == 1:
                    continue
                text = tg_words[wi].text.strip()
                if text:
                    first_words.append((wi, text))
                    if len(first_words) >= 3:
                        break

            if not first_words:
                repaired.append(cur)
                continue

            indices_to_move = []

            # Pattern A: prev ends with incomplete stem (まし/ませ/でし) + next starts with suffix
            if prev_last_text in _INFLECTION_STEMS:
                moved_suffix = False
                for wi, text in first_words:
                    if not moved_suffix and text in _INFLECTION_SUFFIXES:
                        indices_to_move.append(wi)
                        moved_suffix = True
                    elif moved_suffix and text in _CONJ_PARTICLES:
                        indices_to_move.append(wi)
                        break
                    else:
                        break

            # Pattern B: next clause starts with polite suffix word (まし/ます/ませ)
            # e.g., verb stem し | まし た → move まし+た to prev
            if not indices_to_move and first_words[0][1] in _POLITE_SUFFIX_WORDS:
                polite_word = first_words[0][1]
                indices_to_move.append(first_words[0][0])
                # Also grab completions if needed (まし→た, ませ→ん)
                completions = _POLITE_COMPLETIONS.get(polite_word, set())
                conj_done = False
                for wi, text in first_words[1:]:
                    if not conj_done and text in completions:
                        indices_to_move.append(wi)
                    elif text in _CONJ_PARTICLES:
                        indices_to_move.append(wi)
                        conj_done = True
                        break
                    else:
                        break

            if not indices_to_move:
                repaired.append(cur)
                continue

            cur_remaining = [wi for wi in cur_idxs if wi not in set(indices_to_move)]
            if not cur_remaining:
                repaired.append(cur)
                continue

            prev['word_indices'] = sorted(set(prev_idxs + indices_to_move))
            cur['word_indices'] = cur_remaining
            prev = self._rebuild_clause_payload(prev, tg_words, disfluency_labels)
            cur = self._rebuild_clause_payload(cur, tg_words, disfluency_labels)

            repaired[-1] = prev
            if cur.get('word_indices'):
                repaired.append(cur)

        return repaired

    def _repair_auxiliary_splits(
        self,
        clauses: List[Dict],
        tg_words,
        disfluency_labels=None,
    ):
        """
        Repair te-form auxiliary constructions split across clause boundaries.

        When the previous clause ends with て/で (te-form) and the next clause
        starts with an auxiliary continuation (しまい, いく, くる, etc.), move
        the auxiliary word(s) back to the previous clause to keep the compound
        verb unit intact.
        """
        if len(clauses) < 2:
            return clauses

        _TE_ENDINGS = {"て", "で"}
        # Auxiliary continuations that must attach to preceding te-form
        _AUX_STARTS = {
            "しまい", "しまう", "しまっ", "しまいました", "しまいます",
            "いく", "いき", "いった", "いきました", "いきます",
            "くる", "きた", "きました", "きます",
            "いる", "いた", "います", "いました",
            "おく", "おき", "おいた", "おきます", "おきました",
        }

        repaired = [clauses[0]]
        for cur in clauses[1:]:
            prev = repaired[-1]

            prev_idxs = sorted(prev.get('word_indices', []))
            cur_idxs = sorted(cur.get('word_indices', []))
            if not prev_idxs or not cur_idxs:
                repaired.append(cur)
                continue

            gap = float(cur.get('start_time', 0.0)) - float(prev.get('end_time', 0.0))
            if gap > 1.0:  # slightly larger gap for auxiliaries
                repaired.append(cur)
                continue

            # Get previous clause's last non-disfluent word
            prev_last_text = None
            for wi in reversed(prev_idxs):
                if disfluency_labels is not None and wi < len(disfluency_labels) and disfluency_labels[wi] == 1:
                    continue
                text = tg_words[wi].text.strip()
                if text:
                    prev_last_text = text
                    break

            if prev_last_text is None:
                repaired.append(cur)
                continue

            # Get first non-disfluent words of current clause
            first_words = []
            for wi in cur_idxs:
                if disfluency_labels is not None and wi < len(disfluency_labels) and disfluency_labels[wi] == 1:
                    continue
                text = tg_words[wi].text.strip()
                if text:
                    first_words.append((wi, text))
                    if len(first_words) >= 3:
                        break

            if not first_words:
                repaired.append(cur)
                continue

            indices_to_move = []

            # Pattern A: prev ends with て/で + next starts with auxiliary
            if prev_last_text in _TE_ENDINGS:
                if first_words[0][1] in _AUX_STARTS:
                    indices_to_move = [first_words[0][0]]

            # Pattern B: next clause starts with て/で + auxiliary (e.g., れ | て しまい まし た)
            # The て belongs to the verb but was separated; move て+auxiliary+inflection back
            if not indices_to_move and len(first_words) >= 2:
                if first_words[0][1] in _TE_ENDINGS and first_words[1][1] in _AUX_STARTS:
                    indices_to_move = [first_words[0][0], first_words[1][0]]
                    # Also grab inflection continuations (まし+た, ます, ません)
                    _INFLECT_WORDS = {"まし", "ます", "ませ", "た", "ん", "て"}
                    remaining_words = []
                    for wi in cur_idxs:
                        if wi in set(indices_to_move):
                            continue
                        if disfluency_labels is not None and wi < len(disfluency_labels) and disfluency_labels[wi] == 1:
                            continue
                        text = tg_words[wi].text.strip()
                        if text:
                            remaining_words.append((wi, text))
                            if len(remaining_words) >= 3:
                                break
                    for wi, text in remaining_words:
                        if text in _INFLECT_WORDS:
                            indices_to_move.append(wi)
                        else:
                            break

            if not indices_to_move:
                repaired.append(cur)
                continue

            cur_remaining = [wi for wi in cur_idxs if wi not in set(indices_to_move)]
            if not cur_remaining:
                repaired.append(cur)
                continue

            prev['word_indices'] = sorted(set(prev_idxs + indices_to_move))
            cur['word_indices'] = cur_remaining
            prev = self._rebuild_clause_payload(prev, tg_words, disfluency_labels)
            cur = self._rebuild_clause_payload(cur, tg_words, disfluency_labels)

            repaired[-1] = prev
            if cur.get('word_indices'):
                repaired.append(cur)

        return repaired

    def _leading_subordinator_indices(
        self,
        cur_idxs: List[int],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[int]:
        """
        Detect leading subordinator tokens that should attach to the previous clause.
        """
        if not cur_idxs:
            return []

        single_markers = {
            "とき", "時", "ながら", "つつ",
            "うちに", "間に", "ので", "のに",
            "から", "たら", "ば", "なら",
            "ても", "でも", "けど", "けれど", "けれども",
            "まで", "うち", "間", "前", "後", "際",
        }
        pair_markers = {
            "ときに", "時に", "ときは", "時は",
            "うちに", "間に", "前に", "後で",
            "ために", "ように", "際に",
        }
        second_particles = {"に", "は", "で"}

        leading: List[Tuple[int, str]] = []
        for wi in cur_idxs:
            if disfluency_labels is not None and wi < len(disfluency_labels) and disfluency_labels[wi] == 1:
                continue
            text = tg_words[wi].text
            if not text or not text.strip():
                continue
            norm = self.segmenter._normalize_surface_text(text)
            if not norm:
                continue
            leading.append((wi, norm))
            if len(leading) >= 3:
                break

        if not leading:
            return []

        if len(leading) >= 2:
            pair = leading[0][1] + leading[1][1]
            if pair in pair_markers:
                return [leading[0][0], leading[1][0]]

            if (
                leading[0][1] in {"とき", "時", "うち", "間", "前", "後", "ため", "よう", "際"}
                and leading[1][1] in second_particles
            ):
                return [leading[0][0], leading[1][0]]

        if leading[0][1] in single_markers:
            return [leading[0][0]]

        return []

    def _repair_subordinator_hangovers(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """
        Move leading subordinators (e.g., とき/時/うちに/間に/ながら) to the previous clause.
        """
        if len(clauses) < 2:
            return clauses

        repaired = [clauses[0]]
        for cur in clauses[1:]:
            prev = repaired[-1]
            prev_idxs = sorted(prev.get("word_indices", []))
            cur_idxs = sorted(cur.get("word_indices", []))
            if not prev_idxs or not cur_idxs:
                repaired.append(cur)
                continue

            gap = float(cur.get("start_time", 0.0)) - float(prev.get("end_time", 0.0))
            if gap > 1.0:
                repaired.append(cur)
                continue

            move_idxs = self._leading_subordinator_indices(cur_idxs, tg_words, disfluency_labels)
            if not move_idxs:
                repaired.append(cur)
                continue

            prev_text = (prev.get("text") or "").strip()
            prev_feats = _analyze_clause_surface(self.segmenter, prev_text) if prev_text else {}
            if not prev_feats.get("has_pred_verb_adj", False):
                prev_doc = self.segmenter.nlp(prev_text) if prev_text else []
                has_aux_pred = any(
                    t.pos_ == "AUX" and t.lemma_ in {"だ", "です", "ます", "た"}
                    for t in prev_doc
                )
                if not has_aux_pred:
                    repaired.append(cur)
                    continue

            move_set = set(move_idxs)
            cur_remaining = [wi for wi in cur_idxs if wi not in move_set]
            if not cur_remaining:
                repaired.append(cur)
                continue
            if not self._has_content_tokens(cur_remaining, tg_words, disfluency_labels):
                repaired.append(cur)
                continue

            prev_augmented = sorted(set(prev_idxs + move_idxs))
            prev["word_indices"] = prev_augmented
            cur["word_indices"] = cur_remaining
            prev = self._rebuild_clause_payload(prev, tg_words, disfluency_labels)
            cur = self._rebuild_clause_payload(cur, tg_words, disfluency_labels)

            if prev.get("type") in {"IND", "SUB_REL", "fragment", "CHAIN_TE"}:
                prev["type"] = "SUB_ADVCL"

            repaired[-1] = prev
            if cur.get("word_indices"):
                repaired.append(cur)

        return repaired

    def _split_runon_predicate_chains(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """
        Split over-merged predicative clauses that contain multiple predicate chains.
        """
        if not clauses:
            return clauses

        def _split_once(base_clause: Dict) -> Optional[Tuple[Dict, Dict]]:
            ctype = base_clause.get("type", "")
            text = (base_clause.get("text") or "").strip()
            idxs = sorted(base_clause.get("word_indices", []))
            # v16-fix: restrict splitting to IND only; SUB_REL/SUB_ADVCL/CHAIN_TE
            # splits caused regressions (broken auxiliaries, truncated content).
            if ctype != "IND" or len(idxs) < 2:
                return None

            split_char = self._find_runon_split_char(text)
            if split_char is None:
                return None

            char_to_wi: List[int] = []
            for wi in idxs:
                if disfluency_labels is not None and disfluency_labels[wi] == 1:
                    continue
                wtxt = tg_words[wi].text
                if not wtxt:
                    continue
                for _ in wtxt:
                    char_to_wi.append(wi)

            if not char_to_wi:
                return None
            if split_char >= len(char_to_wi):
                split_char = len(char_to_wi) - 1
            if split_char <= 0:
                return None

            right_start_wi = char_to_wi[split_char]
            split_pos = None
            for i, wi in enumerate(idxs):
                if wi == right_start_wi:
                    split_pos = i
                    break
            if split_pos is None or split_pos <= 0 or split_pos >= len(idxs):
                return None

            left_idxs = idxs[:split_pos]
            right_idxs = idxs[split_pos:]
            if not self._has_content_tokens(left_idxs, tg_words, disfluency_labels):
                return None
            if not self._has_content_tokens(right_idxs, tg_words, disfluency_labels):
                return None

            left_clause = dict(base_clause)
            right_clause = dict(base_clause)
            left_clause["word_indices"] = left_idxs
            right_clause["word_indices"] = right_idxs
            left_clause = self._rebuild_clause_payload(left_clause, tg_words, disfluency_labels)
            right_clause = self._rebuild_clause_payload(right_clause, tg_words, disfluency_labels)

            left_text = left_clause.get("text", "")
            if re.search(r"(?:\u305f\u3089|\u304b\u3089|\u306e\u3067|\u306e\u306b|\u3051\u3069|\u3051\u308c\u3069|\u3051\u308c\u3069\u3082|\u3063\u3066)$", left_text):
                left_clause["type"] = "SUB_ADVCL"

            return left_clause, right_clause

        out: List[Dict] = []
        for clause in clauses:
            pending = [clause]
            split_budget = 3
            while pending:
                current = pending.pop(0)
                if split_budget <= 0:
                    out.append(current)
                    continue

                split_pair = _split_once(current)
                if split_pair is None:
                    out.append(current)
                    continue

                left_clause, right_clause = split_pair
                split_budget -= 1
                pending = [left_clause, right_clause] + pending

        return out

    def align_clauses(self, disfluency_labels: List[int] = None) -> List[Dict]:
        """Align clause boundaries to actual word times.
        
        Args:
            disfluency_labels: Per-word labels (0=fluent, 1=disfluent).
                If provided, clause displayed text excludes disfluent words.
        """
        tg_words = self.tg.get_word_list()
        sentence_groups = self._build_sentence_word_groups(tg_words, disfluency_labels)
        if not sentence_groups:
            return []

        aligned_clauses: List[Dict] = []
        claimed_word_indices: Set[int] = set()
        for group in sentence_groups:
            aligned_clauses.extend(
                self._align_group_clauses(
                    group,
                    tg_words,
                    claimed_word_indices,
                    disfluency_labels=disfluency_labels,
                )
            )

        # Repair boundary artifacts before fragment recovery.
        aligned_clauses = self._repair_boundary_hangovers(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._repair_trailing_function_hangovers(aligned_clauses, tg_words, disfluency_labels)

        # Handle unclaimed words as fragments
        all_indices = set(range(len(tg_words)))
        unclaimed = sorted(all_indices - claimed_word_indices)
        
        if unclaimed:
            # Group contiguous unclaimed indices
            groups = []
            if unclaimed:
                curr_group = [unclaimed[0]]
                for i in range(1, len(unclaimed)):
                    if unclaimed[i] == unclaimed[i-1] + 1:
                        curr_group.append(unclaimed[i])
                    else:
                        groups.append(curr_group)
                        curr_group = [unclaimed[i]]
                groups.append(curr_group)
            
            # Create fragment clauses for groups
            for group in groups:
                valid_indices = [idx for idx in group if not tg_words[idx].is_pause and tg_words[idx].text]
                if not valid_indices:
                    continue

                # Keep full timing anchors, but hide disfluent words and fillers from clause text.
                group_words_all = [tg_words[i] for i in valid_indices]
                if disfluency_labels is not None:
                    display_indices = [i for i in valid_indices
                                       if disfluency_labels[i] == 0
                                       and not self.segmenter._is_unambiguous_filler_surface(tg_words[i].text)]
                else:
                    display_indices = [i for i in valid_indices
                                       if not self.segmenter._is_unambiguous_filler_surface(tg_words[i].text)]
                if not display_indices:
                    continue
                group_words_display = [tg_words[i] for i in display_indices]
                group_text = ''.join(w.text for w in group_words_display)

                # Skip if display text is only fillers/backchannels.
                is_all_fillers = all(
                    self.segmenter._is_unambiguous_filler_surface(w.text) for w in group_words_display
                )
                if is_all_fillers:
                    continue

                aligned_clauses.append({
                    'start_time': group_words_all[0].start,
                    'end_time': group_words_all[-1].end,
                    'text': group_text,
                    'type': 'fragment',
                    'word_indices': valid_indices,
                    'mora': sum(w.mora_count for w in group_words_display)
                })

        # Sort before final boundary repair so only true temporal neighbors are touched.
        aligned_clauses.sort(key=lambda c: c['start_time'])
        aligned_clauses = self._repair_boundary_hangovers(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._repair_trailing_function_hangovers(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._merge_boundary_continuations(aligned_clauses, tg_words, disfluency_labels)
        # Run a second pass so newly merged hosts can absorb the next predicate continuation.
        aligned_clauses = self._merge_boundary_continuations(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._split_runon_predicate_chains(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._trim_boundary_noise(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._repair_te_form_boundaries(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._repair_inflection_splits(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._repair_auxiliary_splits(aligned_clauses, tg_words, disfluency_labels)
        # Second inflection pass: auxiliary repair may expose new inflection splits
        aligned_clauses = self._repair_inflection_splits(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._repair_subordinator_hangovers(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses.sort(key=lambda c: c['start_time'])
        return aligned_clauses


# ==============================================================================
# Processing Functions
# ==============================================================================

def _truncate_clause_text(text: str, max_chars: int = 60) -> str:
    """Apply methodology truncation for overly long clause labels."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    # Prefer trimming at a space boundary when available.
    if " " in text:
        cut = text[:max_chars]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        cut = cut.strip()
        if cut:
            return cut
    # Japanese labels are often unspaced; hard trim as fallback.
    return text[:max_chars].strip()


def _strip_trailing_discourse_tail(text: str) -> str:
    """
    Strip clause-final discourse/filler tails conservatively.
    Applies only at the end of the clause label (not internal tokens).
    """
    s = (text or "").strip()
    if not s:
        return s

    trailing_terms = (
        "\u306a\u3093\u304b",  # なんか
        "\u3042\u306e",        # あの
        "\u305d\u306e",        # その
        "\u3048\u3063\u3068",  # えっと
        "\u3048\u3048\u3068",  # ええと
        "\u3046\u30fc\u3093",  # うーん
        "\u3046\u3093",        # うん
        "\u307e\u3042",        # まあ
        "\u307e",              # ま
        "\u3048\u30fc",        # えー
        "\u3042\u30fc",        # あー
    )
    trim_chars = " \u3000\u3001\u3002,.;:!?\uff01\uff1f"

    out = s
    changed = True
    while changed:
        changed = False
        out = out.rstrip(trim_chars)
        for term in trailing_terms:
            if out.endswith(term):
                out = out[:-len(term)].rstrip(trim_chars)
                changed = True
                break
    return out


def _is_discourse_tail_token(segmenter: JapaneseClauseSegmenter, tok) -> bool:
    """Heuristic for clause-final discourse/filler tokens."""
    norm = segmenter._normalize_surface_text(tok.text)
    if not norm:
        return False
    if segmenter._is_unambiguous_filler_surface(tok.text):
        return True
    if norm in {
        "\u306a\u3093\u304b",  # なんか
        "\u3042\u306e",        # あの
        "\u305d\u306e",        # その
        "\u3048\u3063\u3068",  # えっと
        "\u3048\u3048\u3068",  # ええと
        "\u3046\u30fc\u3093",  # うーん
        "\u3046\u3093",        # うん
        "\u307e\u3042",        # まあ
        "\u307e",              # ま
        "\u3048\u30fc",        # えー
        "\u3042\u30fc",        # あー
    }:
        return True
    if tok.dep_ == "discourse":
        return True
    if tok.pos_ in {"INTJ"}:
        return True
    return False


def _strip_trailing_discourse_tail_model(
    segmenter: JapaneseClauseSegmenter,
    text: str,
    max_tokens: int = 2,
) -> str:
    """
    Parse-driven tail cleanup for cases where filler tails are attached to noisy text.
    """
    out = (text or "").strip()
    if not out:
        return out

    for _ in range(max_tokens):
        doc = segmenter.nlp(out)
        toks = [t for t in doc if t.text.strip()]
        if not toks:
            break
        last = toks[-1]
        if not _is_discourse_tail_token(segmenter, last):
            break
        cut = out[:last.idx].strip()
        if len(cut.replace(" ", "").replace("ã€€", "")) < 3:
            break
        out = cut

    return out


def _analyze_clause_surface(segmenter: JapaneseClauseSegmenter, text: str) -> Dict[str, object]:
    """Lightweight predicate/marker checks for methodology post-validation."""
    toks = [t for t in segmenter.nlp(text) if t.text.strip()]
    surfaces = [t.text for t in toks]
    pos = [t.pos_ for t in toks]
    deps = [t.dep_ for t in toks]
    lemmas = [t.lemma_ for t in toks]

    particle_surface = {
        "\u3092", "\u306b", "\u304c", "\u3067", "\u306f", "\u3078", "\u3068", "\u306e", "\u3082", "\u3084",
        "\u304b", "\u306d", "\u3088", "\u305e", "\u3055", "\u306a", "\u308f", "\u3057", "\u3066",
    }
    particle_pos = {"PART", "AUX", "ADP", "PUNCT", "SYM", "SCONJ", "CCONJ"}
    sub_markers = {
        "\u304b\u3089", "\u306a\u304c\u3089", "\u3064\u3064", "\u3051\u3069", "\u3051\u308c\u3069", "\u3051\u308c\u3069\u3082",
        "\u306a\u3089", "\u3070", "\u306e\u3067", "\u306e\u306b", "\u305f\u3081", "\u3088\u3046\u306b",
        "\u307e\u3067", "\u524d\u306b", "\u5f8c\u3067", "\u6642",
    }
    ambiguous_sub_markers = {"\u304c", "\u3068", "\u3057"}
    te_markers = ("\u3066", "\u3067", "\u305f\u308a", "\u3060\u308a")

    has_pred_verb_adj = any(p in {"VERB", "ADJ"} for p in pos)
    has_pred_or_nominal = has_pred_verb_adj or any(p == "NOUN" for p in pos)
    has_nominal_token = any(p in {"NOUN", "PROPN", "PRON"} for p in pos)
    has_element = any(d in {"obj", "iobj", "obl", "advmod", "nmod"} for d in deps)

    has_sub_marker = any(s in sub_markers for s in surfaces)
    if not has_sub_marker:
        for s, p, d in zip(surfaces, pos, deps):
            if s in ambiguous_sub_markers and (p in {"SCONJ", "CCONJ"} or d in {"mark", "cc"}):
                has_sub_marker = True
                break
    if not has_sub_marker and "\u306e" in surfaces:
        for i in range(len(surfaces) - 1):
            if surfaces[i] == "\u306e" and surfaces[i + 1] in {"\u3067", "\u306b"}:
                has_sub_marker = True
                break

    has_te_signal = any(s in te_markers for s in surfaces) or text.endswith(te_markers)
    ends_with_te = text.endswith(te_markers)
    all_particles = bool(toks) and all((s in particle_surface) or (p in particle_pos) for s, p in zip(surfaces, pos))
    has_case_tail = bool(toks) and (
        deps[-1] in {"case", "mark", "cc", "fixed"}
        or pos[-1] in {"ADP", "PART", "SCONJ", "CCONJ"}
        or surfaces[-1] in particle_surface
    )

    light_lemmas = {"\u3059\u308b", "\u307e\u3059", "\u305f", "\u3060", "\u3067\u3059", "\u3042\u308b", "\u3044\u308b", "\u308c\u308b", "\u3089\u308c\u308b"}
    has_lexical_content = any(p in {"NOUN", "PROPN", "ADJ", "ADV", "NUM", "PRON"} for p in pos)
    is_auxiliary_shell = (
        bool(toks)
        and len(text) <= 8
        and not has_lexical_content
        and all(p in {"VERB", "AUX", "PART", "ADP", "SCONJ", "CCONJ", "PUNCT", "SYM"} for p in pos)
        and all((l in light_lemmas) or (s in particle_surface) for l, s in zip(lemmas, surfaces))
    )

    has_stance = any(l in segmenter._minor_matrix_lemmas for l in lemmas) or any(
        x in text for x in ("\u3068\u601d\u3046", "\u3068\u8003\u3048\u308b", "\u3068\u611f\u3058\u308b")
    )

    normalized = unicodedata.normalize("NFKC", text)
    compact = normalized.replace(" ", "").replace("\u3000", "")
    has_meta = bool(re.match(r"^[A-Za-z]{2,}\d+", normalized)) or bool(
        re.search(
            r"(?:\u3042\u308a\u304c\u3068\u3046(?:\u3054\u3056\u3044\u307e\u3059|\u3054\u3056\u3044\u307e\u3057\u305f)?|\u6709\u96e3\u3046(?:\u3054\u3056\u3044\u307e\u3059|\u3054\u3056\u3044\u307e\u3057\u305f)?|\u4ee5\u4e0a(?:\u3067\u3059)?|\u306f\u3044\u3069\u3046\u3082|\u3069\u3046\u3082|\u304a\u9858\u3044\u3057\u307e\u3059)",
            compact,
        )
    )

    return {
        "has_pred_verb_adj": has_pred_verb_adj,
        "has_pred_or_nominal": has_pred_or_nominal,
        "has_nominal_token": has_nominal_token,
        "has_element": has_element,
        "has_sub_marker": has_sub_marker,
        "has_te_signal": has_te_signal,
        "ends_with_te": ends_with_te,
        "all_particles": all_particles,
        "has_case_tail": has_case_tail,
        "is_auxiliary_shell": is_auxiliary_shell,
        "has_stance": has_stance,
        "has_meta": has_meta,
    }


def _has_subordinator_tail(compact_text: str) -> bool:
    """Check whether compact text ends with a subordinator-like tail."""
    tails = (
        "とき", "時", "ときに", "時に", "ときは", "時は",
        "うちに", "間に", "ながら", "つつ",
        "ので", "のに", "から",
        "けど", "けれど", "けれども",
        "たら", "ば", "なら",
        "ても", "でも",
        "ために", "ように",
        "まで", "前に", "後で", "際に",
    )
    return any(compact_text.endswith(t) for t in tails)


def _is_ind_like_sub_rel(segmenter: JapaneseClauseSegmenter, text: str) -> bool:
    """
    Parse-driven check: SUB_REL span looks like an independent sentence.
    """
    compact = text.replace(" ", "").replace("\u3000", "")
    if len(compact) < 6:
        return False
    if _has_subordinator_tail(compact):
        return False

    doc = segmenter.nlp(text)
    toks = [t for t in doc if t.text.strip()]
    if not toks:
        return False

    has_topic = any(
        t.text == "は" and t.dep_ == "case" and t.head.pos_ in {"NOUN", "PRON", "PROPN"}
        for t in toks
    )
    if not has_topic:
        return False

    has_predicate = any(
        t.pos_ in {"VERB", "ADJ"}
        or (t.pos_ == "AUX" and t.lemma_ in {"だ", "です", "ます", "た"})
        for t in toks
    )
    if not has_predicate:
        return False

    last = toks[-1]
    if last.pos_ in {"NOUN", "PROPN", "PRON", "NUM"}:
        return False
    if last.dep_ in {"case", "mark", "cc", "fixed"}:
        return False
    # v16-fix: adnominal / prenominal endings modify a following noun → keep SUB_REL.
    # E.g., そういう, ある, いる, した, etc. in their 連体形 usage.
    if last.dep_ in {"acl", "amod", "relcl"}:
        return False
    # Demonstrative adjectives ending in う (そういう, こういう, ああいう) are prenominal.
    if last.text.endswith("いう") or last.text in {"そういう", "こういう", "ああいう"}:
        return False

    return True


def _apply_methodology_rules(
    clauses: List[Dict],
    segmenter: JapaneseClauseSegmenter,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Apply Japanese validation rules from CLAUSE_SEGMENTATION_METHODOLOGY.md (4.2).
    """
    kept: List[Dict] = []
    stats = {
        "dropped_fragment_lt3": 0,
        "dropped_short_no_pred": 0,
        "dropped_particles_only": 0,
        "dropped_meta": 0,
        "dropped_no_pred_lt10": 0,
        "dropped_no_pred_type": 0,
        "dropped_bare_te": 0,
        "type_relabel_chain_te_to_advcl": 0,
        "type_relabel_chain_te_to_ind": 0,
        "type_relabel_advcl_to_chain_te": 0,
        "type_relabel_advcl_to_ind": 0,
        "type_relabel_minor_to_ind": 0,
        "type_relabel_ind_no_pred_to_fragment": 0,
        "type_relabel_no_pred_to_fragment": 0,
        "v15_ind_to_sub_advcl_subordinator_final": 0,
        "v15_ind_to_minor_stance_recovery": 0,
        "v15_fragment_to_ind_predicate_recovery": 0,
        "v15_sub_rel_to_sub_advcl_toki": 0,
        "v16_fragment_predicate_relabel": 0,
        "v16_sub_rel_to_ind_topic_predicate": 0,
        "v15_orphan_merge": 0,
        "stripped_discourse_tail": 0,
        "truncated_long": 0,
    }

    for clause in clauses:
        text = (clause.get("text") or "").strip()
        ctype = clause.get("type", "")
        if not text:
            continue

        # Remove trailing discourse fillers that frequently leak to clause edges.
        text_stripped = _strip_trailing_discourse_tail_model(segmenter, text)
        if text_stripped == text:
            text_stripped = _strip_trailing_discourse_tail(text)
        if text_stripped and text_stripped != text:
            stats["stripped_discourse_tail"] += 1
            text = text_stripped

        # Very long rule: truncate.
        if len(text) > 60:
            text_new = _truncate_clause_text(text, 60)
            if text_new != text:
                stats["truncated_long"] += 1
                text = text_new

        feats = _analyze_clause_surface(segmenter, text)

        # Type normalization for better rule alignment.
        if ctype == "MINOR" and not feats["has_stance"]:
            ctype = "IND"
            stats["type_relabel_minor_to_ind"] += 1
            feats = _analyze_clause_surface(segmenter, text)

        if ctype == "CHAIN_TE":
            compact_text = text.replace(" ", "").replace("　", "")
            if _has_subordinator_tail(compact_text):
                ctype = "SUB_ADVCL"
                stats["type_relabel_chain_te_to_advcl"] += 1
                feats = _analyze_clause_surface(segmenter, text)

            if ctype == "CHAIN_TE" and not feats["has_te_signal"]:
                if feats["has_sub_marker"]:
                    ctype = "SUB_ADVCL"
                    stats["type_relabel_chain_te_to_advcl"] += 1
                elif feats["has_pred_verb_adj"]:
                    ctype = "IND"
                    stats["type_relabel_chain_te_to_ind"] += 1
                else:
                    ctype = "fragment"
                    stats["type_relabel_no_pred_to_fragment"] += 1
                feats = _analyze_clause_surface(segmenter, text)
            elif ctype == "CHAIN_TE" and not feats["has_element"]:
                # CHAIN_TE requires verb+element; relabel when element evidence is absent.
                if feats["has_sub_marker"]:
                    ctype = "SUB_ADVCL"
                    stats["type_relabel_chain_te_to_advcl"] += 1
                elif feats["has_pred_verb_adj"] or feats["has_pred_or_nominal"]:
                    ctype = "IND"
                    stats["type_relabel_chain_te_to_ind"] += 1
                else:
                    ctype = "fragment"
                    stats["type_relabel_no_pred_to_fragment"] += 1
                feats = _analyze_clause_surface(segmenter, text)

        if ctype == "SUB_ADVCL" and not feats["has_sub_marker"]:
            compact_text = text.replace(" ", "").replace("　", "")
            if _has_subordinator_tail(compact_text):
                pass
            elif feats["has_te_signal"] and feats["has_element"]:
                ctype = "CHAIN_TE"
                stats["type_relabel_advcl_to_chain_te"] += 1
            elif feats["has_pred_verb_adj"]:
                ctype = "IND"
                stats["type_relabel_advcl_to_ind"] += 1
            else:
                ctype = "fragment"
                stats["type_relabel_no_pred_to_fragment"] += 1
            feats = _analyze_clause_surface(segmenter, text)

        # Residual IND spans ending in case/connector tails are typically
        # non-predicative boundary leftovers under noisy learner speech.
        if ctype == "IND" and not feats["has_pred_verb_adj"] and feats["has_case_tail"]:
            ctype = "fragment"
            stats["type_relabel_ind_no_pred_to_fragment"] += 1
            feats = _analyze_clause_surface(segmenter, text)

        # MINOR exemption: keep regardless.
        if ctype == "MINOR":
            clause["type"] = ctype
            clause["text"] = text
            kept.append(clause)
            continue

        # Non-fragment clause types should carry a predicate/nominal core.
        if ctype != "fragment" and not feats["has_pred_or_nominal"]:
            stats["dropped_no_pred_type"] += 1
            continue


        # -- v15 Fix 4: SUB_REL with subordinator -> SUB_ADVCL ------
        # FROZEN_RULES Rule 3a lists subordinators. GiNZA sometimes parses
        # adverbial clauses as acl -> SUB_REL. Relabel when subordinator found.
        if ctype == "SUB_REL":
            compact_text = text.replace(" ", "").replace("　", "")
            _sub_rel_fix_endings = (
                "とき", "時",
                "ときに", "時に",
                "ときは", "時は",
                "ので", "のに",
                "から",
                "けど", "けれど", "けれども",
                "たら", "ば", "なら",
                "ながら", "つつ",
                "ても", "でも",
                "ために", "ように",
                "まで",
                "前に", "後で",
                "うちに", "間に",
                "際に",
            )
            is_sub_rel_to_advcl = any(compact_text.endswith(e) for e in _sub_rel_fix_endings)
            # Also check ambiguous が/と at end with SCONJ POS
            if not is_sub_rel_to_advcl:
                _doc = segmenter.nlp(text)
                _toks = [t for t in _doc if t.text.strip()]
                if _toks:
                    _last = _toks[-1]
                    if (_last.text in ("が", "と", "し") and
                        (_last.pos_ in {"SCONJ", "CCONJ"} or _last.dep_ in {"mark", "cc"})):
                        is_sub_rel_to_advcl = True
            if is_sub_rel_to_advcl:
                ctype = "SUB_ADVCL"
                stats["v15_sub_rel_to_sub_advcl_toki"] += 1
                feats = _analyze_clause_surface(segmenter, text)

        # v16: SUB_REL that behaves like a full sentence -> IND.
        if ctype == "SUB_REL" and _is_ind_like_sub_rel(segmenter, text):
            ctype = "IND"
            stats["v16_sub_rel_to_ind_topic_predicate"] += 1
            feats = _analyze_clause_surface(segmenter, text)

        # -- v15 Fix 1: IND ending with subordinator -> SUB_ADVCL ----
        if ctype == "IND":
            compact_text = text.replace(" ", "").replace("　", "")
            # Strip trailing の/は for check (e.g., からの → から)
            _check_text = compact_text
            for _tp in ("の", "は"):
                if _check_text.endswith(_tp) and len(_check_text) > 3:
                    _check_text = _check_text[:-len(_tp)]
            _unambiguous_subordinators = (
                "から", "ので", "けど", "けれど", "けれども",
                "たら", "ば", "なら",
                "ながら", "つつ",
                "のに", "し",
                "ても", "でも",
                "ために", "ように",
                "まで",
                "前に", "後で",
                "うちに", "間に",
                "際に",
                "とき", "時", "ときに", "時に",
            )
            is_ind_to_advcl = any(_check_text.endswith(sub) for sub in _unambiguous_subordinators)
            if not is_ind_to_advcl:
                _doc = segmenter.nlp(text)
                _toks = [t for t in _doc if t.text.strip()]
                if _toks:
                    _last = _toks[-1]
                    if (_last.text in ("が", "と") and
                        (_last.pos_ in {"SCONJ", "CCONJ"} or _last.dep_ in {"mark", "cc"})):
                        is_ind_to_advcl = True
            if is_ind_to_advcl:
                ctype = "SUB_ADVCL"
                stats["v15_ind_to_sub_advcl_subordinator_final"] += 1
                feats = _analyze_clause_surface(segmenter, text)

        # -- v15 Fix 3: Fragment with valid predicate -> IND recovery --
        if ctype == "fragment" and len(text) >= 10:
            _masu_pats = (
                "ました", "ます", "ません",
                "でした", "です",
                "ている", "ていた",
                "ていました", "ています",
                "しました", "しています",
                "なりました", "ありました",
                "いました", "きました",
                "れました",
                "みました", "します",
                "見ろ", "帰れ", "起きろ",
            )
            compact_text = text.replace(" ", "").replace("　", "")
            has_pred_pattern = any(p in compact_text for p in _masu_pats)
            if has_pred_pattern:
                _doc = segmenter.nlp(text)
                _has_v = any(t.pos_ in {"VERB", "AUX"} for t in _doc)
                if _has_v:
                    _frag_sub_ends = (
                        "とき", "時", "ときに", "時に",
                        "から", "ので", "けど", "けれど",
                        "たら", "ば", "のに", "ながら",
                    )
                    if any(compact_text.endswith(e) for e in _frag_sub_ends):
                        ctype = "SUB_ADVCL"
                    else:
                        ctype = "IND"
                    stats["v15_fragment_to_ind_predicate_recovery"] += 1
                    feats = _analyze_clause_surface(segmenter, text)

        # v16: generic fragment recovery driven by parse features (not string templates).
        # v16-fix: require minimum length to avoid promoting single adverbs like 遅く.
        if ctype == "fragment" and feats["has_pred_verb_adj"] and len(text.replace(" ", "").replace("\u3000", "")) >= 6:
            compact_text = text.replace(" ", "").replace("　", "")
            if feats["has_sub_marker"] or _has_subordinator_tail(compact_text):
                ctype = "SUB_ADVCL"
            elif feats["has_te_signal"] and feats["has_element"]:
                ctype = "CHAIN_TE"
            else:
                ctype = "IND"
            stats["v16_fragment_predicate_relabel"] += 1
            feats = _analyze_clause_surface(segmenter, text)

        # Validation removal rules.
        if ctype == "fragment" and len(text) < 3:
            stats["dropped_fragment_lt3"] += 1
            continue
        if feats.get("is_auxiliary_shell", False):
            stats["dropped_particles_only"] += 1
            continue
        if feats["all_particles"]:
            stats["dropped_particles_only"] += 1
            continue
        if len(text) <= 3 and not feats["has_pred_verb_adj"]:
            stats["dropped_short_no_pred"] += 1
            continue
        if len(text) < 10 and not feats["has_pred_verb_adj"]:
            stats["dropped_no_pred_lt10"] += 1
            continue
        if ctype == "CHAIN_TE" and len(text) < 12 and feats["ends_with_te"] and not feats["has_element"]:
            stats["dropped_bare_te"] += 1
            continue
        if feats["has_meta"]:
            stats["dropped_meta"] += 1
            continue

        clause["type"] = ctype
        clause["text"] = text
        kept.append(clause)


    # -- v15 Patch G: SUB_REL with mid-clause temporal marker -----
    # Some SUB_REL clauses contain ながら/うちに/途中に mid-clause
    # due to under-segmentation by parser. Relabel to SUB_ADVCL.
    _mid_clause_temporals = ("ながら", "うちに", "途中に", "際に", "間に")
    for idx, cl in enumerate(kept):
        ctype = cl.get("type", "")
        text = (cl.get("text") or "").replace(" ", "").replace("　", "")
        if ctype == "SUB_REL":
            if any(marker in text for marker in _mid_clause_temporals):
                cl["type"] = "SUB_ADVCL"
                stats["v15_sub_rel_to_sub_advcl_toki"] += 1

    # -- v15 Patch C: Merge orphaned-て and short renyoukei fragments ---
    _renyou_endings = ("っ", "し", "き", "り", "み", "び",
                       "に", "ぎ", "ち", "ひ", "い")
    _orphan_starts = ("た", "て", "で", "だ")
    i_merge = 0
    while i_merge < len(kept) - 1:
        cur_c = kept[i_merge]
        nxt_c = kept[i_merge + 1]
        cur_text = (cur_c.get("text") or "").replace(" ", "").replace("　", "")
        nxt_text = (nxt_c.get("text") or "").replace(" ", "").replace("　", "")
        do_merge = False
        # Case 1: Very short clause (<=3 chars) ending with renyoukei
        if (len(cur_text) <= 3 and cur_text and
            cur_text[-1] in _renyou_endings and
            nxt_text and nxt_text[0] in _orphan_starts):
            do_merge = True
        # Case 2: Truncated ました/でした/ません
        if (cur_text.endswith("まし") or cur_text.endswith("でし")) and nxt_text.startswith("た"):
            do_merge = True
        if cur_text.endswith("ませ") and nxt_text.startswith("ん"):
            do_merge = True
        # Case 3: Orphaned て/で/た (1-2 chars)
        if len(nxt_text) <= 2 and nxt_text in ("て", "で", "た", "たり"):
            do_merge = True
        if do_merge:
            cur_c["text"] = (cur_c.get("text", "") + " " + nxt_c.get("text", "")).strip()
            cur_indices = cur_c.get("word_indices", [])
            nxt_indices = nxt_c.get("word_indices", [])
            cur_c["word_indices"] = sorted(set(cur_indices + nxt_indices))
            if "end_time" in nxt_c:
                cur_c["end_time"] = nxt_c["end_time"]
            if "mora" in cur_c and "mora" in nxt_c:
                cur_c["mora"] = cur_c["mora"] + nxt_c["mora"]
            kept.pop(i_merge + 1)
            continue
        i_merge += 1

    # -- v15 Fix 2: IND with stance verb -> MINOR (context-aware) --
    _stance_surfaces = (
        "思い", "思っ", "思う",
        "思います", "思いました",
        "考え", "考えて", "考えます", "考えました",
        "感じ", "感じて", "感じます", "感じました",
    )
    for ki in range(1, len(kept)):
        prev_c = kept[ki - 1]
        cur_c = kept[ki]
        if cur_c.get("type") != "IND":
            continue
        if prev_c.get("type") != "SUB_CCOMP":
            continue
        cur_text = (cur_c.get("text") or "").replace(" ", "").replace("　", "")
        has_stance_surface = any(cur_text.startswith(s) for s in _stance_surfaces)
        if not has_stance_surface:
            _doc = segmenter.nlp(cur_c.get("text", ""))
            has_stance_surface = any(
                t.lemma_ in {"思う", "考える", "感じる"} and t.pos_ == "VERB"
                for t in _doc
            )
        if has_stance_surface:
            cur_c["type"] = "MINOR"
            stats["v15_ind_to_minor_stance_recovery"] += 1

    # -- v15 Patch F: Embedded stance verb in IND → MINOR --------
    _embedded_stance_tails = (
        "と思う", "と思い", "と思っ", "と思った",
        "と思います", "と思いまし",
        "と考え", "と考えて", "と考えた",
        "と感じ", "と感じて", "と感じた",
        "かと思い", "だと思います",
    )
    for idx, cl in enumerate(kept):
        ctype = cl.get("type", "")
        text = (cl.get("text") or "").replace(" ", "").replace("　", "")
        if ctype == "IND" and len(text) > 4:
            tail = text[-12:] if len(text) > 12 else text
            if any(tail.endswith(p) for p in _embedded_stance_tails):
                cl["type"] = "MINOR"
                stats["v15_ind_to_minor_stance_recovery"] += 1

    return kept, stats


def process_single_textgrid(input_path: str, output_path: str,
                           segmenter: JapaneseClauseSegmenter) -> List[str]:
    """Process a single TextGrid file with optional disfluency detection.
    
    If segmenter.disfluency_detector is set, disfluency labels are computed
    and used to build clean text for parsing. A disfluency tier is added.
    """
    filename = os.path.basename(input_path)
    print(f"\nProcessing: {filename}")

    tg = TextGridHandler(input_path)
    tg_words = tg.get_word_list()
    print(f"  Words: {len(tg_words)}, Mora: {tg.total_mora}")

    # --- Disfluency detection ---
    disfluency_labels = None
    dd = getattr(segmenter, 'disfluency_detector', None)
    if dd is not None:
        word_texts = [w.text for w in tg_words]
        disfluency_labels = dd.detect(word_texts)
        n_dis = sum(disfluency_labels)
        print(f"  Disfluency tier: {n_dis} disfluent words marked")

        # Add disfluency tier to TextGrid
        _add_disfluency_tier(tg, tg_words, disfluency_labels)

    # --- Clause segmentation + alignment ---
    # For logical clause count, segment full transcript
    transcript = tg.get_transcript()
    logical_clauses = segmenter.segment(transcript, verbose=False)

    aligner = ClauseAligner(tg, segmenter)
    clauses = aligner.align_clauses(disfluency_labels=disfluency_labels)

    # Filter filler-only clauses
    filtered_clauses = []
    for c in clauses:
        if 'word_indices' in c and c['word_indices']:
            words = [tg_words[i].text for i in c['word_indices']]
            # Consider disfluent words as effectively "removed"
            if disfluency_labels is not None:
                content_words = [
                    w for i_w, w in zip(c['word_indices'], words)
                    if w and not segmenter._is_unambiguous_filler_surface(w)
                    and disfluency_labels[i_w] == 0
                    and segmenter._is_content_surface(w)
                ]
            else:
                content_words = [
                    w for w in words
                    if w and not segmenter._is_unambiguous_filler_surface(w) and segmenter._is_content_surface(w)
                ]
            if len(content_words) > 0:
                filtered_clauses.append(c)
        else:
            filtered_clauses.append(c)

    methodology_stats = {}
    if getattr(segmenter, "apply_methodology_rules", True):
        filtered_clauses, methodology_stats = _apply_methodology_rules(
            filtered_clauses,
            segmenter,
        )

    non_fragment = [c for c in filtered_clauses if c['type'] != 'fragment']
    fragments = [c for c in filtered_clauses if c['type'] == 'fragment']

    print(f"  Clauses (Logical): {len(logical_clauses)}")
    print(f"  Clauses (Aligned): {len(non_fragment)} (fragments: {len(fragments)})")
    if methodology_stats:
        dropped = (
            methodology_stats.get("dropped_fragment_lt3", 0)
            + methodology_stats.get("dropped_short_no_pred", 0)
            + methodology_stats.get("dropped_particles_only", 0)
            + methodology_stats.get("dropped_meta", 0)
            + methodology_stats.get("dropped_no_pred_lt10", 0)
            + methodology_stats.get("dropped_no_pred_type", 0)
            + methodology_stats.get("dropped_bare_te", 0)
        )
        relabeled = (
            methodology_stats.get("type_relabel_chain_te_to_advcl", 0)
            + methodology_stats.get("type_relabel_chain_te_to_ind", 0)
            + methodology_stats.get("type_relabel_advcl_to_chain_te", 0)
            + methodology_stats.get("type_relabel_advcl_to_ind", 0)
            + methodology_stats.get("type_relabel_minor_to_ind", 0)
            + methodology_stats.get("type_relabel_ind_no_pred_to_fragment", 0)
            + methodology_stats.get("type_relabel_no_pred_to_fragment", 0)
        )
        truncated = methodology_stats.get("truncated_long", 0)
        print(f"  Methodology rules: dropped={dropped}, relabeled={relabeled}, truncated={truncated}")

    # Add clause tier and save
    final_intervals = tg.add_clause_tier(filtered_clauses, tier_name="clauses")
    tg.save(output_path)
    print(f"  Saved: {output_path}")

    # Build log
    clause_log = []
    clause_log.append(f"File: {filename}")
    clause_log.append(f"Words: {len(tg_words)}, Mora: {tg.total_mora}")
    if disfluency_labels is not None:
        clause_log.append(f"Disfluent words: {sum(disfluency_labels)}")
    clause_log.append(f"Clauses (Logical): {len(logical_clauses)}")
    clause_log.append(f"Clauses (Aligned): {len(non_fragment)}")
    clause_log.append(f"Fragments: {len(fragments)}")
    if methodology_stats:
        clause_log.append(f"Methodology Rules: {methodology_stats}")
    clause_log.append("-" * 80)

    # Show clauses
    for c in (final_intervals or []):
        display_text = c['label'][:60] + '...' if len(c['label']) > 60 else c['label']
        line = f"    {c['type']:<20} | {display_text}"
        try:
            print(line)
        except UnicodeEncodeError:
            # Windows cp1252 terminals can fail on Japanese output.
            print(line.encode("unicode_escape").decode("ascii"))
        clause_log.append(f"{c['type']:<20} | {c['label']}")
    
    clause_log.append("")
    return clause_log


def _add_disfluency_tier(tg_handler: TextGridHandler, tg_words: List[WordInterval],
                         disfluency_labels: List[int]):
    """Add a disfluency tier marking disfluent words in the TextGrid."""
    from praatio.data_classes.interval_tier import IntervalTier

    min_time = tg_handler.tg.minTimestamp
    max_time = tg_handler.tg.maxTimestamp

    intervals = []
    current_time = min_time

    for i, word in enumerate(tg_words):
        # Gap before this word
        if word.start > current_time + 0.001:
            intervals.append(Interval(current_time, word.start, ""))
        
        label = word.text if disfluency_labels[i] == 1 else ""
        intervals.append(Interval(word.start, word.end, label))
        current_time = word.end

    if current_time < max_time - 0.001:
        intervals.append(Interval(current_time, max_time, ""))

    if not intervals:
        intervals = [Interval(min_time, max_time, "")]

    tier_name = "disfluency"
    if tier_name in tg_handler.tg.tierNames:
        tg_handler.tg.removeTier(tier_name)

    new_tier = IntervalTier(tier_name, intervals, min_time, max_time)
    tg_handler.tg.addTier(new_tier)


def process_all_textgrids(input_dir: str, output_dir: str,
                          segmenter: JapaneseClauseSegmenter):
    """Process all TextGrid files in directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.TextGrid')])
    print(f"Processing {len(files)} TextGrid files...")
    
    all_logs = []
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            log = process_single_textgrid(input_path, output_path, segmenter)
            all_logs.extend(log)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Write log
    log_path = os.path.join(output_dir, 'clause_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_logs))
    print(f"\nClause log saved to: {log_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Japanese TextGrid Clause Segmentation V2 (GiNZA + Vercellotti rules)",
        epilog="Uses GiNZA for Japanese NLP with Vercellotti-style clause detection."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input TextGrid file or directory")
    parser.add_argument("-o", "--output", required=True,
                        help="Output TextGrid file or directory")
    parser.add_argument("--model", default="ja_ginza_electra",
                        help="GiNZA model name (default: ja_ginza_electra)")
    parser.set_defaults(use_disfluency=True)
    parser.add_argument("--use-disfluency", dest="use_disfluency", action="store_true",
                        help="Enable neural disfluency detector (default: enabled).")
    parser.add_argument("--no-disfluency", dest="use_disfluency", action="store_false",
                        help="Disable neural disfluency detector.")
    parser.add_argument("--disfluency-model", default=None,
                        help="Path to disfluency model dir. Default: auto-detect project model.")
    parser.set_defaults(use_methodology_rules=True)
    parser.add_argument("--use-methodology-rules", dest="use_methodology_rules", action="store_true",
                        help="Apply CLAUSE_SEGMENTATION_METHODOLOGY.md post-validation rules (default: enabled).")
    parser.add_argument("--no-methodology-rules", dest="use_methodology_rules", action="store_false",
                        help="Disable methodology post-validation rules.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    
    args = parser.parse_args()
    
    # Load GiNZA model
    print(f"Loading GiNZA model: {args.model}")
    try:
        nlp = spacy.load(args.model)
    except OSError:
        print(f"Model '{args.model}' not found. Please install GiNZA:")
        print("  pip install ginza ja_ginza_electra")
        sys.exit(1)
    print("Model loaded!")
    
    # Initialize segmenter
    disfluency_detector = None
    if args.use_disfluency:
        try:
            disfluency_detector = DisfluencyDetector(model_path=args.disfluency_model)
        except Exception as e:
            print(f"Failed to load disfluency detector: {e}")
            sys.exit(1)

    segmenter = JapaneseClauseSegmenter(
        nlp,
        debug=args.debug,
        disfluency_detector=disfluency_detector,
        apply_methodology_rules=args.use_methodology_rules,
    )
    
    # Process
    if os.path.isfile(args.input):
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        process_single_textgrid(args.input, args.output, segmenter)
    else:
        process_all_textgrids(args.input, args.output, segmenter)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

