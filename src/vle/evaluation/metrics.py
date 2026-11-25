"""Evaluation metrics (identical procedure for every method)."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def set_counts(pred: set, ref: set) -> tuple[int, int, int]:
    return len(pred & ref), len(pred - ref), len(ref - pred)


def evaluate_sets(preds: dict[str, set], refs: dict[str, set]) -> dict:
    """Micro P/R/F1, macro-over-labels F1, and per-image F1 vector (for paired tests).

    Images with an empty reference *and* empty prediction get per-image F1 = 1.
    """
    TP = FP = FN = 0
    per_label: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    per_image_f1, per_image_p, per_image_r, ids = [], [], [], []
    for iid in refs:
        pred, ref = preds.get(iid, set()), refs[iid]
        tp, fp, fn = set_counts(pred, ref)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        for l in pred & ref:
            per_label[l][0] += 1
        for l in pred - ref:
            per_label[l][1] += 1
        for l in ref - pred:
            per_label[l][2] += 1
        if not pred and not ref:
            per_image_f1.append(1.0)
            per_image_p.append(1.0)
            per_image_r.append(1.0)
        else:
            p, r, f = prf(tp, fp, fn)
            per_image_f1.append(f)
            per_image_p.append(p if pred else (1.0 if not ref else 0.0))
            per_image_r.append(r if ref else 1.0)
        ids.append(iid)
    p, r, f = prf(TP, FP, FN)
    label_f1 = {l: prf(*c)[2] for l, c in per_label.items() if c[0] + c[2] > 0}
    return {"micro_p": p, "micro_r": r, "micro_f1": f,
            "macro_f1": float(np.mean(list(label_f1.values()))) if label_f1 else 0.0,
            "n_images": len(ids), "tp": TP, "fp": FP, "fn": FN,
            "per_image": {"ids": ids, "f1": per_image_f1, "p": per_image_p, "r": per_image_r},
            "per_label_f1": label_f1}


def chair(preds: dict[str, set], refs: dict[str, set]) -> dict:
    """CHAIR-style hallucination over the 80 COCO categories (exhaustive references only)."""
    mentioned = halluc = imgs_h = 0
    per_image = []
    for iid, ref in refs.items():
        pred = preds.get(iid, set())
        h = len(pred - ref)
        mentioned += len(pred)
        halluc += h
        imgs_h += int(h > 0)
        per_image.append(h / len(pred) if pred else 0.0)
    return {"chair_i": halluc / mentioned if mentioned else 0.0, "chair_s": imgs_h / len(refs) if refs else 0.0,
            "n_mentions": mentioned, "n_hallucinated": halluc, "per_image": per_image}


def color_accuracy(pred_colors: dict[str, dict[str, set]], ref_colors: dict[str, dict[str, list]]) -> dict:
    """For clothing items present in both prediction and reference (with reference colours)."""
    n = correct = 0
    for iid, rc in ref_colors.items():
        pc = pred_colors.get(iid, {})
        for item, cols in rc.items():
            if not cols or item not in pc or not pc[item]:
                continue
            n += 1
            correct += int(bool(set(pc[item]) & set(cols)))
    return {"color_acc": correct / n if n else float("nan"), "n_pairs": n}


_ART = re.compile(r"\b(a|an|the)\b")
_NUM = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
        "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def vqa_normalize(a: str) -> str:
    a = a.lower().strip().rstrip(".")
    a = re.sub(r"[^\w\s]", "", a)
    a = _ART.sub(" ", a)
    toks = [_NUM.get(t, t) for t in a.split()]
    return " ".join(toks)


def vqa_accuracy(pred: str, answers: list[str]) -> float:
    """Standard VQA accuracy, averaged over the 10 leave-one-out subsets of human answers."""
    p = vqa_normalize(pred)
    ans = [vqa_normalize(x) for x in answers]
    accs = []
    for i in range(len(ans)):
        others = ans[:i] + ans[i + 1:]
        accs.append(min(1.0, sum(o == p for o in others) / 3.0))
    return float(np.mean(accs)) if accs else 0.0


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
