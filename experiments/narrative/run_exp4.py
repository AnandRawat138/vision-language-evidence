"""Experiment 4: natural-language observation structuring.

Data
  (a) Flickr30K sentences of the 150 benchmark images (5 per image); reference = Flickr30K Entities
      phrase annotations with character spans (types 'clothing' and 'scene'), i.e. original human labels.
  (b) A synthetic, template-generated certainty suite (explicitly synthetic; labels by construction)
      to test hedging/negation handling and the "never upgrade uncertainty" invariant.

Structurers: rule-v1 (deterministic) and llm-narrative-v1 (Qwen2.5-VL-3B text-only, validated + grounded).

    python experiments/narrative/run_exp4.py --phase run      # GPU: LLM structurer outputs
    python experiments/narrative/run_exp4.py --phase analyze
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from vle.batching.scheduler import BatchScheduler, SchedulerConfig  # noqa: E402
from vle.datasets.manifest import load_manifest  # noqa: E402
from vle.datasets.reference import load_reference  # noqa: E402
from vle.evaluation.metrics import prf  # noqa: E402
from vle.lexicon import canon_color, map_term  # noqa: E402
from vle.narrative.structurer import LLMStructurer, RuleStructurer  # noqa: E402
from vle.reporting.tables import ci, esc, fmt, write_table  # noqa: E402
from vle.runio import RAW, environment, read_jsonl, run_dir, write_json  # noqa: E402
from vle.schemas import Category, Certainty  # noqa: E402
from vle.stats import tests as S  # noqa: E402

PROC = ROOT / "results" / "processed"


def flickr_sentences() -> list[dict]:
    out = []
    for r in load_manifest(ROOT, source="flickr30k"):
        ref = load_reference(ROOT, r["uid"])
        # phrases are stored flat; re-split by sentence using the original Entities order
        for si, cap in enumerate(ref["captions"]):
            out.append({"nid": f"{r['uid']}#s{si}", "uid": r["uid"], "split": r["split"], "text": cap})
    # attach phrases per sentence (phrase spans are relative to their own sentence)
    from vle.datasets.sources import load_flickr_entities
    ent = load_flickr_entities(ROOT / "data" / "raw" / "flickr30k", sorted({o["uid"].split("_", 1)[1] for o in out}))
    for o in out:
        sid = int(o["nid"].split("#s")[1])
        sent = ent[o["uid"].split("_", 1)[1]]["sentences"][sid]
        assert sent["sentence"] == o["text"]
        o["phrases"] = sent["phrases"]
    return out


COLORS = ["red", "blue", "black", "white", "green", "yellow", "gray", "brown"]
ITEMS = [("jacket", "clothing"), ("shirt", "clothing"), ("hat", "clothing"), ("backpack", "carried"),
         ("umbrella", "carried"), ("glasses", "accessory"), ("dress", "clothing"), ("hoodie", "clothing")]
TEMPLATES = {
    "definite": ["The person wore a {c} {i}.", "There was a {c} {i}.", "Someone in the doorway had a {c} {i}."],
    "hedged": ["I think the person wore a {c} {i}.", "Maybe there was a {c} {i}, I'm not sure.",
               "It looked like a {c} {i}.", "The {i} was probably {c}.", "Possibly a {c} {i}, hard to tell.",
               "The {i} was {c} or maybe {c2}."],
    "negated": ["The person was not wearing a {i}.", "There was no {i}.", "I did not see a {i}."],
}


def synthetic_suite(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for (item, _cat), (lab, temps) in itertools.product(ITEMS, TEMPLATES.items()):
        for t in temps:
            c, c2 = rng.sample(COLORS, 2)
            text = t.format(c=c, c2=c2, i=item)
            out.append({"nid": f"syn{len(out):04d}", "text": text, "label": lab, "item": item,
                        "color": None if lab == "negated" else c})
    return out


def run_llm(args) -> None:  # type: ignore[no-untyped-def]
    from vle.inference import make_backend
    out = run_dir("exp4", "llm_narrative")
    cfg = yaml.safe_load(Path(args.config).read_text())
    be = make_backend(cfg)
    st = LLMStructurer(be)
    items = [(s["nid"], s["text"]) for s in flickr_sentences()] + [(s["nid"], s["text"]) for s in synthetic_suite()]
    reqs = st.build_requests(items)
    for r in reqs:
        r.request_id = r.image_ids[0] + "|narrative-v1"
    sc = SchedulerConfig(policy="sorted", batch_size=32, checkpoint_path=str(out / "outputs.jsonl"), timeout_s=1800)
    write_json(out / "config.json", {"backend": be.info(), "scheduler": sc.__dict__, "environment": environment(),
                                     "n_requests": len(reqs)})
    s = BatchScheduler(be, sc)
    stats = s.run(reqs)
    s.close()
    write_json(out / "stats.json", stats.summary())


def span_of(obs) -> tuple[int, int] | None:  # type: ignore[no-untyped-def]
    try:
        a, b = obs.provenance.evidence_ref.split(":")[-1].split("-")
        return int(a), int(b)
    except Exception:  # noqa: BLE001
        return None


def evaluate_structurer(results: dict[str, object], sents: list[dict], typ: str, cat: Category) -> dict:
    tp_ref = n_ref = tp_pred = n_pred = 0
    sem_ok = sem_n = col_ok = col_n = 0
    per_img: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for s in sents:
        r = results.get(s["nid"])
        preds = [o for o in (r.observations if r else []) if o.category == cat]
        pspans = [(span_of(o), o) for o in preds]
        refs = [p for p in s["phrases"] if typ in p["types"]]
        for p in refs:
            hit = [o for sp, o in pspans if sp and sp[0] < p["end"] and sp[1] > p["start"]]
            n_ref += 1
            per_img[s["uid"]][1] += 1
            if hit:
                tp_ref += 1
                per_img[s["uid"]][0] += 1
                rc = map_term(p["phrase"], "clothing" if typ == "clothing" else "scene")
                if rc:
                    sem_n += 1
                    sem_ok += int(any(o.canonical == rc for o in hit))
                rcol = canon_color(p["phrase"])
                if typ == "clothing" and rcol:
                    col_n += 1
                    col_ok += int(any(o.attribute == rcol for o in hit))
        for sp, o in pspans:
            n_pred += 1
            per_img[s["uid"]][3] += 1
            if sp and any(sp[0] < p["end"] and sp[1] > p["start"] for p in refs):
                tp_pred += 1
                per_img[s["uid"]][2] += 1
    prec = tp_pred / n_pred if n_pred else 0.0
    rec = tp_ref / n_ref if n_ref else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # image-level bootstrap CI on F1
    keys = list(per_img)
    arr = np.array([per_img[k] for k in keys], float)
    rng = np.random.default_rng(S.SEED)
    idx = rng.integers(0, len(keys), (S.DEFAULT_B, len(keys)))
    a = arr[idx].sum(1)
    P_ = a[:, 2] / np.maximum(a[:, 3], 1)
    R_ = a[:, 0] / np.maximum(a[:, 1], 1)
    F_ = 2 * P_ * R_ / np.maximum(P_ + R_, 1e-12)
    pim = []
    for k in keys:
        t_r, nr, t_p, npred = per_img[k]
        p_ = t_p / npred if npred else (1.0 if nr == 0 else 0.0)
        r_ = t_r / nr if nr else 1.0
        pim.append(2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)
    return {"precision": prec, "recall": rec, "f1": f1, "f1_ci": (float(np.percentile(F_, 2.5)), float(np.percentile(F_, 97.5))),
            "n_ref_phrases": n_ref, "n_pred": n_pred, "semantic_agreement": sem_ok / sem_n if sem_n else None, "n_semantic": sem_n,
            "color_agreement": col_ok / col_n if col_n else None, "n_color": col_n, "_per_image": pim, "_ids": keys}


def certainty_eval(results: dict, suite: list[dict]) -> dict:
    conf = defaultdict(lambda: defaultdict(int))
    upgrades = n_unc = found = 0
    for s in suite:
        r = results.get(s["nid"])
        canon = map_term(s["item"], "clothing") or map_term(s["item"], "accessories") or map_term(s["item"], "carried")
        obs = [o for o in (r.observations if r else []) if o.canonical == canon or o.raw_text.lower() == s["item"]]
        pred = obs[0].uncertainty.linguistic_certainty.value if obs else "missed"
        found += int(bool(obs))
        conf[s["label"]][pred] += 1
        if s["label"] in ("hedged", "negated"):
            n_unc += 1
            upgrades += int(pred == "definite")
    total = len(suite)
    acc = sum(conf[l][l] for l in conf) / total
    return {"accuracy": acc, "detected": found / total, "upgrade_rate": upgrades / n_unc if n_unc else 0.0,
            "n": total, "n_uncertain": n_unc, "confusion": {k: dict(v) for k, v in conf.items()}}


def analyze(args) -> None:  # type: ignore[no-untyped-def]
    sents = flickr_sentences()
    suite = synthetic_suite()
    rule = RuleStructurer()
    res_rule = {s["nid"]: rule.structure(s["text"], narrative_id=s["nid"]) for s in sents + suite}
    llm_rows = read_jsonl(RAW / "exp4" / "llm_narrative" / "outputs.jsonl")
    llm = LLMStructurer(None)
    texts = {s["nid"]: s["text"] for s in sents + suite}
    res_llm = {}
    parse = defaultdict(int)
    dropped = 0
    for r in llm_rows:
        nid = r["image_ids"][0]
        nr = llm.from_output(texts[nid], r["text"], nid)
        res_llm[nid] = nr
        parse[nr.parse_status] += 1
        dropped += nr.unsupported_dropped
    test = [s for s in sents if s["split"] == "test"]
    out = {"n_sentences_test": len(test), "n_images_test": len({s["uid"] for s in test}), "structurers": {}}
    pvecs = {}
    for name, res in (("rule-v1", res_rule), ("llm-narrative-v1", res_llm)):
        if not res:
            continue
        d = {"clothing": evaluate_structurer(res, test, "clothing", Category.CLOTHING),
             "scene": evaluate_structurer(res, test, "scene", Category.SCENE),
             "certainty_synthetic": certainty_eval(res, suite)}
        # missing-info detection on test: clothing reported missing iff no Entities clothing phrase
        mi_ok = mi_n = 0
        for s in test:
            r = res.get(s["nid"])
            if r is None:
                continue
            truth = not any("clothing" in p["types"] for p in s["phrases"])
            mi_n += 1
            mi_ok += int(("clothing" in r.missing) == truth)
        d["missing_clothing_accuracy"] = mi_ok / mi_n if mi_n else None
        pvecs[name] = d["clothing"].pop("_per_image"), d["clothing"].pop("_ids")
        d["scene"].pop("_per_image"), d["scene"].pop("_ids")
        out["structurers"][name] = d
    if len(pvecs) == 2:
        a_ids = pvecs["rule-v1"][1]
        b_map = dict(zip(pvecs["llm-narrative-v1"][1], pvecs["llm-narrative-v1"][0]))
        common = [i for i in a_ids if i in b_map]
        a = np.array([dict(zip(a_ids, pvecs["rule-v1"][0]))[i] for i in common])
        b = np.array([b_map[i] for i in common])
        out["paired_rule_vs_llm_clothing_f1"] = {**S.paired_bootstrap(a, b), "wilcoxon_p": S.wilcoxon(a, b)["p"], "n_images": len(common)}
        out["llm_parse_status"] = dict(parse)
        out["llm_unsupported_items_dropped"] = dropped
    write_json(PROC / "exp4_summary.json", out)
    rows = []
    for name, d in out["structurers"].items():
        c, sc, ce = d["clothing"], d["scene"], d["certainty_synthetic"]
        rows.append([esc(name), fmt(c["precision"]), fmt(c["recall"]), ci(c["f1"], *c["f1_ci"]),
                     fmt(c["semantic_agreement"]), fmt(c["color_agreement"]), fmt(sc["f1"]),
                     fmt(ce["accuracy"]), fmt(ce["upgrade_rate"])])
    write_table("exp4_narrative", ["Structurer", "Cloth. P", "Cloth. R", "Cloth. F1", "Class agr.", "Colour agr.", "Scene F1",
                                   "Cert. acc.$^\\dagger$", "Upgrade$^\\dagger$"], rows,
                f"Observation structuring of human-written Flickr30K sentences (test split, {out['n_sentences_test']} sentences) "
                "against Flickr30K Entities phrase annotations (span overlap). $^\\dagger$Synthetic template suite "
                f"({len(suite)} statements, labels by construction); upgrade = hedged/negated statement emitted as definite.",
                "tab:exp4", colspec="lcccccccc", size=r"\scriptsize")
    print(json.dumps(out, indent=1)[:3000])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["run", "analyze"], default="analyze")
    ap.add_argument("--config", default=str(ROOT / "configs" / "local_gpu.yaml"))
    args = ap.parse_args()
    run_llm(args) if args.phase == "run" else analyze(args)


if __name__ == "__main__":
    main()
