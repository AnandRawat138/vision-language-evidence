"""Unit tests: lexicon, schema validation, parsing, narrative, aggregation, uncertainty, metrics, stats."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from vle.aggregation.aggregator import AggregationConfig, aggregate, kept_label_sets
from vle.datasets.manifest import ManifestEntry, deduplicate, dhash, hamming
from vle.datasets.sources import parse_flickr_sentence
from vle.evaluation.metrics import chair, color_accuracy, evaluate_sets, jaccard, vqa_accuracy
from vle.evidence import label_sets, observations_from_vlm
from vle.inference.base import InferenceResult
from vle.lexicon import canon_color, find_terms, map_term
from vle.narrative.structurer import LLMStructurer, RuleStructurer
from vle.parsing import parse_naive, parse_structured
from vle.preprocessing.quality import assess_quality, crop_view, degrade
from vle.schemas import Category, Certainty, Observation, Provenance, UncertaintyInfo, VLMStructuredOutput
from vle.stats.tests import bootstrap_ci, cliffs_delta, cohens_dz, holm, paired_bootstrap
from vle.uncertainty.calibration import Combiner, brier, ece, risk_coverage, threshold_for_precision


# ----------------------------------------------------------------- lexicon
def test_map_term_synonyms_and_plural():
    assert map_term("two men", "coco") == "person"
    assert map_term("Sneakers", "clothing") == "shoes"
    assert map_term("a red T-shirt", "clothing") == "shirt"
    assert map_term("spaceship", "coco") is None
    assert canon_color("navy blue jacket") == "blue"
    assert canon_color("grey") == "gray"


def test_find_terms_no_false_top():
    assert "shirt" not in find_terms("a cat on top of a car", "clothing")


# ------------------------------------------------------------------ schema
def test_schema_coercion():
    m = VLMStructuredOutput.model_validate({"objects": ["Dog", {"item": "car", "confidence": "VERY HIGH"}],
                                            "scene": {"setting": "Outside"}, "image_quality": {"blur": "lots"}})
    assert [o.name for o in m.objects] == ["dog", "car"]
    assert m.objects[1].confidence == "medium"
    assert m.scene.setting == "unclear"
    assert m.image_quality.blur == "none"


# ----------------------------------------------------------------- parsing
def test_parse_fenced_and_repair():
    txt = 'Here:\n```json\n{"objects": [{"name": "dog", "confidence": "high"},], "clothing": []}\n```'
    m, status, _ = parse_structured(txt)
    assert m is not None and status == "repaired" and m.objects[0].name == "dog"
    assert parse_naive(txt)[1] == "failed"


def test_parse_truncated():
    m, status, _ = parse_structured('{"objects": [{"name": "dog"}, {"name": "ca')
    assert m is not None and status == "repaired"


def test_parse_garbage():
    assert parse_structured("no json here")[1] == "failed"


# ---------------------------------------------------------------- evidence
def test_observations_have_provenance_and_token_conf():
    raw = '{"objects": [{"name": "dog", "confidence": "high"}], "clothing": [{"name": "jacket", "color": "red"}]}'
    toks = [raw[i:i + 3] for i in range(0, len(raw), 3)]
    res = InferenceResult(request_id="r", text=raw, tokens=toks, token_logprobs=[-0.1] * len(toks))
    parsed, _, _ = parse_structured(raw)
    obs = observations_from_vlm(parsed, res, "img1", {"model_id": "m", "model_revision": "x", "quantization": "nf4"},
                                "struct-v1", "run1")
    assert {o.canonical for o in obs} == {"dog", "jacket"}
    for o in obs:
        assert o.provenance.image_id == "img1" and o.provenance.prompt_version == "struct-v1"
        assert o.provenance.model_name == "m" and o.provenance.timestamp
        assert 0 < o.uncertainty.token_confidence <= 1
        assert o.review_status.value == "pending"
    assert [o for o in obs if o.canonical == "jacket"][0].attribute == "red"
    ls = label_sets(obs)
    assert ls["object"] == {"dog"} and ls["clothing_colors"]["jacket"] == {"red"}


# --------------------------------------------------------------- narrative
@pytest.mark.parametrize("text,item,cert", [
    ("He wore a red jacket.", "jacket", Certainty.DEFINITE),
    ("I think he had a blue cap.", "hat", Certainty.HEDGED),
    ("Maybe a backpack, I'm not sure.", "bag", Certainty.HEDGED),
    ("She was not carrying an umbrella.", "umbrella", Certainty.NEGATED),
    ("The shirt was black or dark blue.", "shirt", Certainty.HEDGED),
])
def test_rule_structurer_certainty(text, item, cert):
    r = RuleStructurer().structure(text)
    assert r.original_text == text
    o = [o for o in r.observations if o.canonical == item]
    assert o, r.observations
    assert o[0].uncertainty.linguistic_certainty == cert


def test_rule_structurer_missing_and_color():
    r = RuleStructurer().structure("A man in a green sweater stood on the street at night.")
    sw = [o for o in r.observations if o.canonical == "sweater"][0]
    assert sw.attribute == "green"
    assert "accessories" in r.missing and "lighting" not in r.missing and "scene" not in r.missing


def test_llm_structurer_never_upgrades_and_drops_ungrounded():
    text = "I think she wore a yellow raincoat."
    out = json.dumps({"clothing": [{"name": "raincoat", "color": "yellow", "certainty": "definite", "quote": "yellow raincoat"},
                                   {"name": "boots", "certainty": "definite", "quote": "boots"}]})
    r = LLMStructurer(backend=None).from_output(text, out, "n1")
    assert r.unsupported_dropped == 1
    assert len(r.observations) == 1
    assert r.observations[0].uncertainty.linguistic_certainty == Certainty.HEDGED


def test_flickr_sentence_spans():
    s = parse_flickr_sentence("[/EN#1/people A man] in [/EN#2/clothing a red shirt] walks")
    ph = s["phrases"][1]
    assert s["sentence"][ph["start"]:ph["end"]] == "a red shirt"


# ------------------------------------------------------------- aggregation
def _obs(img, cat, lab, attr=None, score=None, flags=None, src="vlm", cert=None):
    return Observation(category=cat, label=lab, canonical=lab, raw_text=lab, attribute=attr,
                       provenance=Provenance(source_type=src, source_id=img, image_id=img),
                       uncertainty=UncertaintyInfo(combined_score=score, image_quality_flags=flags or [],
                                                   linguistic_certainty=cert))


def test_aggregation_support_and_contradiction():
    per = {"a": [_obs("a", Category.OBJECT, "dog"), _obs("a", Category.CLOTHING, "shirt", "red")],
           "b": [_obs("b", Category.OBJECT, "dog"), _obs("b", Category.CLOTHING, "shirt", "blue")],
           "c": [_obs("c", Category.OBJECT, "cat")]}
    agg = aggregate("g", per, AggregationConfig(min_support_frac=0.5, use_scores=False))
    kept = kept_label_sets(agg)
    assert "dog" in kept["object"] and "cat" not in kept["object"]
    assert any(c.slot == "clothing:shirt:color" for c in agg.contradictions)
    assert "same person" in agg.disclaimer


def test_aggregation_quality_weighting():
    per = {"a": [_obs("a", Category.OBJECT, "car", flags=["blur", "low_light"])],
           "b": [_obs("b", Category.OBJECT, "bus")], "c": [_obs("c", Category.OBJECT, "bus")]}
    agg = aggregate("g", per, AggregationConfig(min_support_frac=0.2, use_scores=False))
    car = [d for d in agg.consolidated if d["label"] == "car"][0]
    assert car["weighted_support"] < 1 / 3


def test_narrative_vs_image():
    per = {"a": [_obs("a", Category.CLOTHING, "jacket", "black")]}
    narr = [_obs("n", Category.CLOTHING, "jacket", "red", src="narrative", cert=Certainty.DEFINITE),
            _obs("n", Category.CARRIED, "umbrella", src="narrative", cert=Certainty.NEGATED),
            _obs("n", Category.CARRIED, "bag", src="narrative", cert=Certainty.HEDGED)]
    agg = aggregate("g", per, narrative=narr)
    st = {d["label"]: d["status"] for d in agg.narrative_support}
    assert st == {"jacket": "attribute_mismatch", "umbrella": "not_observed", "bag": "not_observed"}
    assert any(c.kind == "narrative_vs_image" for c in agg.contradictions)


# ------------------------------------------------------------- uncertainty
def test_calibration_metrics():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 1])
    perfect = y.astype(float)
    assert ece(perfect, y) == pytest.approx(0)
    assert brier(perfect, y) == 0
    conf = np.array([.9, .8, .3, .2, .7, .4, .6, .95])
    cov, risk, aurc = risk_coverage(conf, y)
    assert risk[0] == 0 and cov[-1] == 1 and 0 <= aurc <= 1
    t = threshold_for_precision(conf, y, 1.0)
    assert (y[conf >= t] == 1).all()


def test_combiner_fits():
    rng = np.random.default_rng(0)
    rows = [{"token": float(v), "verbal": 0.6, "ground": float(v) * 0.3, "n_quality_flags": 0, "is_object": 1, "is_clothing": 0}
            for v in rng.random(200)]
    y = np.array([int(r["token"] > 0.5) for r in rows])
    p = Combiner().fit(rows, y).predict(rows)
    assert ((p > 0.5) == y).mean() > 0.9


# ----------------------------------------------------------------- metrics
def test_set_metrics_and_chair():
    preds = {"a": {"dog", "cat"}, "b": set()}
    refs = {"a": {"dog"}, "b": set()}
    m = evaluate_sets(preds, refs)
    assert m["tp"] == 1 and m["fp"] == 1 and m["per_image"]["f1"][1] == 1.0
    c = chair(preds, refs)
    assert c["chair_i"] == 0.5 and c["chair_s"] == 0.5
    assert jaccard(set(), set()) == 1
    assert color_accuracy({"a": {"shirt": {"red"}}}, {"a": {"shirt": ["red", "white"]}})["color_acc"] == 1


def test_vqa_accuracy():
    assert vqa_accuracy("Two", ["2"] * 10) == 1.0
    assert vqa_accuracy("yes", ["no"] * 10) == 0.0


# ------------------------------------------------------------------- stats
def test_stats():
    a = np.array([0.8, 0.7, 0.9, 0.85, 0.75] * 20)
    b = a - 0.1
    pb = paired_bootstrap(a, b, B=2000)
    assert pb["mean_diff"] == pytest.approx(0.1) and pb["p_bootstrap"] < 0.01
    lo, hi = bootstrap_ci(a, B=2000)
    assert lo <= a.mean() <= hi
    assert cliffs_delta(a, b) > 0
    assert cohens_dz(a, b + np.linspace(-0.01, 0.01, len(b))) > 1
    adj = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj["x"] == pytest.approx(0.03) and adj["y"] >= adj["z"]


# ------------------------------------------------------- images and dedup
def test_quality_and_degradation():
    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.random((240, 320, 3)) * 255).astype(np.uint8))
    q = assess_quality(img)
    assert "blur" not in q.flags and "low_resolution" in q.flags
    blurred = degrade(img, "blur", 4)
    assert assess_quality(blurred).blur_laplacian_var < q.blur_laplacian_var
    dark = degrade(img, "lowlight", 2.5)
    assert np.asarray(dark).mean() < np.asarray(img).mean()
    for k, lv in [("downscale", 0.25), ("occlusion", 0.25), ("jpeg", 10)]:
        assert degrade(img, k, lv).size == img.size
    c, box = crop_view(img, "left")
    assert c.size == (224, 168) and box[0] == 0
    # source image unchanged
    assert assess_quality(img).blur_laplacian_var == q.blur_laplacian_var


def test_dedup():
    rng = np.random.default_rng(1)
    a = Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8)).resize((256, 256))
    b = a.copy()
    assert hamming(dhash(a), dhash(b)) == 0
    es = [ManifestEntry(uid=f"x{i}", source="coco", source_id=str(i), source_split="", url="", md5=m, dhash=d, status="ok")
          for i, (m, d) in enumerate([("1", dhash(a)), ("2", dhash(b)), ("1", "ffffffffffffffff")])]
    deduplicate(es)
    assert [e.status for e in es] == ["ok", "duplicate", "duplicate"]
