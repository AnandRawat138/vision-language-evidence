"""Audit of claims whose support was doubtful. Prints evidence only; changes nothing."""
import json, sys, re, glob, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vle.runio import read_jsonl
from vle.evaluation import pipeline as P
from vle.lexicon import map_term

man = P.manifest_index()
test = [u for u in man if man[u]["split"] == "test"]
refs = P.references(list(man))

print("== A. BLIP vs structured: are predictions really different?")
blip = {json.loads(l)["uid"]: json.loads(l)["caption"] for l in (ROOT/"results/raw/baselines/b1_b2/blip_captions.jsonl").read_text(encoding="utf-8").splitlines()}
obs = collections.defaultdict(set)
for l in (ROOT/"results/processed/exp1_observations.jsonl").read_text(encoding="utf-8").splitlines():
    o = json.loads(l)
    if o["category"] == "object" and o["canonical"]:
        obs[o["provenance"]["image_id"]].add(o["canonical"])
ou = [u for u in test if refs[u]["objects_complete"]]
def cnt(pred):
    tp = sum(len(pred(u) & set(refs[u]["objects"])) for u in ou); fp = sum(len(pred(u) - set(refs[u]["objects"])) for u in ou)
    fn = sum(len(set(refs[u]["objects"]) - pred(u)) for u in ou); return tp, fp, fn
print("  BLIP tp/fp/fn", cnt(lambda u: P.caption_label_sets(blip[u])["object"]), "struct", cnt(lambda u: obs[u]), "n", len(ou))
print("  identical sets on", sum(P.caption_label_sets(blip[u])["object"] == obs[u] for u in ou), "of", len(ou))

print("== B. recorded model revision / prompt per run")
for f in sorted(glob.glob(str(ROOT/"results/raw/*/*/config.json"))):
    c = json.load(open(f)); b = c.get("backend", {})
    print("  ", Path(f).parent.relative_to(ROOT/"results/raw"), "| rev", b.get("model_revision") if isinstance(b, dict) else b, "| prompt", bool(c.get("prompt")), "| git", (c.get("environment") or {}).get("git_commit"))

print("== C. Exp2 workloads: output tokens of B=1 images vs all 32")
b1 = read_jsonl(ROOT/"results/raw/exp2/offline/independent_b1_r0.jsonl"); b16 = read_jsonl(ROOT/"results/raw/exp2/offline/static_b16_r0.jsonl")
print("  mean tokens B=1(16 img)", np.mean([r["n_output_tokens"] for r in b1]), " B=16(32 img)", np.mean([r["n_output_tokens"] for r in b16]))

print("== D. online: arrival sequences / request order identical across repeats? spans")
for f in sorted(glob.glob(str(ROOT/"results/raw/exp2/online/*.stats.json"))):
    s = json.load(open(f)); print("  ", s["tag"], "wall", s["wall_time_s"], "ok", s["n_ok"], "gpu peak", s.get("gpu_mem_peak_mb"))

print("== E. contradictions ever flagged in stored observations?")
reasons = collections.Counter()
for l in (ROOT/"results/processed/exp1_observations.jsonl").read_text(encoding="utf-8").splitlines():
    for r in json.loads(l)["uncertainty"]["review_reasons"]: reasons[r.split(":")[0]] += 1
print("  ", dict(reasons))

print("== F. Exp4: Entities garments missed by the rule structurer (are they all outside the lexicon?)")
sys.path.insert(0, str(ROOT/"experiments/narrative"))
import run_exp4 as X
from vle.narrative.structurer import RuleStructurer
from vle.schemas import Category
rs = RuleStructurer(); miss = []
for s in [s for s in X.flickr_sentences() if s["split"] == "test"]:
    r = rs.structure(s["text"], narrative_id=s["nid"]); sp = [X.span_of(o) for o in r.observations if o.category == Category.CLOTHING]
    for p in s["phrases"]:
        if "clothing" in p["types"] and not any(a and a[0] < p["end"] and a[1] > p["start"] for a in sp): miss.append(p["phrase"])
print("  missed", len(miss), "of which map to clothing lexicon", sum(map_term(m, "clothing") is not None for m in miss)); print("  ", miss[:15])
print("   sentences per image in test:", collections.Counter(collections.Counter(s["uid"] for s in X.flickr_sentences() if s["split"]=="test").values()))

print("== G. COCO val label: structured output confidence fields")
print("== H. duplicates across same Flickr/COCO handled by dhash; VQA questions evaluated anywhere?", bool(re.search("vqa_accuracy", "".join(open(f, encoding="utf-8").read() for f in glob.glob(str(ROOT/"experiments/**/*.py"), recursive=True)))))
