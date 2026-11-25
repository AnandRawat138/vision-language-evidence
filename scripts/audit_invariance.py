"""Output invariance to batch size from ALL stored runs (no GPU). Writes results/processed/audit_invariance.json."""
import json, sys, glob, re
from pathlib import Path
import numpy as np
from scipy.stats import beta
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vle.runio import read_jsonl

def load(p): return {r["image_ids"][0]: r for r in read_jsonl(Path(p)) if not r.get("error")}
off = {Path(f).stem: load(f) for f in glob.glob(str(ROOT/"results/raw/exp2/offline/*.jsonl"))}
onl = {Path(f).stem: load(f) for f in glob.glob(str(ROOT/"results/raw/exp2/online/*.jsonl"))}
e1 = load(ROOT/"results/raw/exp1/struct_b16/outputs.jsonl")
b1 = {}
for k, v in {**off, **onl}.items():
    if "indep" in k:
        for u, r in v.items(): b1.setdefault(u, r["text"])
out = {"n_images_with_B1": len(b1)}
def cmp(a, b):
    c = [u for u in a if u in b]; s = sum(a[u] == b[u] for u in c)
    lo = float(beta.ppf(0.025, s, len(c)-s+1)) if s else 0.0
    return {"n": len(c), "identical": s, "cp95_lower": round(lo, 3)}
T = lambda v: {u: r["text"] for u, r in v.items()}
out["B1_vs_E1_B16"] = cmp(b1, T(e1))
for k in sorted(off):
    if k.endswith("_r0"): out[f"E1_B16_vs_{k}"] = cmp(T(e1), T(off[k])); out[f"B1_vs_{k}"] = cmp(b1, T(off[k]))
for k in sorted(onl): out[f"E1_B16_vs_{k}"] = cmp(T(e1), T(onl[k]))
# matched throughput: same 16 images cannot be isolated from batched runs -> compare output tokens/s
tok = {}
for f in glob.glob(str(ROOT/"results/raw/exp2/offline/*.stats.json")):
    s = json.load(open(f)); n = sum(r["n_output_tokens"] for r in off[Path(f).name.replace(".stats.json","")].values())
    tok.setdefault(re.sub(r"_r\d+$","",s["tag"]), []).append(n/s["wall_time_s"])
out["output_tokens_per_s"] = {k: [round(x,2) for x in v] for k, v in tok.items()}
out["token_speedup_b16_vs_b1"] = float(np.mean(tok["static_b16"])/np.mean(tok["independent_b1"]))
(ROOT/"results/processed/audit_invariance.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
