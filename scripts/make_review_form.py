"""Offline blinded review form (single HTML). Reviewer sees image + object label only: no score, no dataset label.
40 COCO test images whose reference has no 'person' (avoids identifiable people). Answers are exported as JSON."""
import base64, csv, io, json, random, sys, collections
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
from vle.datasets.reference import load_reference
man = {r["uid"]: r for r in csv.DictReader(open(ROOT/"data/manifests/manifest.csv", encoding="utf8"))}
obs = collections.defaultdict(list)
for l in open(ROOT/"results/processed/exp1_observations.jsonl", encoding="utf8"):
    o = json.loads(l)
    if o["category"] == "object" and o["canonical"] and o["canonical"] not in [x["label"] for x in obs[o["provenance"]["image_id"]]]:
        obs[o["provenance"]["image_id"]].append({"id": o["observation_id"], "label": o["canonical"]})
ids = [u for u in sorted(man) if man[u]["split"] == "test" and man[u]["source"] == "coco" and man[u]["has_person"] == "False" and obs[u]
       and "person" not in [x["label"] for x in obs[u]]]
pick = random.Random(2026).sample(ids, min(40, len(ids))); items = []
for u in pick:
    im = Image.open(ROOT/man[u]["local_path"]).convert("RGB"); im.thumbnail((640, 480)); b = io.BytesIO(); im.save(b, "JPEG", quality=82)
    src = "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    for x in obs[u]: items.append({"uid": u, "oid": x["id"], "label": x["label"], "src": src})
random.Random(7).shuffle(items)
imgs = {}; [imgs.setdefault(i["uid"], i.pop("src")) or i.pop("src", None) for i in items]
for i in items: i.pop("src", None)
html = """<!doctype html><meta charset=utf-8><title>Blind review</title><style>body{font:18px sans-serif;text-align:center;margin:20px}img{max-width:640px;max-height:440px}button{font-size:20px;margin:8px;padding:10px 24px}</style>
<div id=s><h2>Blind image review</h2><p>For each picture say whether the named object is visible. Work alone; do not discuss answers. You may stop at any time.</p>
<p>Taking part is voluntary. Only your code, answers and timings are saved, in a file you send yourself.</p><p>Reviewer code <input id=r> Are you an author? <select id=a><option>choose<option>yes<option>no</select></p><button onclick=go()>I agree, start</button></div>
<div id=m hidden><p id=p></p><img id=i><h2 id=q></h2><button onclick=ans('yes')>Visible (Y)</button><button onclick=ans('no')>Not visible (N)</button><button onclick=ans('unsure')>Cannot tell (U)</button></div>
<div id=e hidden><h2>Finished</h2><button onclick=save()>Save answers</button></div><script>
const I=__IMGS__,T=__ITEMS__;let k=0,t0,A=[],R,S;const $=x=>document.getElementById(x);
function go(){R=$('r').value.trim();if(!R||$('a').value=='choose'){alert('Enter your code and author status');return}S=new Date().toISOString();$('s').hidden=true;$('m').hidden=false;show()}
function show(){if(k>=T.length){$('m').hidden=true;$('e').hidden=false;save();return}$('p').textContent=(k+1)+' / '+T.length;$('i').src=I[T[k].uid];$('q').textContent='Is this visible: '+T[k].label+' ?';t0=performance.now()}
function ans(v){if($('m').hidden)return;A.push({uid:T[k].uid,oid:T[k].oid,label:T[k].label,answer:v,ms:Math.round(performance.now()-t0)});k++;show()}
onkeydown=e=>{let v={y:'yes',n:'no',u:'unsure'}[e.key.toLowerCase()];if(v)ans(v)};
function save(){let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({reviewer:R,author:$('a').value,started:S,ended:new Date().toISOString(),answers:A})]));a.download='review_'+R+'.json';a.click()}</script>"""
out = ROOT/"deliverables/human_review"; out.mkdir(parents=True, exist_ok=True)
(out/"blind_review_form.html").write_text(html.replace("__IMGS__", json.dumps(imgs)).replace("__ITEMS__", json.dumps(items)), encoding="utf8")
print(len(pick), "images", len(items), "judgements", (out/"blind_review_form.html").stat().st_size//1024, "KB")
