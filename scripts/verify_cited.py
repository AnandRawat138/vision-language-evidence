"""Verify EVERY cited reference; non-DOI/arXiv items via Crossref bibliographic search and DBLP."""
import json, re, time, urllib.parse, urllib.request, glob, xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def get(u, n=3):
    for i in range(n):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "reference-check/0.1"}), timeout=40).read()
        except Exception as e:
            err = e; time.sleep(4 * (i + 1))
    raise err
bib = (ROOT/"paper/references.bib").read_text(encoding="utf-8")
cited = set()
for f in glob.glob(str(ROOT/"paper/sections/*.tex")):
    for m in re.findall(r"\\cite\{([^}]*)\}", Path(f).read_text(encoding="utf-8")): cited |= {x.strip() for x in m.split(",")}
out = {}
for m in re.finditer(r"@(\w+)\{(\w+),(.*?)\n  ?.*?\}\}\s*(?=\n@|\n%|\Z)|@(\w+)\{(\w+),(.*?)(?=\n@|\Z)", bib, re.S):
    key, body = (m.group(2), m.group(0)) if m.group(2) else (m.group(5), m.group(0))
    if key not in cited: continue
    t = re.search(r"title\s*=\s*\{(.*?)\}\s*,\s*\n", body, re.S); title = re.sub(r"[{}]", "", t.group(1)) if t else ""
    doi = re.search(r"doi\s*=\s*\{(.*?)\}\s*[,}]", body); ax = re.search(r"arXiv:(\d{4}\.\d{5})", body)
    rec = {"title": title}
    try:
        if doi:
            d = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi.group(1))))["message"]
            rec.update(src="crossref", found=d["title"][0], authors=[f"{a.get('family','')}, {a.get('given','')}" for a in d.get("author", [])], year=d["issued"]["date-parts"][0][0], venue=(d.get("container-title") or [""])[0], vol=d.get("volume"), page=d.get("page"), doi=doi.group(1))
        elif ax:
            e = ET.fromstring(get("https://export.arxiv.org/api/query?id_list=" + ax.group(1))).find("{*}entry")
            j = e.find("{*}journal_ref"); rec.update(src="arxiv", found=" ".join(e.find("{*}title").text.split()), authors=[a.find("{*}name").text for a in e.findall("{*}author")], year=e.find("{*}published").text[:4], arxiv=ax.group(1), jref=j.text if j is not None else None); time.sleep(3)
        else:
            d = json.loads(get("https://dblp.org/search/publ/api?format=json&h=3&q=" + urllib.parse.quote(title)))["result"]["hits"].get("hit", [])
            if d:
                i = d[0]["info"]; a = i.get("authors", {}).get("author", []); a = a if isinstance(a, list) else [a]
                rec.update(src="dblp", found=i["title"], authors=[x["text"] for x in a], year=i["year"], venue=i.get("venue"), page=i.get("pages"), url=i.get("ee"))
            else:
                d = json.loads(get("https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(title)))["message"]["items"]
                rec.update(src="crossref-search", candidates=[((x.get("title") or [""])[0], x.get("DOI"), (x.get("container-title") or [""])[0], x["issued"]["date-parts"][0][0]) for x in d])
    except Exception as e:
        rec["error"] = str(e)
    n = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    rec["title_match"] = n(rec.get("found", "")) == n(title)
    out[key] = rec; print(f"{key:24s} {rec.get('src')} match={rec['title_match']} {rec.get('year')} n_auth={len(rec.get('authors', []))} {rec.get('error','')} {rec.get('candidates','')}", flush=True)
print("cited", len(cited), "checked", len(out), "missing from parse", cited - set(out))
(ROOT/"results/raw/reference_verification_cited.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
