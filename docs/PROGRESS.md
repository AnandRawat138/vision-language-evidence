# Progress log

| Milestone | Status | Evidence |
|---|---|---|
| M1 Environment & feasibility | done | docs/feasibility_report.md, results/raw/feasibility/ |
| M2 Literature & dataset audit | done | paper/references.bib (57 entries verified via Crossref/arXiv: results/raw/reference_verification.json), docs/datasets.md |
| M3 Dataset & annotation framework | done | data/manifests/, data/annotations/reference/, docs/annotation_schema.md |
| M4 Baseline VLM | done | src/vle/inference/, experiments/baselines/ |
| M5 Batch inference & scheduling | done | src/vle/batching/, tests/test_scheduler.py |
| M6 Aggregation & uncertainty | done | src/vle/aggregation/, src/vle/uncertainty/ |
| M7 Dashboard & reporting | done | src/vle/api/, frontend/, src/vle/reporting/ |
| M8 Experiments | running (GPU queue) | results/logs/gpu_queue.log |
| M9 Statistics & figures | scripts ready | experiments/analysis/, scripts/make_figures.py |
| M10 Manuscript | in progress | paper/ |
| M11 Audit & packaging | pending | |

## Decisions recorded
* 2026-11-22: struct-v1 kept after dev-only comparison with struct-v2 (results/raw/prompt_dev/check/summary.json).
* Batch size 16 used for the full extraction runs (all images fit; B=64 tested in Exp. 2).
* Exp. 3 per-view requests executed batched (B=16) for all 160 groups; B=1 only for 30 test groups (time budget:
  NF4 single-stream decoding ~12 tok/s).
* vLLM not executed (no Linux distribution in WSL2; installing one is a system change not authorised).
* "Without structured validation" ablation = code-fence strip + json.loads, no repair.

## Known limitations (running list)
* No human study; Exp. 8 is an oracle simulation of review prioritisation only.
* Clothing/accessory references are not exhaustive (precision lower bound).
* Synthetic certainty suite and rule cues written by the same author (circular for rule-v1).
* GPU also drives the Windows desktop (WDDM), adding timing noise.

* 2026-11-22 13:07: budget cut to a 5-hour total: Exp2 32 images (B=1 on 16) + one B=64 run on 64 images; online 24 requests at 0.1/0.3 Hz; Exp3 120 groups (30/90), no B=1 subset (B=1 identity measured in Exp2); Exp5 120 images; B3 captions on test split only.

## Paused 2026-11-22 ~15:20 (user request)
Completed: Exp1 (+B1-B3 baselines), Exp2 offline, Exp3 per-view runs, Exp4, analyses E1/E2/E4/E6/E7/E8, results text for these.
Remaining GPU steps: exp3_context (24/120 groups checkpointed), exp5 (10 degradations x 120 images), exp2_online.
Resume: powershell -ExecutionPolicy Bypass -File scripts/run_gpu_queue.ps1   (completed steps are skipped automatically)
Then: analyze_exp3.py, analyze_exp5.py, analyze_exp2.py (online part), make_case_figure.py (case D), make_figures.py,
write res_e3/res_e5/online part of res_e2, discussion, conclusion, abstract; trim main paper to 12 pages; build PDF/DOCX; package ZIP.
