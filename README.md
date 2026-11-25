# VLE — Batched, uncertainty-aware vision-language evidence description

Repository: https://github.com/AnandRawat138/vision-language-evidence

Research code, data manifests, raw experiment outputs and manuscript for the study
*"Batched, Uncertainty-Aware Vision-Language Evidence Description on a 6 GB GPU: Throughput, Calibration and
Cross-Image Consolidation"* (manuscript prepared for SN Computer Science).

**Scope.** The software describes observable content only (objects, garments, accessories, carried objects, scene,
image quality). It contains **no** face recognition, identity matching, re-identification, person ranking or line-up
generation, and it never assumes that different images show the same person.

## Layout

```
configs/            local_gpu.yaml (HF, NF4), cloud_vllm.yaml (vLLM, not executed), cpu_smoke.yaml, prompts.yaml
data/manifests/     manifest.csv (1,000 images), exclusions.csv, manifest_all_candidates.json (meta, rules, versions)
data/annotations/   reference/ (derived from ORIGINAL annotations only), new/ (empty: no new annotation)
docs/               feasibility_report.md, datasets.md, annotation_schema.md, human_evaluation_protocol.md, PROGRESS.md
src/vle/            datasets (A), narrative (B), preprocessing (C), inference (D), batching (E),
                    aggregation (F), uncertainty (G), api + reporting (H), evaluation, stats
frontend/           React (Vite) review dashboard
experiments/        run_inference.py, batching/run_exp2.py, narrative/run_exp4.py, baselines/,
                    analysis/analyze_exp{1,2,3,5}.py, human_evaluation/simulate_review.py, prompt_dev_check.py
scripts/            prepare_data.py, run_gpu_queue.ps1, make_figures.py, make_data_table.py,
                    verify_references.py, ingest_review_store.py, smoke_gpu.py, package.py
tests/              unit, scheduler and API tests (CPU only)
results/raw/        raw JSONL outputs, run configs, per-run stats, logs  (source of every reported number)
results/processed/  metric summaries (*.json);  results/tables/*.tex;  results/figures/*.pdf|png
paper/              main.tex (JRTIP, svjour3 two-column), main_neutral.tex, sections/, tables/, figures/, references.bib
```

## Setup (Windows, Python 3.12)

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements/requirements.txt
```

CPU-only smoke test (no GPU, no weights):

```bash
.venv\Scripts\python -m pytest tests -q
```

## Reproduce

```bash
.venv\Scripts\python scripts/prepare_data.py                 # downloads annotations must exist in data/raw (see docs/datasets.md)
.venv\Scripts\python scripts/smoke_gpu.py --batches 1 4 8     # local-GPU feasibility probe
powershell -ExecutionPolicy Bypass -File scripts/run_gpu_queue.ps1   # all GPU experiments (~7-9 h on RTX 3060 Laptop)
.venv\Scripts\python experiments/analysis/analyze_exp1.py     # E1, E6, E7 (needs exp1 + baselines)
.venv\Scripts\python experiments/analysis/analyze_exp2.py     # E2
.venv\Scripts\python experiments/analysis/analyze_exp3.py     # E3
.venv\Scripts\python experiments/narrative/run_exp4.py --phase analyze   # E4
.venv\Scripts\python experiments/analysis/analyze_exp5.py     # E5
.venv\Scripts\python experiments/human_evaluation/simulate_review.py  # E8 (oracle simulation only)
.venv\Scripts\python scripts/make_data_table.py
.venv\Scripts\python scripts/make_figures.py
cd paper && latexmk -pdf main.tex && latexmk -pdf main_neutral.tex
```

Environment variable `HF_HOME` selects the model cache (we used `E:\hf_cache`). Model: `Qwen/Qwen2.5-VL-3B-Instruct`
revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, NF4 (bitsandbytes), `max_pixels = 448*448`, greedy decoding.

## Review dashboard

```bash
.venv\Scripts\python scripts/ingest_review_store.py
cd frontend && npm install && npm run build && cd ..
.venv\Scripts\python -m uvicorn vle.api.app:app --app-dir src --port 8008    # open http://127.0.0.1:8008
```

## Data and licences

Images are **not** redistributed. `data/manifests/manifest.csv` lists source ids, URLs, MD5 and dHash values;
`scripts/prepare_data.py` re-downloads them from the official hosts. See `docs/datasets.md`.

## Status and main results (test split)

* E1: structured objects P 0.91 / R 0.40 / F1 0.55 (CHAIR_i 8.9 %); free-form captions of the same VLM F1 0.69;
  garments F1 0.47, colour accuracy 0.83.
* E2: batching 10.4× faster (0.047 → 0.491 images/s images/s at B=16), 9.97× in output tokens/s; output text identical to the B=16 run for 62/64 images at B=64 and 24/24 at B=1; B=64 slowed sharply without an OOM error.
* E3: consolidation of four captures F1 0.69 vs 0.55 for a single view; contextual 4-image prompting 0.50.
* E4: rule structurer garment F1 0.82, never upgrades uncertainty; LLM structurer 0.65, needs grounding.
* E5: occlusion is the most damaging degradation (−0.22 F1) and is not detected by quality flags or the model.
* E6: calibrated object probabilities, ECE 0.031, AUROC 0.80. E7: only JSON validation significant.
* E8: no user study; oracle simulation plus a blinded check of 59 object labels by one author (56/59 agree with the references). vLLM backend implemented but **not executed**.

## Licence of third-party material

The MIT licence covers the code only. Reference files under `data/annotations/` contain labels and captions derived from
MS COCO, VQA v2, Visual Genome and Flickr30K / Flickr30K Entities, and model outputs under `results/raw/` may quote them.
These remain under the terms of the original datasets (Flickr30K: non-commercial research use).
No images are redistributed. See `docs/completion_report.md` for status notes.
