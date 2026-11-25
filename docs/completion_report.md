> **Superseded (25 Nov 2026).** Written for an earlier submission; some statements were corrected in the manuscript.
> The README summary is current. `paper/` and `deliverables/` are not part of this repository.

# Project completion report

**Project:** Batched, uncertainty-aware vision-language evidence description on a 6 GB GPU
**Authors (paper):** Anand Rawat (corresponding), Ananya Agarwal, Anandi Singh — PSIT Kanpur
**Target journal:** Journal of Real-Time Image Processing (Springer), svjour3 two-column, 12-page limit
**Date:** 2026-11-23

## 1. What was implemented

| Module | Implementation | Location |
|---|---|---|
| A Dataset manager | sampling, download (incl. HTTP range extraction from the Flickr30K archive), PIL validation, MD5 + dHash de-duplication, dev/test splits, manifests, derived references | `src/vle/datasets/`, `scripts/prepare_data.py` |
| B Statement structuring | deterministic `rule-v1` (clause/cue scoping, colour attachment, missing slots) and grounded LLM `narrative-v1` | `src/vle/narrative/` |
| C Image engine | read-only loading, blur/light/contrast/resolution flags, 5 controlled degradations, crops | `src/vle/preprocessing/` |
| D Inference API | common request/response contract; HF Transformers NF4 backend with on-the-fly log-probs; vLLM OpenAI client; CPU mock | `src/vle/inference/` |
| E Batch scheduler | independent/static/sorted/dynamic/adaptive (multiplicative increase/decrease); OOM split, failure bisection, retries, timeout watchdog, JSONL checkpoint/resume, NVML/psutil monitor | `src/vle/batching/` |
| F Aggregation | quality- and score-weighted support, contradictions (colour, setting, statement vs image), duplicates, missing slots, no-identity disclaimer | `src/vle/aggregation/` |
| G Uncertainty | token/verbal/CLIP/quality signals, dev-fit logistic calibrator, ECE/Brier/AUROC/AURC, selective reporting, review flags | `src/vle/uncertainty/` |
| H Review | FastAPI + SQLite store with append-only audit trail; React dashboard (overview, browser, batch panel, observation review, consistency, performance, audit, report export) | `src/vle/api/`, `frontend/` |
| Evaluation & statistics | set metrics, CHAIR, colour accuracy, VQA accuracy; bootstrap CIs, paired bootstrap, Wilcoxon, Welch, Cohen's d_z, Cliff's δ, Holm | `src/vle/evaluation/`, `src/vle/stats/` |

## 2. What was tested

38 automated tests (CPU-only) pass: lexicon, schema coercion, JSON repair, provenance, statement certainty, aggregation and
contradictions, calibration metrics, statistics, degradations, de-duplication, scheduler (OOM split, failure bisection,
timeout, resume, adaptive, online dynamic) and API endpoints including the audit trail (`results/logs/pytest_report.txt`).
A local-GPU smoke test is in `scripts/smoke_gpu.py`. The dashboard was run and exercised in a browser (screenshot = Fig. S5).

## 3. Experiments

| Exp. | Status | Key result (test split) |
|---|---|---|
| E1 single-image understanding (700 images) + baselines B1–B6 | complete | Objects: P 0.91, R 0.40, F1 0.55, CHAIR_i 8.9 %; free-form caption of the same VLM F1 0.69 (significantly higher); garments F1 0.46–0.47 (best), colour acc. 0.83 |
| E2 batching (offline, 3 repeats) | complete | 10.4× throughput (0.047→0.491 img/s at B=16), near-identical outputs (62/64 at B=64); B=64 slowed sharply (spill inferred, not measured) (0.135 img/s); the adaptive controller finds B=16 but ramp-up halves end-to-end throughput on 32 images |
| E2 online (Poisson arrivals) | complete | see paper Sect. 9.2 / Table S6 |
| E3 multi-image (90 test groups) | complete | aggregation F1 0.69 vs single view 0.55; contextual 4-image prompt 0.50 (slower, conservative) |
| E4 statement structuring (525 sentences) | complete | rule F1 0.82 vs LLM 0.65 (n.s.); LLM: 58 % ungrounded items dropped, 3/24 negations upgraded |
| E5 robustness (120 images × 10 conditions) | complete | occlusion −0.22 F1, undetected by quality flags and by the model; blur/low-light/down-scaling flagged ≥89 % |
| E6 calibration | complete | combined calibrator ECE 0.031, AUROC 0.80 (objects); garments weak (AUROC ≤0.59) |
| E7 ablation | complete | only structured validation significant (objects −0.016 F1, Holm p=0.006) |
| E8 human-in-the-loop | **incomplete** | no user study; one-author blinded check of 59 labels plus oracle simulation: ranked review finds 66 % of object errors after 30 % vs 31 % random |
| vLLM throughput | **not executed** | backend + config implemented; no Linux/WSL distribution available locally |

Workload sizes for E2/E3/E5 were reduced to meet a 5-hour compute budget (documented in the paper and `docs/PROGRESS.md`).

## 4. Hypotheses

* **H1** (batching improves throughput): supported (10.4×, p<1e-4).
* **H2** (aggregation improves consistency/quality): supported for completeness (F1 +0.137, p<1e-8); precision gain only at a non-zero threshold (descriptive).
* **H3** (uncertainty improves reliability and review): supported for objects (calibration, review ranking); not for F1 and not for garments.
* **H4** (adaptive scheduling improves utilisation within constraints): partly supported (finds optimum safely; worse end-to-end on short workloads).

## 5. Integrity notes

* All paper numbers are produced by scripts from `results/raw/`; tables/figures are generated, not typed.
* Bugs found during the work and fixed before the reported analysis: request metadata overwritten in checkpoints (quality flags had not been attached), scene observations wrongly filtered by the object threshold, missing provenance span for LLM scene observations. All reported numbers are from the corrected pipeline.
* One online run (independent, 0.1 Hz, repeat 0) overlapped an analysis job on the GPU and was re-run; the contaminated output is kept as `*.contaminated*.bak`.
* No annotations, citations or results were fabricated; 57 references were checked against Crossref/arXiv.
* The software and an initial draft were produced with AI assistance (disclosed in the manuscript declarations).

## 6. Submission readiness (honest assessment)

* **Format:** JRTIP svjour3 two-column, 12 pages incl. references, abstract ≤250 words, 6 keywords, Statements & Declarations, numbered references, flat LaTeX ZIP, Word version in the journal template layout, cover letter addressing real-time issues.
* **Still required from the authors:** author contributions and competing-interest/funding statements (placeholders), a public repository link for data/code availability, ORCID iDs, final proof-reading. Please double-check all author e-mail addresses in the title block.
* **Scientific readiness:** the study is reproducible and honest, but its novelty is mainly in the systems integration and the careful negative/positive findings on consumer hardware. Reviewers are likely to ask for (i) a human study, (ii) vLLM/continuous-batching measurements on Linux, (iii) more than one VLM, and (iv) real investigative-style footage. It is **not** guaranteed to be accepted; the manuscript is ready for submission as a careful empirical systems paper, with those limitations stated.
