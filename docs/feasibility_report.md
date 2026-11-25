# Milestone 1 — Feasibility report (measured 2026-11-22)

## Machine (inspected, not assumed)

| Item | Observed |
|---|---|
| CPU | AMD Ryzen 9 5900HX (16 logical cores) |
| RAM | 16 GB (16,539,463,680 bytes) |
| dGPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB, driver 576.02 (CUDA 12.9 driver API), WDDM mode, ~0.4 GB already used by the desktop |
| iGPU | AMD Radeon Graphics (not used) |
| Storage | E: 239 GB free (project + model cache `E:\hf_cache`), C: 58 GB free |
| OS | Windows 11 Home 10.0.26200 |
| Python | 3.14 (default), 3.12.7 used for the project venv (`.venv`) |
| PyTorch | system Python had a CPU-only build (2.10.0+cpu); project venv: torch 2.7.1+cu126 (CUDA available) |
| Libraries | transformers 4.57.3, bitsandbytes 0.50.2, accelerate 1.12.0 |
| Node | v25.1.0 (React/Vite dashboard) |
| TeX | MiKTeX (pdflatex, bibtex, latexmk available) |
| WSL2 | installed but **no Linux distribution** present |
| Network | ~1.4 MB/s from Hugging Face (measured with a 100 MB range download) |

## Model feasibility (measured, `results/raw/feasibility/`)

* **Qwen2.5-VL-3B-Instruct** (revision 66285546d2b821cf421d4f5eb2576359d3770cd3). BF16 weights are
  ~7.5 GB, which exceeds 6 GB VRAM, so FP16/BF16 inference does not fit. **bitsandbytes NF4** (double quantisation,
  FP16 compute) fits: 2.30 GB allocated after loading; peak 2.41 GB (B=1), 2.61 GB (B=4), 2.85 GB (B=8) at
  `max_pixels = 448×448` (~256 visual tokens per image, ~520 input tokens incl. prompt).
* Decoding speed is low (≈12 output tokens/s for a single stream, 26.7 s for a 320-token output), which is typical of
  NF4 de-quantisation on a laptop GPU. Batching raises throughput (0.037 → 0.29 images/s from B=1 to B=8 in the probe).
* Full-vocabulary score storage (`output_scores=True`) would require B·V·T·4 bytes (~12 GB at B=64, T=320); token
  log-probabilities are therefore recorded on the fly by a logits processor (one step of log-softmax kept).
* Multi-image input: Qwen2.5-VL supports several images per prompt; we cap contextual requests at 4 images.

## vLLM

vLLM does not support native Windows, and WSL2 has no distribution installed. Installing a distribution is a system
change that was **not** performed without the owner's consent. Therefore:

* a vLLM backend (`src/vle/inference/vllm_backend.py`, OpenAI-compatible API, same request/response interface) and a
  cloud configuration (`configs/cloud_vllm.yaml`) are implemented but **were not executed**;
* all reported measurements use the Hugging Face Transformers backend with static left-padded batching on the local GPU.
  No result in the manuscript is attributed to vLLM.

## Datasets

All four sources were reachable from their official hosts (COCO 2017 annotations and images, Visual Genome v1.2 JSON
and images, VQA v2 val annotations on S3, Flickr30K Entities on GitHub). Flickr30K images are only distributed through
a request form or mirrors; individual images were extracted by HTTP range requests from the Hugging Face mirror
`nlphuji/flickr30k` (no full 4.4 GB download). See `docs/datasets.md` for licences.

## Decision

The project is feasible locally with a 3B VLM in NF4, at the cost of low absolute throughput. Experiments were sized
accordingly (Exp. 2 workload: 64 images × 3 repeats; Exp. 5: 200 images × 10 conditions; Exp. 3: 160 groups).
