# Datasets, licences and benchmark construction

The benchmark is an *investigative-style evidence-description benchmark* built from general-purpose public visual
understanding datasets. **None of these datasets is a criminal-investigation dataset**, and the benchmark contains
no forensic material.

| Source | Version / files | Images used | Licence / terms | Redistributed here? |
|---|---|---|---|---|
| MS COCO | 2017, `annotations_trainval2017.zip`, val2017 images | 500 | Annotations CC BY 4.0; images under their Flickr licences (recorded per image in the manifest) | No — URLs + md5 in manifest |
| VQA v2 | val (v2_Questions/Annotations_Val_mscoco), COCO val2014 images present in train2017 | 100 | VQA annotations CC BY 4.0; COCO image terms | No |
| Visual Genome | v1.2 `image_data.json`, `attributes.json` | 250 | CC BY 4.0 | No |
| Flickr30K + Entities | Flickr30K images (HF mirror `nlphuji/flickr30k`), Entities annotations (GitHub), official test split | 150 | Flickr terms of use; non-commercial research use | No |

Download dates: 2026-11-22. Raw archives live in `data/raw/` and images in `data/images/` (both git-ignored).

## Selection rules (deterministic, seed 2026) — see `scripts/prepare_data.py`

* COCO: val2017 images with ≥1 instance and captions; 60 % with ≥1 person, 40 % without.
* VQA v2: ≥3 val questions, image file available in COCO train2017, not in the COCO subset.
* VG: images **without** a `coco_id` (avoids COCO overlap) having ≥1 object name mapped to the clothing lexicon.
* Flickr30K: random sample of the Flickr30K Entities official test split.
* 15 % oversampling per source; candidates validated (PIL decodable, min side ≥ 64 px), then exact (md5) and
  near-duplicate (64-bit dHash, Hamming ≤ 4) removal across all sources in priority COCO > VQA > VG > Flickr30K.
  Result: 0 invalid, 0 duplicates, 150 surplus reserves (listed in `data/manifests/exclusions.csv`).
* Splits: 30 % dev / 70 % test per source (image-disjoint). Dev is used for prompt choice, thresholds and calibrator
  fitting; **test labels are never used for any design decision**.

Final: 1,000 images = COCO 500 (150/350), VG 250 (75/175), Flickr30K 150 (45/105), VQA 100 (30/70).

## Annotation separation

* `data/annotations/reference/` — derived **only** from original dataset annotations (rules in `annotation_schema.md`).
* `data/annotations/new/` — reserved for new human annotations; **empty** (no new annotation was collected).
* `results/raw/**/outputs.jsonl` — model outputs. Never mixed with the above.
