# Reference annotation schema (`ref-v1`, lexicon `lex-v1`)

No new annotations were created and none were fabricated. Every reference field is derived deterministically from an
original human annotation of the source dataset by `src/vle/datasets/reference.py`.

| Field | Derivation | Exhaustive? | Sources |
|---|---|---|---|
| `objects` | COCO instance categories present in the image | **Yes** (80 COCO classes) | COCO, VQA |
| `objects` (VG/Flickr) | VG object names / Entities phrases mapped to COCO classes | No | VG, Flickr30K |
| `object_boxes` | COCO boxes `[x,y,w,h]` | Yes | COCO, VQA |
| `clothing` | canonical clothing class → colours. VG: objects whose name maps to the clothing lexicon, colours from that object's VG attributes; Flickr30K: Entities phrases typed `clothing` mapped by head noun, colour adjectives within the phrase | No (precision is a lower bound) | VG, Flickr30K |
| `accessories`, `carried` | same procedure with the accessory / carried-object lexicons | No | VG, Flickr30K |
| `scene_places` | canonical place named by ≥ 2 of the image's human captions | No | COCO, VQA, Flickr30K |
| `captions` | original captions / sentences | — | COCO, VQA, Flickr30K |
| `flickr_phrases` | Entities phrases with types and character spans | — | Flickr30K |
| `vqa` | VQA v2 questions with 10 human answers | — | VQA |

Category definitions used across the project (lexicon in `src/vle/lexicon.py`):

* **Objects** – the 80 COCO categories, with CHAIR-style synonyms.
* **Clothing** – 13 canonical garment classes (shirt, jacket, sweater, pants, shorts, dress, skirt, suit, uniform,
  vest, shoes, socks, hat).
* **Accessories** – glasses, watch, jewelry, tie, scarf, gloves, belt, mask, badge.
* **Carried objects** – bag, suitcase, umbrella, phone, bottle, cup, camera, leash (+ sports equipment, food, book are
  recognised but excluded from evaluation because their references are inconsistent).
* **Image quality** – blur (Laplacian variance < 60), low light (mean grey < 50), low contrast (std < 25), low resolution
  (min side < 320). These are indicator flags, not ground truth.
* **Ambiguous / unsupported** – a model observation is *unsupported* when its canonical label is absent from an
  exhaustive reference (COCO objects; this is the CHAIR hallucination definition). For non-exhaustive references the
  term "reference-inconsistent" is used instead.

New-annotation protocol (prepared, **not executed**): two independent annotators per image using the review UI
(`frontend/`), the categories above, adjudication by a third annotator, Cohen's κ / Krippendorff's α per category.
