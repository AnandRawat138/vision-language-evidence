# Qualitative test cases (test split)

Selection rules are in `scripts/make_case_figure.py`. TP = label in the original reference annotation; FP = not in an exhaustive reference (hallucination); FN = reference label not produced; n/a = no exhaustive reference. p = calibrated score; [review] = flagged for human review.

## Case A: coco_000000076416
* Source: coco id 76416, licence: Attribution License, URL: http://images.cocodataset.org/val2017/000000076416.jpg
* Quality flags: none
* Reference objects: bicycle, bus, cell phone, person; reference garments: —

```
 TP  bus  p=0.97
n/a  red jacket  p=0.56
n/a  black sweater  p=0.48
 FN  bicycle  (missed)
 FN  cell phone  (missed)
 FN  person  (missed)
```

## Case B: coco_000000060899
* Source: coco id 60899, licence: Attribution License, URL: http://images.cocodataset.org/val2017/000000060899.jpg
* Quality flags: none
* Reference objects: book, chair, horse, person; reference garments: —

```
 TP  person  p=0.91
 FP  couch  p=0.93
 TP  chair  p=0.93
n/a  blue shorts  p=0.80
n/a  white socks  p=0.82
 FN  book  (missed)
 FN  horse  (missed)
```

## Case C: vg_107925
* Source: vg id 107925, licence: CC BY 4.0 (Visual Genome), URL: https://cs.stanford.edu/people/rak248/VG_100K/107925.jpg
* Quality flags: none
* Reference objects: apple, person, tie; reference garments: {'dress': ['blue'], 'shirt': []}

```
n/a  apple  p=0.94
 TP  blue dress  p=0.48
 TP  white shirt  p=0.39
 FP  black jacket  p=0.53
```

## Case D: coco_000000076416 (blur:5)
* Source: coco id 76416, licence: Attribution License, URL: http://images.cocodataset.org/val2017/000000076416.jpg
* Quality flags: blur
* Reference objects: bicycle, bus, cell phone, person; reference garments: —

```
 TP  bus  p=0.77 [review]
 FN  bicycle  (missed)
 FN  cell phone  (missed)
 FN  person  (missed)
```
