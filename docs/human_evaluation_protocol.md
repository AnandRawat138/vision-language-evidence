# Experiment 8 — Human-in-the-loop evaluation protocol (status: NOT EXECUTED)

No human participants took part in this project. The numbers in `results/processed/exp8_summary.json` come from an
**oracle simulation** that uses dataset annotations to rank review effort; they are not human results and are labelled
as such in the manuscript.

## Planned study (requires ethics approval and authorised reviewers)

* Participants: ≥ 3 reviewers trained on `docs/annotation_schema.md`. No victims or witnesses; no identification tasks.
* Material: 100 test images (stratified by source), framework observations with scores and flags.
* Conditions (within-subject, counterbalanced): (A) observations in random order without scores; (B) uncertainty-ranked
  queue with flags and CLIP/token/verbal signals shown.
* Logged automatically by the dashboard audit trail: action (accept/reject/correct/add), before/after, reviewer id,
  time per decision (`duration_ms`).
* Outcomes: correction rate; agreement of the reviewed record with reference annotations; unsupported-observation rate
  after review; review time per image; inter-annotator agreement (Fleiss' κ on accept/reject, Krippendorff's α on labels).
* Analysis: mixed-effects logistic regression (condition fixed, reviewer and image random), paired Wilcoxon on time.
* Excluded by design: identification accuracy, suspect selection, line-ups, any judgement about persons.
