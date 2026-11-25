"""Module F: cross-image evidence consolidation and consistency analysis.

The aggregator works at the level of *observation labels*, never people: it
reports how many images support each label and where observations disagree.
It does not decide that observations in different images refer to the same
individual; colour disagreements are reported as consistency issues for human
review, with the explicit caveat that they may reflect different people or objects.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from vle.schemas import AggregatedEvidence, Category, Certainty, Contradiction, Observation

QUALITY_PENALTY = {"blur": 0.5, "low_light": 0.6, "low_contrast": 0.8, "low_resolution": 0.7}


def image_weight(flags: list[str]) -> float:
    w = 1.0
    for f in flags:
        w *= QUALITY_PENALTY.get(f, 1.0)
    return w


@dataclass
class AggregationConfig:
    min_support_frac: float = 0.5        # keep a label if weighted support >= this fraction of total weight
    use_quality_weights: bool = True
    use_scores: bool = True               # weight each vote by its observation score when available
    score_field: str = "combined_score"


def _score(o: Observation, field: str) -> float:
    s = getattr(o.uncertainty, field, None)
    if s is None:
        s = o.uncertainty.token_confidence
    return 1.0 if s is None else float(s)


def aggregate(group_id: str, per_image: dict[str, list[Observation]], cfg: AggregationConfig | None = None,
              narrative: list[Observation] | None = None) -> AggregatedEvidence:
    cfg = cfg or AggregationConfig()
    image_ids = list(per_image)
    weights = {}
    for iid, obs in per_image.items():
        flags = obs[0].uncertainty.image_quality_flags if obs else []
        weights[iid] = image_weight(flags) if cfg.use_quality_weights else 1.0
    total_w = sum(weights.values()) or 1.0

    support: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    obs_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    colors: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    settings: dict[str, list[str]] = defaultdict(list)
    duplicates: list[list[str]] = []
    for iid, obs in per_image.items():
        seen: dict[tuple[str, str], str] = {}
        for o in obs:
            if o.canonical is None:
                continue
            key = (o.category.value, o.canonical)
            if key in seen:
                duplicates.append([seen[key], o.observation_id])
            seen.setdefault(key, o.observation_id)
            vote = weights[iid] * (_score(o, cfg.score_field) if cfg.use_scores else 1.0)
            support[key][iid] = max(support[key].get(iid, 0.0), vote)
            obs_ids[key].append(o.observation_id)
            if o.category == Category.CLOTHING and o.attribute:
                colors[o.canonical][o.attribute].append(o.observation_id)
            if o.category == Category.SCENE and o.attribute in ("indoor", "outdoor"):
                settings[o.attribute].append(iid)

    consolidated = []
    for (cat, lab), sup in support.items():
        wsum = sum(sup.values())
        consolidated.append({"category": cat, "label": lab, "support_images": sorted(sup),
                             "n_support": len(sup), "support_frac": round(len(sup) / len(image_ids), 4),
                             "weighted_support": round(wsum / total_w, 4),
                             "kept": wsum / total_w >= cfg.min_support_frac,
                             "observation_ids": obs_ids[(cat, lab)]})
    consolidated.sort(key=lambda d: (-d["weighted_support"], d["category"], d["label"]))

    contradictions: list[Contradiction] = []
    for item, cmap in colors.items():
        if len(cmap) > 1:
            contradictions.append(Contradiction(
                slot=f"clothing:{item}:color", values={k: v for k, v in cmap.items()}, sources=image_ids,
                kind="cross_image" if len(image_ids) > 1 else "intra_source", severity="low",
                explanation="Different colours reported for the same clothing type; may be different people or "
                            "objects, or a perception error. Requires human review."))
    if len(settings) > 1:
        contradictions.append(Contradiction(slot="scene:setting", values={k: v for k, v in settings.items()},
                                            sources=image_ids, kind="cross_image", severity="high",
                                            explanation="Images disagree on indoor/outdoor setting."))

    narrative_support = []
    if narrative:
        img_labels = {k for k in support}
        for n in narrative:
            if n.canonical is None:
                continue
            key = (n.category.value, n.canonical)
            cert = n.uncertainty.linguistic_certainty
            seen_in = sorted(support.get(key, {}))
            status = "supported" if seen_in else "not_observed"
            if cert == Certainty.NEGATED and seen_in:
                status = "contradicted"
                contradictions.append(Contradiction(slot=f"{key[0]}:{key[1]}", values={"narrative:absent": [n.observation_id],
                                                    "image:present": obs_ids[key]}, sources=["narrative"] + seen_in,
                                                    kind="narrative_vs_image", severity="high",
                                                    explanation="Statement says the item was absent but images show it."))
            if n.category == Category.CLOTHING and n.attribute and seen_in:
                img_cols = set(colors.get(n.canonical, {}))
                if img_cols and n.attribute not in img_cols:
                    status = "attribute_mismatch"
                    contradictions.append(Contradiction(
                        slot=f"clothing:{n.canonical}:color",
                        values={f"narrative:{n.attribute}": [n.observation_id],
                                **{f"image:{c}": colors[n.canonical][c] for c in img_cols}},
                        sources=["narrative"] + seen_in, kind="narrative_vs_image",
                        severity="low" if cert == Certainty.HEDGED else "high",
                        explanation="Reported colour differs from colours observed in the images."))
            narrative_support.append({"observation_id": n.observation_id, "category": key[0], "label": key[1],
                                      "attribute": n.attribute, "certainty": cert.value if cert else None,
                                      "status": status, "images": seen_in})
        del img_labels

    present_cats = {k[0] for k in support}
    missing = [c.value for c in (Category.CLOTHING, Category.ACCESSORY, Category.CARRIED, Category.SCENE)
               if c.value not in present_cats]
    return AggregatedEvidence(group_id=group_id, image_ids=image_ids, consolidated=consolidated,
                              contradictions=contradictions, duplicates=duplicates, missing_information=missing,
                              narrative_support=narrative_support)


def kept_label_sets(agg: AggregatedEvidence) -> dict[str, set]:
    out: dict[str, set] = defaultdict(set)
    for d in agg.consolidated:
        if d["kept"]:
            out[d["category"]].add(d["label"])
    return out
