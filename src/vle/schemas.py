"""Pydantic data model for evidence records.

Design rules enforced here:
* Every observation carries full provenance (source, image id, model and
  prompt version, timestamp, evidence reference) and a human review status.
* Witness-reported observations (``source_type="narrative"``) and
  model-observed image content (``source_type="vlm"``) are distinct types and
  are never merged into one record.
* Confidence fields are *scores*, not probabilities, unless the field name says
  ``calibrated``; the calibration status is recorded explicitly.
* There are no fields for identity, names, faces, demographics or biometrics.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "evidence-schema-v1"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Category(str, Enum):
    OBJECT = "object"
    CLOTHING = "clothing"
    ACCESSORY = "accessory"
    CARRIED = "carried_object"
    SCENE = "scene"
    ENVIRONMENT = "environment"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class Certainty(str, Enum):
    """Linguistic certainty of a narrative statement (never upgraded)."""
    DEFINITE = "definite"
    HEDGED = "hedged"
    NEGATED = "negated"


# ---------------------------------------------------------------------------
# Raw VLM structured output (what the model is asked to emit).
# ---------------------------------------------------------------------------
class VLMItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = Field(min_length=1, max_length=80)
    color: str | None = None
    count: int | None = Field(default=None, ge=0, le=500)
    confidence: Literal["high", "medium", "low"] = "medium"

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: object) -> str:
        s = str(v or "medium").lower().strip()
        return s if s in {"high", "medium", "low"} else "medium"

    @field_validator("count", mode="before")
    @classmethod
    def _count(cls, v: object) -> int | None:
        try:
            return None if v in (None, "", "null") else int(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None


class VLMScene(BaseModel):
    model_config = ConfigDict(extra="ignore")
    setting: Literal["indoor", "outdoor", "unclear"] = "unclear"
    place: str | None = None
    lighting: str | None = None
    weather: str | None = None

    @field_validator("setting", mode="before")
    @classmethod
    def _setting(cls, v: object) -> str:
        s = str(v or "unclear").lower().strip()
        return s if s in {"indoor", "outdoor", "unclear"} else "unclear"


class VLMQuality(BaseModel):
    model_config = ConfigDict(extra="ignore")
    blur: Literal["none", "mild", "severe"] = "none"
    occlusion: Literal["none", "partial", "heavy"] = "none"
    lighting: Literal["good", "dim", "very dark"] = "good"

    @field_validator("blur", "occlusion", "lighting", mode="before")
    @classmethod
    def _lower(cls, v: object, info) -> str:  # type: ignore[no-untyped-def]
        allowed = {"blur": {"none", "mild", "severe"}, "occlusion": {"none", "partial", "heavy"},
                   "lighting": {"good", "dim", "very dark"}}[info.field_name]
        s = str(v or "").lower().strip()
        default = {"blur": "none", "occlusion": "none", "lighting": "good"}[info.field_name]
        return s if s in allowed else default


class VLMStructuredOutput(BaseModel):
    """Schema the VLM must produce for a single image (prompt struct-v1)."""
    model_config = ConfigDict(extra="ignore")
    objects: list[VLMItem] = Field(default_factory=list, max_length=60)
    clothing: list[VLMItem] = Field(default_factory=list, max_length=40)
    accessories: list[VLMItem] = Field(default_factory=list, max_length=30)
    carried_objects: list[VLMItem] = Field(default_factory=list, max_length=30)
    scene: VLMScene = Field(default_factory=VLMScene)
    image_quality: VLMQuality = Field(default_factory=VLMQuality)
    unclear: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("objects", "clothing", "accessories", "carried_objects", mode="before")
    @classmethod
    def _coerce_items(cls, v: object) -> list:
        if v is None:
            return []
        if isinstance(v, (str, dict)):
            v = [v]
        out = []
        for it in v:  # type: ignore[union-attr]
            if isinstance(it, str):
                if it.strip():
                    out.append({"name": it})
            elif isinstance(it, dict):
                if "name" not in it:
                    for k in ("item", "object", "label", "type"):
                        if k in it:
                            it = {**it, "name": it[k]}
                            break
                if it.get("name"):
                    out.append(it)
        return out

    @field_validator("unclear", mode="before")
    @classmethod
    def _unclear(cls, v: object) -> list:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v if x]  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Evidence records (what the framework stores)
# ---------------------------------------------------------------------------
class Provenance(BaseModel):
    source_type: Literal["vlm", "narrative", "human"]
    source_id: str                       # image id or narrative id or reviewer id
    image_id: str | None = None
    model_name: str | None = None
    model_revision: str | None = None
    quantization: str | None = None
    prompt_version: str | None = None
    run_id: str | None = None
    timestamp: str = Field(default_factory=utcnow)
    evidence_ref: str | None = None       # image path / region / narrative char span


class UncertaintyInfo(BaseModel):
    verbal_confidence: Literal["high", "medium", "low"] | None = None
    token_confidence: float | None = Field(default=None, ge=0, le=1)  # mean token prob of span
    grounding_score: float | None = None  # CLIP image-text similarity (raw cosine)
    combined_score: float | None = Field(default=None, ge=0, le=1)
    calibrated: bool = False              # True only when produced by a calibrator fit on dev data
    image_quality_flags: list[str] = Field(default_factory=list)
    linguistic_certainty: Certainty | None = None  # narrative only
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)


class Observation(BaseModel):
    observation_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    category: Category
    label: str                            # canonical label if mapped, else normalised text
    raw_text: str                         # exactly what the model / witness said
    attribute: str | None = None          # e.g. colour
    canonical: str | None = None          # lexicon canonical class (None = unmapped)
    provenance: Provenance
    uncertainty: UncertaintyInfo = Field(default_factory=UncertaintyInfo)
    review_status: ReviewStatus = ReviewStatus.PENDING
    schema_version: str = SCHEMA_VERSION


class ImageQuality(BaseModel):
    width: int
    height: int
    blur_laplacian_var: float
    brightness_mean: float
    contrast_std: float
    jpeg_quality_est: float | None = None
    flags: list[str] = Field(default_factory=list)


class Contradiction(BaseModel):
    slot: str                              # e.g. "clothing:shirt:color"
    values: dict[str, list[str]]           # value -> list of observation ids
    sources: list[str]
    kind: Literal["intra_source", "cross_image", "narrative_vs_image"]
    severity: Literal["low", "high"] = "low"
    explanation: str = ""


class AggregatedEvidence(BaseModel):
    """Consolidated view over a *set of images* — never an identity claim."""
    group_id: str
    image_ids: list[str]
    consolidated: list[dict]               # label/category/support/sources/score
    contradictions: list[Contradiction] = Field(default_factory=list)
    duplicates: list[list[str]] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    narrative_support: list[dict] = Field(default_factory=list)
    disclaimer: str = ("Consolidation summarises observations across the listed images. It does not "
                       "assert that different images depict the same person or object.")


class CorrectionEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    observation_id: str
    reviewer_id: str
    action: Literal["accept", "reject", "correct", "add"]
    before: dict | None = None
    after: dict | None = None
    comment: str = ""
    timestamp: str = Field(default_factory=utcnow)
    duration_ms: int | None = None
