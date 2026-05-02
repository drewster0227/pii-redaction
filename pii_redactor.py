"""PII detection and redaction helpers.

This module keeps the model-loading code separate from the pure redaction logic.
That makes the app easier to test because unit tests can validate span replacement
without downloading a Hugging Face model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable


DEFAULT_MODEL_NAME = "openai/privacy-filter"


@dataclass(frozen=True)
class EntitySpan:
    """A detected PII span returned by the model pipeline."""

    entity_group: str
    word: str
    start: int
    end: int
    score: float

    @property
    def redaction_label(self) -> str:
        """Return a readable replacement label for this entity."""
        clean_label = self.entity_group.upper().replace("-", "_").replace(" ", "_")
        return f"[REDACTED_{clean_label}]"


def normalize_pipeline_output(raw_entities: Iterable[dict[str, Any]]) -> list[EntitySpan]:
    """Convert raw Hugging Face pipeline dictionaries into EntitySpan objects.

    The token-classification pipeline returns dictionaries with fields such as
    entity_group, word, start, end, and score when aggregation_strategy="simple".
    This function validates the key fields and skips malformed rows.
    """
    normalized: list[EntitySpan] = []

    for item in raw_entities:
        try:
            entity_group = str(item["entity_group"])
            word = str(item.get("word", ""))
            start = int(item["start"])
            end = int(item["end"])
            score = float(item.get("score", 0.0))
        except (KeyError, TypeError, ValueError):
            continue

        if start < 0 or end <= start:
            continue

        normalized.append(
            EntitySpan(
                entity_group=entity_group,
                word=word,
                start=start,
                end=end,
                score=score,
            )
        )

    return normalized


def filter_entities(entities: Iterable[EntitySpan], threshold: float) -> list[EntitySpan]:
    """Keep entities at or above the requested confidence threshold."""
    return [entity for entity in entities if entity.score >= threshold]


def deduplicate_overlapping_spans(entities: Iterable[EntitySpan]) -> list[EntitySpan]:
    """Remove overlapping spans, keeping the highest-confidence span.

    Token-classification models can occasionally return overlapping entities.
    Redacting overlapping character ranges can corrupt indexes, so we keep the
    best-scoring candidate when spans overlap.
    """
    sorted_entities = sorted(
        entities,
        key=lambda entity: (entity.start, -(entity.end - entity.start), -entity.score),
    )

    kept: list[EntitySpan] = []

    for entity in sorted_entities:
        overlaps = [
            existing
            for existing in kept
            if not (entity.end <= existing.start or entity.start >= existing.end)
        ]

        if not overlaps:
            kept.append(entity)
            continue

        best_existing = max(overlaps, key=lambda existing: existing.score)
        if entity.score > best_existing.score:
            kept = [
                existing
                for existing in kept
                if entity.end <= existing.start or entity.start >= existing.end
            ]
            kept.append(entity)

    return sorted(kept, key=lambda entity: entity.start)


def redact_with_spans(text: str, entities: Iterable[EntitySpan]) -> str:
    """Redact text using character spans.

    Spans are applied from the end of the string backward so earlier replacements
    do not shift the indexes of later replacements.
    """
    redacted = text
    safe_entities = deduplicate_overlapping_spans(entities)

    for entity in sorted(safe_entities, key=lambda item: item.start, reverse=True):
        if entity.start > len(redacted) or entity.end > len(redacted):
            continue

        redacted = redacted[: entity.start] + entity.redaction_label + redacted[entity.end :]

    return redacted


@lru_cache(maxsize=1)
def load_pii_pipeline(model_name: str = DEFAULT_MODEL_NAME):
    """Load and cache the Hugging Face token-classification pipeline."""
    from transformers import pipeline

    return pipeline(
        task="token-classification",
        model=model_name,
        aggregation_strategy="simple",
    )


def detect_entities(text: str, threshold: float = 0.5, model_name: str = DEFAULT_MODEL_NAME) -> list[EntitySpan]:
    """Detect PII entities in text using the Hugging Face model."""
    if not text.strip():
        return []

    pii_pipeline = load_pii_pipeline(model_name)
    raw_entities = pii_pipeline(text)
    entities = normalize_pipeline_output(raw_entities)
    filtered = filter_entities(entities, threshold)
    return deduplicate_overlapping_spans(filtered)


def redact_text(text: str, threshold: float = 0.5, model_name: str = DEFAULT_MODEL_NAME) -> tuple[str, list[EntitySpan]]:
    """Detect and redact PII from text.

    Returns a tuple containing the redacted text and the detected entity spans.
    """
    entities = detect_entities(text=text, threshold=threshold, model_name=model_name)
    return redact_with_spans(text, entities), entities


def entities_to_rows(entities: Iterable[EntitySpan]) -> list[dict[str, Any]]:
    """Convert entity spans into rows for display in Gradio or pandas."""
    return [
        {
            "entity_group": entity.entity_group,
            "text": entity.word,
            "start": entity.start,
            "end": entity.end,
            "score": round(entity.score, 4),
        }
        for entity in entities
    ]
