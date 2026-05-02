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


def redact_csv(
    filepath: str,
    threshold: float = 0.5,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple["pd.DataFrame", dict[str, Any]]:
    """Redact PII from all string columns in a CSV file.

    Each row's string values are joined into a single sentence before being
    passed to the model so that it has enough context to recognise PII (e.g. a
    lone first name cell gives the model no signal, but "Mary. Miller. …" does).
    Detected entity spans are then mapped back to individual cells using
    character offsets before the cells are updated in place.

    Returns a tuple of the redacted DataFrame and a stats dict with keys
    ``columns_processed`` (list of column names) and ``total_entities`` (int).
    """
    import pandas as pd

    df = pd.read_csv(filepath)
    total_entities = 0
    string_cols = [col for col in df.columns if df[col].dtype == object]

    _SEPARATOR = ". "

    for idx, row in df.iterrows():
        # Collect the string values present in this row and their column names.
        col_order: list[str] = []
        str_values: list[str] = []
        for col in string_cols:
            val = row[col]
            if isinstance(val, str):
                col_order.append(col)
                str_values.append(val)

        if not str_values:
            continue

        # Build the joined text and record where each cell starts and ends.
        offsets: list[tuple[int, int]] = []
        pos = 0
        for val in str_values:
            offsets.append((pos, pos + len(val)))
            pos += len(val) + len(_SEPARATOR)
        joined = _SEPARATOR.join(str_values)

        # Run the model once on the full row text for proper context.
        entities = detect_entities(joined, threshold=threshold, model_name=model_name)
        total_entities += len(entities)

        # Map each entity back to whichever cell it falls inside, adjusting
        # the span offsets to be relative to the start of that cell.
        for col, cell_val, (cell_start, cell_end) in zip(col_order, str_values, offsets):
            cell_entities = [
                EntitySpan(
                    entity_group=e.entity_group,
                    word=e.word,
                    start=e.start - cell_start,
                    end=e.end - cell_start,
                    score=e.score,
                )
                for e in entities
                if e.start >= cell_start and e.end <= cell_end
            ]
            if cell_entities:
                df.at[idx, col] = redact_with_spans(cell_val, cell_entities)

    return df, {"columns_processed": string_cols, "total_entities": total_entities}


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
