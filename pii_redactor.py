"""PII detection and redaction helpers.

This module keeps the model-loading code separate from the pure redaction logic.
That makes the app easier to test because unit tests can validate span replacement
without downloading a Hugging Face model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Optional


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


# ---------------------------------------------------------------------------
# CSV redaction helpers
# ---------------------------------------------------------------------------

# Maps PII category keys to a redaction label, column-name keywords, and an
# optional compiled regex that validates whether a cell value looks like that
# PII type before blindly redacting it.  The regex acts as a safety check only
# — if None, any non-empty cell in a matched column is redacted.
COLUMN_PII_PROFILES: dict[str, dict[str, Any]] = {
    "private_person": {
        "label": "[REDACTED_PRIVATE_PERSON]",
        "keywords": [
            "first_name", "last_name", "name", "full_name", "firstname",
            "lastname", "fname", "lname", "given_name", "surname", "fullname",
        ],
        "regex": None,
    },
    "private_email": {
        "label": "[REDACTED_PRIVATE_EMAIL]",
        "keywords": ["email", "email_address", "e_mail", "emailaddress", "mail"],
        "regex": re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
    },
    "private_phone": {
        "label": "[REDACTED_PRIVATE_PHONE]",
        "keywords": [
            "phone", "phone_number", "telephone", "tel", "mobile",
            "cell", "fax", "phonenumber", "cellphone",
        ],
        "regex": re.compile(r"^[\d\s\-\(\)\+\.]{7,20}$"),
    },
    "account_number": {
        "label": "[REDACTED_ACCOUNT_NUMBER]",
        "keywords": [
            "ssn", "social_security", "sin", "national_id", "tax_id",
            "account_number", "acct", "account",
        ],
        "regex": re.compile(r"^\d[\d\-\s]{5,}$"),
    },
    "private_date": {
        "label": "[REDACTED_PRIVATE_DATE]",
        "keywords": [
            "dob", "date_of_birth", "birth_date", "birthdate", "birthday",
            "date_of_birth",
        ],
        "regex": re.compile(
            r"^\d{4}[-/]\d{2}[-/]\d{2}$"   # YYYY-MM-DD
            r"|^\d{2}[-/]\d{2}[-/]\d{4}$"  # MM-DD-YYYY
        ),
    },
    "private_address": {
        "label": "[REDACTED_PRIVATE_ADDRESS]",
        "keywords": [
            "address", "street_address", "street", "addr", "address_line",
            "city", "state", "zip", "zip_code", "zipcode", "postal",
            "postal_code",
        ],
        "regex": None,
    },
}


def classify_column(column_name: str) -> Optional[str]:
    """Return the PII category for a column name, or None if unrecognised.

    Normalises the column name to lowercase with underscores, then checks for
    substring matches against the keyword list for each PII category.
    """
    normalised = column_name.lower().replace(" ", "_").replace("-", "_").strip("_")
    for category, profile in COLUMN_PII_PROFILES.items():
        for keyword in profile["keywords"]:
            if keyword == normalised or keyword in normalised:
                return category
    return None


def redact_cell_by_category(value: str, category: str) -> tuple[str, bool]:
    """Redact a cell value for a known PII category.

    If the category has a validation regex, the value must match before it is
    redacted.  Returns ``(redacted_label, True)`` or ``(original_value, False)``.
    """
    if not value or not value.strip():
        return value, False

    profile = COLUMN_PII_PROFILES[category]
    regex: Optional[re.Pattern[str]] = profile["regex"]

    if regex is not None and not regex.match(value.strip()):
        return value, False

    return profile["label"], True


def redact_row_with_model(
    row_values: dict[str, str],
    threshold: float = 0.5,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[dict[str, str], int]:
    """Redact PII in a set of unclassified cells using column-labelled context.

    Joins cells as ``"Col_Name: value, Col_Name: value, ..."`` so the model has
    natural-language context (the column name acts as a label).  Detected entity
    spans are mapped back to their originating cells by character offset and the
    whole cell is replaced with the entity's redaction label.
    """
    items = [(col, val) for col, val in row_values.items() if val.strip()]
    if not items:
        return row_values, 0

    separator = ", "
    parts = [f"{col}: {val}" for col, val in items]
    joined = separator.join(parts)

    # Track the start/end offset of each *value* (not the key prefix) inside joined.
    value_offsets: dict[str, tuple[int, int]] = {}
    pos = 0
    for i, (col, val) in enumerate(items):
        key_len = len(f"{col}: ")
        v_start = pos + key_len
        v_end = v_start + len(val)
        value_offsets[col] = (v_start, v_end)
        pos += key_len + len(val)
        if i < len(items) - 1:
            pos += len(separator)

    entities = detect_entities(joined, threshold=threshold, model_name=model_name)

    result = dict(row_values)
    for entity in entities:
        for col, (v_start, v_end) in value_offsets.items():
            if entity.start < v_end and entity.end > v_start:
                result[col] = entity.redaction_label
                break

    return result, len(entities)


def redact_csv(
    filepath: str,
    threshold: float = 0.5,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple["pd.DataFrame", dict[str, Any]]:
    """Redact PII from all string columns in a CSV file.

    Uses a three-phase hybrid strategy:

    1. Classify each string column by name against ``COLUMN_PII_PROFILES``.
    2. Redact classified columns directly (whole-cell replacement, with optional
       regex validation) — no model inference needed.
    3. For any unclassified string columns, reconstruct each row as
       ``"Col_Name: value, Col_Name: value, ..."`` and run the model once per
       row so the column labels supply the linguistic context the model needs.

    Returns a tuple of the redacted DataFrame and a stats dict with keys
    ``columns_processed`` (list of column names) and ``total_entities`` (int).
    """
    import pandas as pd

    df = pd.read_csv(filepath)
    total_entities = 0
    string_cols = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]

    # Phase 1: classify columns by name.
    col_categories = {col: classify_column(col) for col in string_cols}
    classified_cols = [c for c in string_cols if col_categories[c] is not None]
    unclassified_cols = [c for c in string_cols if col_categories[c] is None]

    # Phase 2: redact classified columns without touching the model.
    for col in classified_cols:
        category = col_categories[col]
        for idx in df.index:
            val = df.at[idx, col]
            if isinstance(val, str):
                redacted_val, was_redacted = redact_cell_by_category(val, category)
                if was_redacted:
                    df.at[idx, col] = redacted_val
                    total_entities += 1

    # Phase 3: model fallback for unclassified string columns.
    if unclassified_cols:
        for idx, row in df.iterrows():
            row_values = {
                col: row[col]
                for col in unclassified_cols
                if isinstance(row[col], str) and row[col].strip()
            }
            if row_values:
                updated, count = redact_row_with_model(row_values, threshold, model_name)
                for col, val in updated.items():
                    df.at[idx, col] = val
                total_entities += count

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
