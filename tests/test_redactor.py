from pii_redactor import (
    EntitySpan,
    deduplicate_overlapping_spans,
    normalize_pipeline_output,
    redact_with_spans,
)


def test_redact_with_single_span():
    text = "Email Alice at alice@example.com."
    entities = [
        EntitySpan(
            entity_group="private_email",
            word="alice@example.com",
            start=15,
            end=32,
            score=0.99,
        )
    ]

    result = redact_with_spans(text, entities)

    assert result == "Email Alice at [REDACTED_PRIVATE_EMAIL]."


def test_redact_with_multiple_spans_keeps_indexes_correct():
    text = "Alice can be reached at alice@example.com."
    entities = [
        EntitySpan(
            entity_group="private_person",
            word="Alice",
            start=0,
            end=5,
            score=0.95,
        ),
        EntitySpan(
            entity_group="private_email",
            word="alice@example.com",
            start=24,
            end=41,
            score=0.99,
        ),
    ]

    result = redact_with_spans(text, entities)

    assert result == "[REDACTED_PRIVATE_PERSON] can be reached at [REDACTED_PRIVATE_EMAIL]."


def test_deduplicate_overlapping_spans_keeps_higher_score():
    entities = [
        EntitySpan(
            entity_group="private_person",
            word="Alice Johnson",
            start=0,
            end=13,
            score=0.75,
        ),
        EntitySpan(
            entity_group="private_person",
            word="Alice",
            start=0,
            end=5,
            score=0.95,
        ),
    ]

    deduped = deduplicate_overlapping_spans(entities)

    assert len(deduped) == 1
    assert deduped[0].word == "Alice"
    assert deduped[0].score == 0.95


def test_normalize_pipeline_output_skips_malformed_rows():
    raw = [
        {
            "entity_group": "private_email",
            "word": "alice@example.com",
            "start": 10,
            "end": 27,
            "score": 0.99,
        },
        {"entity_group": "bad_row"},
    ]

    entities = normalize_pipeline_output(raw)

    assert len(entities) == 1
    assert entities[0].entity_group == "private_email"
