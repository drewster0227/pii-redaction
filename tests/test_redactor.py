import os
from unittest.mock import patch

from pii_redactor import (
    EntitySpan,
    classify_column,
    deduplicate_overlapping_spans,
    normalize_pipeline_output,
    redact_cell_by_category,
    redact_csv,
    redact_row_with_model,
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


# ---------------------------------------------------------------------------
# classify_column
# ---------------------------------------------------------------------------

def test_classify_column_matches_email_variants():
    assert classify_column("Email_Address") == "private_email"
    assert classify_column("email") == "private_email"
    assert classify_column("EMAIL_ADDRESS") == "private_email"


def test_classify_column_matches_phone_variants():
    assert classify_column("Phone_Number") == "private_phone"
    assert classify_column("telephone") == "private_phone"
    assert classify_column("mobile") == "private_phone"


def test_classify_column_matches_name_variants():
    assert classify_column("First_Name") == "private_person"
    assert classify_column("Last_Name") == "private_person"
    assert classify_column("full_name") == "private_person"


def test_classify_column_matches_ssn():
    assert classify_column("SSN") == "account_number"
    assert classify_column("social_security") == "account_number"


def test_classify_column_matches_dob():
    assert classify_column("Date_of_Birth") == "private_date"
    assert classify_column("dob") == "private_date"


def test_classify_column_matches_address_variants():
    assert classify_column("Street_Address") == "private_address"
    assert classify_column("city") == "private_address"
    assert classify_column("zip_code") == "private_address"


def test_classify_column_returns_none_for_generic_columns():
    assert classify_column("ID") is None
    assert classify_column("Status") is None
    assert classify_column("Notes") is None
    assert classify_column("Score") is None


# ---------------------------------------------------------------------------
# redact_cell_by_category
# ---------------------------------------------------------------------------

def test_redact_cell_phone_valid():
    result, was_redacted = redact_cell_by_category("555-123-4567", "private_phone")
    assert was_redacted is True
    assert result == "[REDACTED_PRIVATE_PHONE]"


def test_redact_cell_phone_invalid_value_not_redacted():
    # A non-phone-like value in a phone column should not be redacted
    result, was_redacted = redact_cell_by_category("not-a-phone!", "private_phone")
    assert was_redacted is False
    assert result == "not-a-phone!"


def test_redact_cell_name_always_redacted():
    # Names have no regex, so any non-empty value is redacted
    result, was_redacted = redact_cell_by_category("Mary", "private_person")
    assert was_redacted is True
    assert result == "[REDACTED_PRIVATE_PERSON]"


def test_redact_cell_email_valid():
    result, was_redacted = redact_cell_by_category("alice@example.com", "private_email")
    assert was_redacted is True
    assert result == "[REDACTED_PRIVATE_EMAIL]"


def test_redact_cell_ssn_valid():
    result, was_redacted = redact_cell_by_category("202-60-2536", "account_number")
    assert was_redacted is True
    assert result == "[REDACTED_ACCOUNT_NUMBER]"


# ---------------------------------------------------------------------------
# redact_row_with_model
# ---------------------------------------------------------------------------

def test_redact_row_with_model_maps_entity_to_correct_cell():
    # Simulate the model detecting "Mary" in "Notes: Mary, Description: software engineer"
    fake_entity = EntitySpan(
        entity_group="private_person",
        word="Mary",
        start=7,   # "Notes: " is 7 chars, so "Mary" starts at 7
        end=11,
        score=0.95,
    )
    with patch("pii_redactor.detect_entities", return_value=[fake_entity]):
        result, count = redact_row_with_model(
            {"Notes": "Mary", "Description": "software engineer"}
        )
    assert result["Notes"] == "[REDACTED_PRIVATE_PERSON]"
    assert result["Description"] == "software engineer"
    assert count == 1


# ---------------------------------------------------------------------------
# redact_csv (end-to-end with mocked model)
# ---------------------------------------------------------------------------

def test_redact_csv_end_to_end():
    csv_path = os.path.join(os.path.dirname(__file__), "fake_pii_test_data.csv")
    # The model should never be called — all columns are classifiable by name.
    with patch("pii_redactor.detect_entities", return_value=[]) as mock_model:
        df, stats = redact_csv(csv_path, threshold=0.5)

    mock_model.assert_not_called()

    assert df.at[0, "First_Name"] == "[REDACTED_PRIVATE_PERSON]"
    assert df.at[0, "Last_Name"] == "[REDACTED_PRIVATE_PERSON]"
    assert df.at[0, "Email_Address"] == "[REDACTED_PRIVATE_EMAIL]"
    assert df.at[0, "Phone_Number"] == "[REDACTED_PRIVATE_PHONE]"
    assert df.at[0, "SSN"] == "[REDACTED_ACCOUNT_NUMBER]"
    assert df.at[0, "Date_of_Birth"] == "[REDACTED_PRIVATE_DATE]"
    assert df.at[0, "Street_Address"] == "[REDACTED_PRIVATE_ADDRESS]"
    # Numeric ID column must be untouched
    assert df.at[0, "ID"] == 1
    assert stats["total_entities"] > 0


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
