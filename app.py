"""Gradio app for local PII redaction."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import gradio as gr

from pii_redactor import DEFAULT_MODEL_NAME, entities_to_rows, redact_csv, redact_text


EXAMPLE_TEXT = """My name is Alice Johnson. I live at 123 Main Street, Detroit, MI 48201.
You can email me at alice.johnson@example.com or call 555-123-4567.
My backup contact is Bob Smith at bob.smith@example.org.
"""


def redact_for_ui(text: str, threshold: float):
    """Run redaction and return values for Gradio outputs."""
    if not text or not text.strip():
        empty_table = pd.DataFrame(columns=["entity_group", "text", "start", "end", "score"])
        return "", empty_table, "No text provided. Paste text into the input box and try again."

    redacted, entities = redact_text(text=text, threshold=threshold)
    rows = entities_to_rows(entities)
    table = pd.DataFrame(rows, columns=["entity_group", "text", "start", "end", "score"])

    summary = f"Detected {len(entities)} PII span(s) using {DEFAULT_MODEL_NAME}."
    return redacted, table, summary


def redact_csv_for_ui(file, threshold: float):
    """Redact PII from an uploaded CSV and return a downloadable file path and summary."""
    if file is None:
        return None, "No file uploaded. Please upload a CSV file and try again."

    redacted_df, stats = redact_csv(file.name, threshold=threshold)

    suffix = "_redacted" + os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", newline="", encoding="utf-8") as tmp:
        redacted_df.to_csv(tmp, index=False)
        out_path = tmp.name

    cols = ", ".join(stats["columns_processed"]) if stats["columns_processed"] else "none"
    summary = (
        f"Redacted {stats['total_entities']} PII span(s) across "
        f"{len(stats['columns_processed'])} string column(s) ({cols}) "
        f"using {DEFAULT_MODEL_NAME}."
    )
    return out_path, summary


def build_app() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(title="PII Redaction App") as demo:
        gr.Markdown(
            """
            # PII Redaction App

            Detect and redact personally identifiable information using a local Hugging Face model.
            """
        )

        with gr.Tabs():
            with gr.Tab("Text"):
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(
                            label="Original text",
                            value=EXAMPLE_TEXT,
                            lines=12,
                            placeholder="Paste text containing possible PII here...",
                        )
                        text_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Confidence threshold",
                        )
                        redact_button = gr.Button("Redact PII", variant="primary")

                    with gr.Column():
                        redacted_text = gr.Textbox(
                            label="Redacted text",
                            lines=12,
                        )
                        text_summary = gr.Markdown()

                entity_table = gr.Dataframe(
                    label="Detected entities",
                    headers=["entity_group", "text", "start", "end", "score"],
                    interactive=False,
                )

                redact_button.click(
                    fn=redact_for_ui,
                    inputs=[input_text, text_threshold],
                    outputs=[redacted_text, entity_table, text_summary],
                )

            with gr.Tab("CSV"):
                with gr.Row():
                    with gr.Column():
                        csv_upload = gr.File(
                            label="Upload CSV",
                            file_types=[".csv"],
                        )
                        csv_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Confidence threshold",
                        )
                        csv_button = gr.Button("Redact CSV", variant="primary")

                    with gr.Column():
                        csv_output = gr.File(label="Download redacted CSV")
                        csv_summary = gr.Markdown()

                csv_button.click(
                    fn=redact_csv_for_ui,
                    inputs=[csv_upload, csv_threshold],
                    outputs=[csv_output, csv_summary],
                )

        gr.Markdown(
            """
            ## Reminder

            This tool helps reduce exposure of sensitive text, but it is not a perfect anonymization or compliance solution.
            Review the output before sharing or storing redacted content.
            """
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
