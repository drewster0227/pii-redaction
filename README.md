# PII Redaction App

A local Hugging Face app that detects and redacts personally identifiable information (PII) from text using [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter).

This is built as a practical starter project for learning how to run Hugging Face models locally with `transformers` while creating something useful for privacy-aware text workflows.

## What it does

- Loads the `openai/privacy-filter` token-classification model
- Detects likely PII spans in pasted text
- Redacts sensitive spans with readable placeholders
- Shows detected entities in a review table
- Lets you adjust the confidence threshold
- Runs locally through a Gradio web app

## Project structure

```text
.
├── app.py                  # Gradio web interface
├── pii_redactor.py          # Model wrapper + redaction logic
├── requirements.txt         # Python dependencies
├── sample_text.txt          # Demo text for testing
├── tests/
│   └── test_redactor.py     # Unit tests for redaction logic
└── README.md
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/drewster0227/pii-redaction.git
cd pii-redaction
```

### 2. Create a virtual environment

Windows PowerShell:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the app

```bash
python app.py
```

Gradio will print a local URL, usually:

```text
http://127.0.0.1:7860
```

Open that in your browser.

## Run tests

```bash
pytest
```

The tests avoid downloading the Hugging Face model. They focus on the pure redaction/index logic.

## Example

Input:

```text
My name is Alice Johnson. Email me at alice.johnson@example.com or call 555-123-4567.
```

Possible output:

```text
My name is [REDACTED_PRIVATE_PERSON]. Email me at [REDACTED_PRIVATE_EMAIL] or call [REDACTED_PRIVATE_PHONE_NUMBER].
```

## Important note

This app is a privacy helper, not a legal/compliance guarantee. Always review redacted output before using it in sensitive workflows.

## Next features

- File upload support for `.txt` and `.csv`
- Batch redaction for folders
- Export redacted text
- Export detected entities as CSV
- Regex fallback for emails and phone numbers
- Dockerfile
- Hugging Face Space deployment
