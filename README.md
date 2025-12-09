# LLM-powered Column Similarity

This Streamlit app compares two Excel column selections using embeddings from a selectable LLM endpoint.

## Setup
1. Create a virtual environment (optional) and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

## Features
- Upload two Excel files (.xlsx, .xlsm, .xls) and choose sheets/columns.
- Discover LLM endpoints from a Swagger/OpenAPI URL, pick a model, and provide an API key.
- Generate embeddings with SSL verification disabled, compute cosine similarity, and download results as Excel.
