# LLM-powered Column Similarity

This Streamlit app compares the text in two Excel columns using embeddings from a model discovered via a Swagger/OpenAPI endpoint. You can declare which uploaded column holds **policy statements** and which holds **control descriptions**, then the app finds the best-matching control for every policy.

## Requirements
- Python 3.9+
- Excel support via `pandas`, `openpyxl`, `xlrd`
- Network access to a Swagger/OpenAPI docs URL that exposes LLM/embedding endpoints

Install dependencies locally with:
```bash
pip install -r requirements.txt
```

## Running the app
```bash
streamlit run app.py
```
Then open the provided local URL in your browser.

## How to use it
1. **Upload two Excel files** (`.xlsx`, `.xlsm`, `.xls`) from the sidebar. For each file, choose a sheet and a column; a small preview appears for sanity-checking.
2. **Discover an LLM endpoint** by pasting a Swagger/OpenAPI docs URL (e.g., `https://your-llm-host/docs`). The app tries `/openapi.json` fallbacks automatically and disables SSL verification for these discovery calls.
3. **Select a model** by choosing a discovered `GET` endpoint containing "model"/"llm", clicking **Fetch LLMs**, and picking one from the dropdown. Provide an API key if required; it is sent as `Authorization: Bearer ...` on all requests.
4. **Pick the comparison direction** using the radio control:
   - *File 1 as policy statements ➜ best control in File 2*
   - *File 2 as policy statements ➜ best control in File 1*
5. Click **Generate similarity scores**. For each policy, the app:
   - Calls `/embeddings` with `{model, input}` to embed every policy and control row (blank rows are skipped with an error note).
   - Computes cosine similarity between each policy vector and all control vectors.
   - Chooses the single highest-scoring control for each policy statement.
6. **Review and export** the results table (Policy Statement, Best Matching Control, Similarity) and download an Excel file of the matches.

## Notes and limitations
- SSL certificate verification is disabled (`verify=False`) for all HTTP calls to simplify connecting to self-signed endpoints—use only with trusted hosts.
- Data you upload is sent to the selected LLM host for embedding; avoid uploading sensitive content unless you trust the endpoint.
- If no embeddings are returned for a row, the app records an error message and continues with available data.
