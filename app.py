import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests import Response
from urllib.parse import urljoin, urlsplit

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

SUPPORTED_EXTENSIONS = {".xlsx", ".xlsm", ".xls"}


@dataclass
class ExcelSelection:
    file: st.runtime.uploaded_file_manager.UploadedFile
    excel_file: pd.ExcelFile
    sheet_name: str
    column: str


@st.cache_data(show_spinner=False)
def load_excel_file(upload: st.runtime.uploaded_file_manager.UploadedFile) -> pd.ExcelFile:
    ext = Path(upload.name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Please upload an Excel file.")

    engine = "openpyxl"
    if ext == ".xls":
        engine = "xlrd"

    return pd.ExcelFile(upload, engine=engine)


def read_sheet(excel_file: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    return excel_file.parse(sheet_name=sheet)


@st.cache_data(show_spinner=False)
def fetch_openapi(swagger_url: str) -> Optional[Dict]:
    candidates = []
    swagger_url = swagger_url.strip()
    if not swagger_url:
        return None

    candidates.append(swagger_url)
    if swagger_url.endswith("/docs"):
        candidates.append(swagger_url[:-5] + "/openapi.json")
    if not swagger_url.endswith(".json"):
        candidates.append(swagger_url.rstrip("/") + "/openapi.json")

    tried = set()
    for url in candidates:
        if url in tried:
            continue
        tried.add(url)
        try:
            resp = requests.get(url, verify=False, timeout=15)
            if resp.ok:
                return resp.json()
        except Exception:
            continue
    return None


def derive_base_url(swagger_url: str, spec: Optional[Dict]) -> str:
    if spec:
        servers = spec.get("servers")
        if servers:
            url = servers[0].get("url")
            if url:
                return url.rstrip("/")

    parsed = urlsplit(swagger_url)
    return f"{parsed.scheme}://{parsed.netloc}"


@dataclass
class ModelEndpoint:
    path: str
    method: str


@st.cache_data(show_spinner=False)
def discover_model_endpoints(spec: Dict) -> List[ModelEndpoint]:
    endpoints: List[ModelEndpoint] = []
    for path, ops in spec.get("paths", {}).items():
        for method, op in ops.items():
            if method.lower() != "get":
                continue
            lower = path.lower()
            if any(token in lower for token in ["llm", "model"]):
                endpoints.append(ModelEndpoint(path=path, method=method))
    return endpoints


def fetch_llms(base_url: str, endpoint: ModelEndpoint, api_key: str) -> List[str]:
    url = urljoin(base_url + "/", endpoint.path.lstrip("/"))
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.request(endpoint.method, url, headers=headers, verify=False, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list):
        return [str(item) for item in data]

    if isinstance(data, dict):
        for key in ["models", "llms", "items", "data"]:
            if key in data and isinstance(data[key], list):
                return [
                    str(item.get("id", item.get("name", item))) if isinstance(item, dict) else str(item)
                    for item in data[key]
                ]
    return []


def extract_embedding_from_response(response: Response) -> Optional[List[float]]:
    try:
        payload = response.json()
    except Exception:
        return None

    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list) and payload["data"]:
            first = payload["data"][0]
            if isinstance(first, dict) and "embedding" in first:
                return first["embedding"]
        if "embedding" in payload:
            return payload["embedding"]
    return None


def request_embedding(base_url: str, api_key: str, model: str, text: str) -> Optional[List[float]]:
    url = urljoin(base_url + "/", "embeddings")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"model": model, "input": text}
    resp = requests.post(url, headers=headers, json=payload, verify=False, timeout=60)
    if not resp.ok:
        return None
    return extract_embedding_from_response(resp)


def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    if a.size == 0 or b.size == 0:
        return 0.0
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def fetch_embeddings(
    values: List[str], base_url: str, api_key: str, model: str, label: str
) -> Tuple[List[Optional[List[float]]], List[str]]:
    embeddings: List[Optional[List[float]]] = []
    errors: List[str] = []
    for idx, value in enumerate(values, start=1):
        if not value:
            errors.append(f"{label} row {idx}: missing text to embed")
            embeddings.append(None)
            continue
        embedding = request_embedding(base_url, api_key, model, str(value))
        if not embedding:
            errors.append(f"{label} row {idx}: failed to fetch embedding")
        embeddings.append(embedding)
    return embeddings, errors


def build_best_match_dataframe(
    policies: List[str], controls: List[str], base_url: str, api_key: str, model: str
) -> Tuple[pd.DataFrame, List[str]]:
    policy_embeddings, errors = fetch_embeddings(policies, base_url, api_key, model, "Policy")
    control_embeddings, control_errors = fetch_embeddings(
        controls, base_url, api_key, model, "Control"
    )
    errors.extend(control_errors)

    valid_policy_indices = [i for i, emb in enumerate(policy_embeddings) if emb]
    valid_control_indices = [i for i, emb in enumerate(control_embeddings) if emb]

    if not valid_policy_indices or not valid_control_indices:
        if not valid_policy_indices:
            errors.append("No valid policy embeddings were generated.")
        if not valid_control_indices:
            errors.append("No valid control embeddings were generated.")
        return pd.DataFrame(), errors

    policy_matrix = np.array([policy_embeddings[i] for i in valid_policy_indices])
    control_matrix = np.array([control_embeddings[i] for i in valid_control_indices])

    policy_norms = np.linalg.norm(policy_matrix, axis=1, keepdims=True)
    control_norms = np.linalg.norm(control_matrix, axis=1, keepdims=True)
    denominator = policy_norms * control_norms.T
    similarity_matrix = np.divide(
        policy_matrix @ control_matrix.T,
        denominator,
        out=np.zeros_like(policy_matrix @ control_matrix.T),
        where=denominator != 0,
    )

    rows = []
    for row_idx, policy_idx in enumerate(valid_policy_indices):
        best_control_idx = int(np.argmax(similarity_matrix[row_idx]))
        control_idx = valid_control_indices[best_control_idx]
        score = float(similarity_matrix[row_idx, best_control_idx])
        rows.append(
            {
                "Policy Statement": policies[policy_idx],
                "Best Matching Control": controls[control_idx],
                "Similarity": score,
            }
        )

    return pd.DataFrame(rows), errors


def make_download(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="similarity")
    buffer.seek(0)
    return buffer.read()


st.set_page_config(page_title="LLM Column Matcher", layout="wide")
st.title("LLM-powered Column Similarity")

st.sidebar.header("File uploads")
file1 = st.sidebar.file_uploader("Upload File 1", type=list(SUPPORTED_EXTENSIONS), key="file1")
file2 = st.sidebar.file_uploader("Upload File 2", type=list(SUPPORTED_EXTENSIONS), key="file2")

selection_a: Optional[ExcelSelection] = None
selection_b: Optional[ExcelSelection] = None

col1, col2 = st.columns(2)

with col1:
    st.subheader("File 1")
    if file1:
        try:
            excel1 = load_excel_file(file1)
            sheet1 = st.selectbox("Sheet", excel1.sheet_names, key="sheet1")
            df1 = read_sheet(excel1, sheet1)
            column1 = st.selectbox("Column", df1.columns, key="col1")
            st.dataframe(df1[[column1]].head())
            selection_a = ExcelSelection(file1, excel1, sheet1, column1)
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Unable to load File 1: {exc}")

with col2:
    st.subheader("File 2")
    if file2:
        try:
            excel2 = load_excel_file(file2)
            sheet2 = st.selectbox("Sheet", excel2.sheet_names, key="sheet2")
            df2 = read_sheet(excel2, sheet2)
            column2 = st.selectbox("Column", df2.columns, key="col2")
            st.dataframe(df2[[column2]].head())
            selection_b = ExcelSelection(file2, excel2, sheet2, column2)
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Unable to load File 2: {exc}")

st.markdown("---")
st.header("LLM Discovery")

swagger_url = st.text_input("Swagger/OpenAPI docs URL", placeholder="https://example.com/docs")
api_key = st.text_input("API key", type="password")

openapi_spec = fetch_openapi(swagger_url) if swagger_url else None
if swagger_url and not openapi_spec:
    st.warning("Unable to fetch OpenAPI document. Check the URL or network access.")

base_url = derive_base_url(swagger_url, openapi_spec) if swagger_url else ""
model_endpoint = None
models: List[str] = []

if openapi_spec:
    endpoints = discover_model_endpoints(openapi_spec)
    if endpoints:
        endpoint_labels = {f"{ep.method.upper()} {ep.path}": ep for ep in endpoints}
        chosen_label = st.selectbox("Available model endpoints", list(endpoint_labels.keys()))
        model_endpoint = endpoint_labels[chosen_label]
        if st.button("Fetch LLMs", type="secondary"):
            try:
                models = fetch_llms(base_url, model_endpoint, api_key)
                st.session_state["llm_options"] = models
                if not models:
                    st.warning("No models were returned by the endpoint.")
            except Exception as exc:  # pragma: no cover - UI feedback
                st.error(f"Failed to fetch models: {exc}")
    else:
        st.info("No GET endpoints containing 'model' or 'llm' were discovered in the OpenAPI spec.")

if "llm_options" in st.session_state:
    models = st.session_state["llm_options"]

chosen_model = None
if models:
    chosen_model = st.selectbox("Select an LLM", models)

st.markdown("---")
st.header("Compare columns")

if selection_a and selection_b and chosen_model:
    df1 = read_sheet(selection_a.excel_file, selection_a.sheet_name)
    df2 = read_sheet(selection_b.excel_file, selection_b.sheet_name)

    col_values_a = df1[selection_a.column].dropna().astype(str).tolist()
    col_values_b = df2[selection_b.column].dropna().astype(str).tolist()

    direction = st.radio(
        "Choose comparison direction",
        (
            "File 1 as policy statements ➜ best control in File 2",
            "File 2 as policy statements ➜ best control in File 1",
        ),
    )

    if not col_values_a or not col_values_b:
        st.warning("Selected columns are empty.")
    else:
        if direction.startswith("File 1"):
            policies, controls = col_values_a, col_values_b
            policy_label, control_label = selection_a.column, selection_b.column
        else:
            policies, controls = col_values_b, col_values_a
            policy_label, control_label = selection_b.column, selection_a.column

        st.write(
            f"Finding the best matching {control_label!r} control for each {policy_label!r} policy statement."
        )
        if st.button("Generate similarity scores", type="primary"):
            with st.spinner("Contacting the LLM and computing similarities..."):
                results_df, errors = build_best_match_dataframe(
                    policies, controls, base_url, api_key, chosen_model
                )
            if not results_df.empty:
                st.dataframe(results_df)
                excel_bytes = make_download(results_df)
                st.download_button(
                    label="Download results as Excel",
                    data=excel_bytes,
                    file_name="similarity_scores.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.warning("No similarity scores were generated.")

            if errors:
                st.info("\n".join(errors))
else:
    st.info("Upload both files, select columns, and choose an LLM to start.")
