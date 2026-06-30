"""Streamlit dashboard for customer churn monitoring and prediction."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable regardless of how this script is launched
# (`streamlit run dashboard/app.py`, `python dashboard/app.py`, or as a package).
# Running a script directly puts its own directory on sys.path, not the project
# root, which breaks the `dashboard.*` and `src.*` absolute imports below.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.service import (  # noqa: E402
    DEFAULT_CUSTOMER,
    build_customer_payload,
    churn_rate_by,
    get_health,
    get_model_info,
    load_dashboard_dataset,
    load_local_model_info,
    local_prediction,
    request_prediction,
)
from src.config import settings  # noqa: E402

st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.24);
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: inherit;
    }
    div[data-testid="stForm"] {
        border: 1px solid rgba(128, 128, 128, 0.24);
        border-radius: 8px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def cached_dataset() -> pd.DataFrame:
    """Load dashboard data with a short cache."""
    return load_dashboard_dataset()


@st.cache_data(ttl=30)
def cached_health(api_url: str) -> dict:
    """Load API health with a short cache."""
    return get_health(api_url)


def model_info(api_url: str) -> dict:
    """Read model metadata from the API or local disk."""
    try:
        return get_model_info(api_url)
    except requests.RequestException:
        return load_local_model_info()


def prediction_panel(api_url: str) -> None:
    """Render the customer prediction form."""
    defaults = DEFAULT_CUSTOMER
    with st.form("customer_prediction"):
        left, middle, right = st.columns(3)
        with left:
            customer_id = st.text_input("Customer ID", value=str(defaults["customerID"]))
            gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            senior_citizen = st.selectbox(
                "Senior Citizen", [0, 1], index=int(defaults["SeniorCitizen"])
            )
            partner = st.selectbox("Partner", ["Yes", "No"], index=0)
            dependents = st.selectbox("Dependents", ["Yes", "No"], index=1)
            tenure = st.number_input(
                "Tenure", min_value=0, max_value=120, value=int(defaults["tenure"])
            )
        with middle:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0)
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"], index=0
            )
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"], index=1
            )
            online_security = st.selectbox(
                "Online Security", ["No", "Yes", "No internet service"], index=0
            )
            online_backup = st.selectbox(
                "Online Backup", ["No", "Yes", "No internet service"], index=1
            )
            device_protection = st.selectbox(
                "Device Protection", ["No", "Yes", "No internet service"], index=0
            )
        with right:
            tech_support = st.selectbox(
                "Tech Support", ["No", "Yes", "No internet service"], index=0
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"], index=1
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["No", "Yes", "No internet service"], index=1
            )
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=0,
            )

        charge_left, charge_right = st.columns(2)
        with charge_left:
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=float(defaults["MonthlyCharges"]),
                step=1.0,
            )
        with charge_right:
            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=float(defaults["TotalCharges"]),
                step=10.0,
            )

        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if not submitted:
        return

    payload = build_customer_payload(
        {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
    )

    try:
        result = request_prediction(payload, api_url)
        source = "API"
    except requests.RequestException:
        try:
            result = local_prediction(payload)
            source = "local model"
        except Exception as exc:
            st.error(f"Prediction unavailable: {exc}")
            return

    probability = result["churn_probability"]
    label = result["prediction_label"]
    st.subheader(f"Prediction: {label}")
    st.progress(min(max(probability, 0.0), 1.0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn Probability", f"{probability:.1%}")
    c2.metric("Threshold", f"{result['threshold']:.1%}")
    c3.metric("Source", source)


def artifact_panel(info: dict) -> None:
    """Render model metadata and generated artifacts."""
    if info:
        st.json(info, expanded=False)
    else:
        st.warning("Model metadata is not available.")

    comparison_path = settings.comparison_path
    if comparison_path.exists():
        st.dataframe(pd.read_csv(comparison_path), use_container_width=True)

    image_paths = [
        settings.figures_dir / "confusion_matrix.png",
        settings.figures_dir / "roc_curve.png",
        settings.figures_dir / "precision_recall_curve.png",
        settings.figures_dir / "feature_importance.png",
        settings.figures_dir / "class_imbalance.png",
    ]
    available = [path for path in image_paths if Path(path).exists()]
    for row_index in range(0, len(available), 2):
        cols = st.columns(2)
        for column, image_path in zip(cols, available[row_index : row_index + 2], strict=False):
            column.image(str(image_path), caption=image_path.stem.replace("_", " ").title())


def overview_panel(data: pd.DataFrame) -> None:
    """Render dataset and churn charts."""
    st.dataframe(data.head(20), use_container_width=True)
    left, right = st.columns(2)
    with left:
        contract_rates = churn_rate_by(data, "Contract")
        if not contract_rates.empty:
            st.bar_chart(contract_rates.set_index("Contract"))
    with right:
        payment_rates = churn_rate_by(data, "PaymentMethod")
        if not payment_rates.empty:
            st.bar_chart(payment_rates.set_index("PaymentMethod"))

    charge_cols = [
        column for column in ["MonthlyCharges", "TotalCharges", "tenure"] if column in data.columns
    ]
    if charge_cols:
        st.line_chart(
            data[charge_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        )


def main() -> None:
    """Run the dashboard."""
    st.title("Customer Churn Dashboard")
    api_url = st.sidebar.text_input("API URL", value=settings.dashboard_api_url)
    health = cached_health(api_url)
    data = cached_dataset()
    info = model_info(api_url)

    target = data.get(settings.target_column)
    if target is not None and target.dtype.kind not in {"i", "u", "f", "b"}:
        target = target.astype(str).str.strip().map({"No": 0, "Yes": 1}).fillna(0).astype(int)

    metric_cols = st.columns(4)
    metric_cols[0].metric("API Status", str(health.get("status", "unknown")).title())
    metric_cols[1].metric("Rows", f"{len(data):,}")
    metric_cols[2].metric("Churn Rate", f"{float(target.mean() if target is not None else 0):.1%}")
    metric_cols[3].metric("Model", info.get("model_name", "Not trained"))

    overview, predictor, artifacts = st.tabs(["Overview", "Predict", "Artifacts"])
    with overview:
        overview_panel(data)
    with predictor:
        prediction_panel(api_url)
    with artifacts:
        artifact_panel(info)


def _is_running_under_streamlit() -> bool:
    """Return True when executing inside an active Streamlit runtime."""
    try:
        from streamlit.runtime import exists

        return exists()
    except Exception:
        return False


if __name__ == "__main__":
    if _is_running_under_streamlit():
        # Launched via `streamlit run dashboard/app.py`.
        main()
    else:
        # Launched via `python dashboard/app.py`: re-exec under the Streamlit
        # runtime so the dashboard server actually starts.
        import subprocess

        sys.exit(
            subprocess.call(
                [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]
            )
        )
