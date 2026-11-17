"""Shared Streamlit helpers and visual styling."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DISCLAIMER = "Educational project — not for clinical use."

APP_CSS = """
<style>
:root { --ink:#173b3a; --muted:#607674; --teal:#16877d; --mint:#eaf6f2; --line:#dce9e5; }
.stApp { background: #f7faf9; color: var(--ink); }
.block-container { max-width: 1180px; padding-top: 2.1rem; padding-bottom: 3rem; }
h1, h2, h3 { color: #173b3a; letter-spacing: -0.025em; }
h1 { font-weight: 750; }
[data-testid="stSidebar"] { background: #f0f7f5; border-right: 1px solid #dce9e5; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color:#173b3a; }
[data-testid="stMetric"] { background:#ffffff; border:1px solid var(--line); padding:1rem 1.1rem; border-radius:14px; }
div[data-testid="stForm"] { background:#ffffff; border:1px solid var(--line); padding:1.35rem; border-radius:18px; }
.stButton > button, .stFormSubmitButton > button { border-radius:10px; font-weight:650; min-height:2.7rem; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background:#16877d; border-color:#16877d; }
div[data-testid="stAlert"] { border-radius:12px; }
div[data-testid="stExpander"] { background:#ffffff; border:1px solid var(--line); border-radius:12px; }
.ds-eyebrow { color:#16877d; font-size:.78rem; font-weight:750; letter-spacing:.11em; text-transform:uppercase; margin-bottom:.35rem; }
.ds-subtitle { color:#607674; font-size:1.03rem; line-height:1.65; max-width:850px; }
.ds-panel { background:#ffffff; border:1px solid #dce9e5; border-radius:16px; padding:1.1rem 1.25rem; }
.ds-note { background:#eaf6f2; border-left:4px solid #16877d; padding:.9rem 1rem; border-radius:8px; color:#234b47; }
footer { visibility:hidden; }
</style>
"""


def setup_page(title: str, icon: str = "🩺") -> None:
    """Configure a consistent, calm healthcare-style layout on every page."""
    st.set_page_config(
        page_title=f"{title} · Diabetes Screening",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.sidebar.markdown("## 🩺 Diabetes Screening")
    st.sidebar.caption("A public education and screening-support project")
    st.sidebar.warning(DISCLAIMER)
    st.sidebar.caption("Your screening answers are processed for this session and are not saved to a database.")


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the trained sklearn Pipeline once per server process (never train inside the app)."""
    return joblib.load(ROOT / "models" / "pipeline.joblib")


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    return json.loads((ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_report(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / name)


@st.cache_data(show_spinner=False)
def load_eval_predictions() -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / "eval_predictions.csv.gz")


def active_threshold(meta: dict) -> tuple[float, bool]:
    """Threshold used by the Risk Screening page. Returns (value, is_user_override)."""
    override = st.session_state.get("threshold_override")
    if override is not None:
        return float(override), True
    return float(meta["threshold"]), False
