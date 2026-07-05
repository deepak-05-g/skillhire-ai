"""
SkillHire AI — Streamlit frontend for resume parsing, job fetching, and recommendations.
"""

from __future__ import annotations

import base64
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SkillHire AI — Resume-Based Job Finder",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_local_env() -> None:
    """Load simple KEY=VALUE pairs from the project .env file for local runs."""
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


load_local_env()

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:8501")
AUTH_BACKEND_URL = os.environ.get("AUTH_BACKEND_URL", "http://localhost:8000")
PARSE_ENDPOINT = f"{BACKEND_URL}/api/v1/resume/parse"
PARSE_TEXT_ENDPOINT = f"{BACKEND_URL}/api/v1/resume/parse-text"
FETCH_JOBS_ENDPOINT = f"{BACKEND_URL}/api/v1/jobs/fetch"
STORED_JOBS_ENDPOINT = f"{BACKEND_URL}/api/v1/jobs/stored"
SEED_JOBS_ENDPOINT = f"{BACKEND_URL}/api/v1/jobs/seed"
RECOMMEND_ENDPOINT = f"{BACKEND_URL}/api/v1/recommend/jobs"
RECOMMEND_STORED_ENDPOINT = f"{BACKEND_URL}/api/v1/recommend/stored-jobs"
OFFICIAL_SEARCH_ENDPOINT = f"{BACKEND_URL}/api/v1/jobs/official-search"
SAVE_JOB_ENDPOINT = f"{BACKEND_URL}/api/v1/jobs/save"
SAVED_JOBS_ENDPOINT = f"{BACKEND_URL}/api/v1/jobs/saved"
ML_STATUS_ENDPOINT = f"{BACKEND_URL}/api/v1/recommend/ml-model/status"
ML_TRAIN_ENDPOINT = f"{BACKEND_URL}/api/v1/recommend/ml-model/train"
ML_DOWNLOAD_ENDPOINT = f"{BACKEND_URL}/api/v1/recommend/ml-model/download"
CHAT_ENDPOINT = f"{BACKEND_URL}/api/v1/chat"

FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY", "")
FIREBASE_AUTH_DOMAIN = os.environ.get("FIREBASE_AUTH_DOMAIN", "")
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "")
FIREBASE_STORAGE_BUCKET = os.environ.get("FIREBASE_STORAGE_BUCKET", "")
FIREBASE_MESSAGING_SENDER_ID = os.environ.get("FIREBASE_MESSAGING_SENDER_ID", "")
FIREBASE_APP_ID = os.environ.get("FIREBASE_APP_ID", "")
FIREBASE_CONFIGURED = all(
    [FIREBASE_API_KEY, FIREBASE_AUTH_DOMAIN, FIREBASE_PROJECT_ID, FIREBASE_APP_ID]
)


JOB_SOURCE_SAMPLE = "Use sample jobs"
JOB_SOURCE_GREENHOUSE = "Fetch from Greenhouse"
JOB_SOURCE_LEVER = "Fetch from Lever"
JOB_SOURCE_ASHBY = "Fetch from Ashby"

SECTION_LABELS = {
    "skills": "Skills",
    "projects": "Projects",
    "education": "Education",
    "experience": "Experience",
    "certifications": "Certifications",
}

CLOUD_SKILLS = {"AWS", "Azure", "Google Cloud", "GCP"}
ML_SKILLS = {
    "Machine Learning", "Deep Learning", "NLP", "TensorFlow",
    "PyTorch", "Scikit-learn", "Pandas", "NumPy",
}
DEPLOYMENT_TERMS = re.compile(
    r"\b(streamlit|fastapi|flask|deploy(?:ment|ed)?|docker|kubernetes|api|heroku|vercel)\b",
    re.IGNORECASE,
)
LINK_PATTERN = re.compile(
    r"(https?://\S+|github\.com/\S+|gitlab\.com/\S+)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Global Styles — professional light theme
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Layout ─────────────────────────────────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }

    /* Clean off-white page background */
    .stApp {
        background: #f1f5f9;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #334155 !important;
    }

    /* ── Hero header ────────────────────────────────────────────────────── */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1 0%, #0ea5e9 60%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.8rem;
        font-weight: 400;
    }
    .auth-product-row {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        min-height: 2.6rem;
    }
    .auth-product-mark {
        width: 2.35rem;
        height: 2.35rem;
        border-radius: 10px;
        display: grid;
        place-items: center;
        color: #ffffff;
        background: linear-gradient(135deg, #2f4d46, #659287);
        font-size: 0.95rem;
        font-weight: 900;
        letter-spacing: 0;
        box-shadow: 0 10px 22px rgba(47, 77, 70, 0.18);
    }
    .auth-product-name {
        color: #0f172a;
        font-size: 0.92rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .auth-product-subtitle {
        color: #64748b;
        font-size: 0.76rem;
        line-height: 1.25;
        margin-top: 0.12rem;
    }
    .auth-popover-title {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .auth-popover-copy {
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.45;
        margin-bottom: 0.75rem;
    }
    .auth-benefit-row {
        display: flex;
        align-items: flex-start;
        gap: 0.55rem;
        padding: 0.55rem 0;
        border-top: 1px solid #e2e8f0;
    }
    .auth-benefit-dot {
        width: 0.48rem;
        height: 0.48rem;
        border-radius: 50%;
        background: #659287;
        margin-top: 0.42rem;
        flex: 0 0 auto;
    }
    .auth-benefit-title {
        color: #0f172a;
        font-size: 0.82rem;
        font-weight: 800;
        line-height: 1.25;
    }
    .auth-benefit-copy {
        color: #64748b;
        font-size: 0.76rem;
        line-height: 1.35;
        margin-top: 0.1rem;
    }
    .firebase-status {
        border: 1px solid #d6e6dc;
        border-radius: 10px;
        background: #f6faf3;
        color: #2f4d46;
        padding: 0.65rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 700;
        line-height: 1.35;
        margin: 0.75rem 0;
    }
    div[data-testid="stPopover"] button {
        border-radius: 999px !important;
        font-weight: 800 !important;
    }

    /* ── Cards ──────────────────────────────────────────────────────────── */
    .glass-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(99,102,241,0.06);
        transition: box-shadow 0.2s;
    }
    .glass-card:hover {
        box-shadow: 0 4px 20px rgba(99,102,241,0.12);
    }

    .job-result-card {
        background: #ffffff;
        border: 1.8px solid #2f4d46;
        border-radius: 18px;
        padding: 1.1rem 1.15rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 10px 24px rgba(47, 77, 70, 0.08);
    }
    .job-result-card .job-card-title,
    .job-result-card .job-card-meta,
    .job-result-card .job-card-summary {
        color: #2f4d46;
    }
    .job-result-card .source-chip {
        border-color: #88bda4;
        background: #e6f2dd;
        color: #4f7f74;
    }
    .job-result-card:hover {
        box-shadow: 0 14px 30px rgba(47, 77, 70, 0.12);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1.8px solid #2f4d46 !important;
        border-radius: 18px !important;
        background: #ffffff !important;
        box-shadow: 0 10px 24px rgba(47, 77, 70, 0.08) !important;
        padding: 0.95rem 1rem !important;
        margin-bottom: 0.95rem !important;
    }

    /* ── Skill tags ─────────────────────────────────────────────────────── */
    .skill-tag {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        padding: 0.18rem 0.65rem;
        margin: 0.15rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        border: 1px solid #c4b5fd;
        letter-spacing: 0.01em;
    }

    /* ── Fit label badges ───────────────────────────────────────────────── */
    .badge-high   { background: #dcfce7; color: #166534; border: 1px solid #86efac; padding: 0.2rem 0.75rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }
    .badge-medium { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; padding: 0.2rem 0.75rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }
    .badge-low    { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; padding: 0.2rem 0.75rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }

    /* ── Section headers ────────────────────────────────────────────────── */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e0e7ff;
    }

    /* ── ML metric cards ────────────────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e0f2fe 100%);
        border: 1px solid #c7d2fe;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(99,102,241,0.08);
    }
    .metric-value { font-size: 1.9rem; font-weight: 800; color: #4f46e5; }
    .metric-label { font-size: 0.78rem; color: #64748b; font-weight: 600; margin-top: 0.15rem; text-transform: uppercase; letter-spacing: 0.05em; }

    /* ── Status pills ───────────────────────────────────────────────────── */
    .status-online  { background: #dcfce7; color: #166534; border: 1px solid #86efac; padding: 0.28rem 0.9rem; border-radius: 999px; font-size: 0.83rem; font-weight: 600; display: inline-block; }
    .status-offline { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; padding: 0.28rem 0.9rem; border-radius: 999px; font-size: 0.83rem; font-weight: 600; display: inline-block; }

    /* ── Streamlit widget overrides ─────────────────────────────────────── */
    /* Main content text colour */
    .stMarkdown p, .stMarkdown li, .stText, .stCaption,
    [data-testid="stText"], label, .stRadio label span {
        color: #334155;
    }
    h1, h2, h3, h4, h5 { color: #1e293b !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid #e2e8f0;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.92rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #6366f1, #0ea5e9) !important;
        color: #ffffff !important;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #6366f1, #0ea5e9);
        border: none;
        color: #fff;
        font-weight: 600;
        border-radius: 8px;
        transition: opacity 0.2s, transform 0.1s;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"] {
        border: 1.5px solid #6366f1;
        color: #6366f1;
        background: #ffffff;
        font-weight: 600;
        border-radius: 8px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #f8faff;
        border: 2px dashed #a5b4fc;
        border-radius: 10px;
        padding: 0.5rem;
    }

    /* Metric widgets */
    [data-testid="stMetricValue"] { color: #4f46e5 !important; font-weight: 800; }
    [data-testid="stMetricLabel"] { color: #64748b !important; }

    /* Info/Success/Warning/Error message boxes */
    .stAlert { border-radius: 10px; }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
    }

    /* Charts — light background */
    .element-container iframe { border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
    :root {
        --surface: #ffffff;
        --surface-subtle: #e6f2dd;
        --page: #e6f2dd;
        --border: #b1d3b9;
        --border-strong: #88bda4;
        --text: #2f4d46;
        --muted: #667d74;
        --accent: #659287;
        --accent-dark: #4f7f74;
        --success: #659287;
        --warning: #88bda4;
        --danger: #a97268;
    }

    .stApp {
        background: var(--page);
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }

    [data-testid="stSidebar"] {
        background: #f4faf0;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] hr {
        margin: 1rem 0;
        border-color: var(--border);
    }

    .brand-block {
        padding: 0.25rem 0 0.8rem 0;
    }
    .brand-title {
        color: var(--text);
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 0;
        line-height: 1.2;
    }
    .brand-subtitle,
    .app-subtitle,
    .panel-subtitle {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.45;
        margin-top: 0.2rem;
    }

    .app-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(230,242,221,0.92));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        position: relative;
        overflow: hidden;
    }
    .app-header::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, rgba(101, 146, 135, 0.10), rgba(136, 189, 164, 0.10), transparent 70%);
        pointer-events: none;
    }
    .app-title {
        color: var(--text);
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.2;
        letter-spacing: 0;
        position: relative;
        z-index: 1;
    }
    .app-eyebrow {
        color: var(--accent);
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
        position: relative;
        z-index: 1;
    }

    .hero-shell {
        background: radial-gradient(circle at top right, rgba(101, 146, 135, 0.12), transparent 30%),
                    radial-gradient(circle at bottom left, rgba(177, 211, 185, 0.16), transparent 26%),
                    linear-gradient(135deg, rgba(255,255,255,0.98), rgba(230,242,221,0.94));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.25rem 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 36px rgba(15, 23, 42, 0.06);
    }
    .hero-shell__meta {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.55;
        max-width: 860px;
    }
    .hero-shell__badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.9rem;
    }
    .pill-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.34rem 0.68rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.78);
        color: var(--text);
        font-size: 0.77rem;
        font-weight: 700;
    }
    .kpi-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.95rem 1rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
        height: 100%;
    }
    .kpi-card__label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .kpi-card__value {
        color: var(--text);
        font-size: 1.4rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .kpi-card__hint {
        color: var(--muted);
        font-size: 0.8rem;
        margin-top: 0.35rem;
        line-height: 1.35;
    }
    .kpi-card--accent {
        border-color: rgba(101, 146, 135, 0.24);
        background: linear-gradient(180deg, rgba(230, 242, 221, 0.96), rgba(255,255,255,1));
    }
    .kpi-card--success {
        border-color: rgba(136, 189, 164, 0.24);
        background: linear-gradient(180deg, rgba(177, 211, 185, 0.30), rgba(255,255,255,1));
    }
    .kpi-card--warning {
        border-color: rgba(177, 211, 185, 0.30);
        background: linear-gradient(180deg, rgba(230, 242, 221, 0.90), rgba(255,255,255,1));
    }
    .kpi-card--neutral {
        background: linear-gradient(180deg, rgba(230, 242, 221, 0.58), rgba(255,255,255,1));
    }

    .dashboard-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem 1.05rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
    }
    .dashboard-panel + .dashboard-panel {
        margin-top: 1rem;
    }
    .panel-title {
        color: var(--text);
        font-size: 0.92rem;
        font-weight: 800;
        letter-spacing: 0;
        margin-bottom: 0.2rem;
    }
    .panel-subtitle {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.45;
        margin-bottom: 0.85rem;
    }
    .workflow-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.7rem;
        margin: 0.7rem 0 0.15rem 0;
    }
    .workflow-step {
        background: var(--surface-subtle);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.8rem 0.85rem;
    }
    .workflow-step__index {
        color: var(--accent);
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .workflow-step__title {
        color: var(--text);
        font-size: 0.86rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }
    .workflow-step__copy {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.4;
        margin-top: 0.18rem;
    }

    .glass-card,
    [data-testid="stExpander"] {
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
        background: var(--surface) !important;
        box-shadow: 0 8px 22px rgba(16, 24, 40, 0.05) !important;
    }
    .glass-card {
        padding: 1.05rem 1.15rem;
        margin-bottom: 0.85rem;
    }
    .glass-card:hover {
        box-shadow: 0 12px 30px rgba(16, 24, 40, 0.08) !important;
    }

    .section-header {
        color: var(--text);
        font-size: 0.9rem;
        font-weight: 800;
        letter-spacing: 0;
        padding-bottom: 0.45rem;
        margin: 0.2rem 0 0.8rem 0;
        border-bottom: 1px solid var(--border);
    }

    .skill-tag,
    .missing-tag {
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 650;
        padding: 0.16rem 0.48rem;
        margin: 0.12rem;
        letter-spacing: 0;
    }
    .skill-tag {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }
    .missing-tag {
        display: inline-block;
        background: #fff7ed;
        color: #9a3412;
        border: 1px solid #fed7aa;
    }

    .badge-high,
    .badge-medium,
    .badge-low,
    .status-online,
    .status-offline {
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 750;
        padding: 0.22rem 0.55rem;
        letter-spacing: 0;
        display: inline-block;
    }
    .badge-high,
    .status-online {
        background: #ecfdf3;
        color: #027a48;
        border: 1px solid #abefc6;
    }
    .badge-medium {
        background: #fffaeb;
        color: #b54708;
        border: 1px solid #fedf89;
    }
    .badge-low,
    .status-offline {
        background: #fef3f2;
        color: #b42318;
        border: 1px solid #fecdca;
    }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        box-shadow: none;
        text-align: left;
    }
    .metric-value {
        color: var(--text);
        font-size: 1.45rem;
    }
    .metric-label {
        color: var(--muted);
        font-size: 0.72rem;
        letter-spacing: 0.04em;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border: 0;
        border-radius: 0;
        border-bottom: 1px solid var(--border);
        padding: 0;
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0;
        color: var(--muted);
        font-size: 0.9rem;
        font-weight: 700;
        padding: 0.75rem 0.1rem;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent);
    }

    .stButton > button,
    .stDownloadButton > button,
    .stLinkButton > a {
        border-radius: 6px !important;
        font-weight: 700 !important;
        letter-spacing: 0 !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        border: 1px solid var(--accent-dark) !important;
        color: white !important;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.08);
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--accent-dark) !important;
        transform: none !important;
    }
    .stButton > button[kind="secondary"] {
        border: 1px solid var(--border-strong) !important;
        color: var(--text) !important;
        background: var(--surface) !important;
    }

    [data-testid="stFileUploader"] {
        background: var(--surface-subtle);
        border: 1px dashed var(--border-strong);
        border-radius: 8px;
    }
    textarea,
    input,
    [data-baseweb="select"] > div {
        border-radius: 6px !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 1.45rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.76rem !important;
    }
    .stAlert {
        border-radius: 14px;
        border: 1px solid var(--border);
    }

    .job-card-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.75rem;
        margin-bottom: 0.6rem;
    }
    .job-card-title {
        color: var(--text);
        font-size: 1rem;
        font-weight: 800;
        line-height: 1.25;
        margin: 0;
    }
    .job-card-meta {
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.45;
        margin-top: 0.16rem;
    }
    .job-card-summary {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.5;
        margin: 0.35rem 0 0.7rem 0;
    }
    .job-card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.65rem;
    }
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.24rem 0.5rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: var(--surface-subtle);
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
    }

    .chat-launcher-anchor {
        scroll-margin-top: 120px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# SECTION: Helper — data loading
# ============================================================================

@st.cache_data
def load_skills_taxonomy() -> Dict[str, List[str]]:
    """Load skill categories from data/skills.json."""
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "..", "data", "skills.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_sample_jobs() -> List[Dict[str, Any]]:
    """Load offline sample jobs from data/sample_jobs.json."""
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "..", "data", "sample_jobs.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.error(f"Could not load sample jobs: {exc}")
        return []


def group_skills_by_category(skills: List[str]) -> Dict[str, List[str]]:
    """Map extracted skills into taxonomy categories."""
    taxonomy = load_skills_taxonomy()
    skills_lower = {s.lower(): s for s in skills}
    grouped: Dict[str, List[str]] = {}
    assigned: set = set()

    for category, category_skills in taxonomy.items():
        matches = []
        for skill in category_skills:
            if skill.lower() in skills_lower:
                display = skills_lower[skill.lower()]
                matches.append(display)
                assigned.add(display.lower())
        if matches:
            grouped[category] = sorted(matches)

    uncategorized = sorted(s for s in skills if s.lower() not in assigned)
    if uncategorized:
        grouped["Other"] = uncategorized

    return grouped


def get_fit_label(score: int) -> str:
    """Return a human-readable fit label from match score."""
    if score >= 80:
        return "Excellent Fit"
    if score >= 65:
        return "Good Fit"
    if score >= 50:
        return "Moderate Fit"
    return "Stretch Role"


@st.cache_data
def _all_taxonomy_skills() -> List[str]:
    """Flat list of canonical skills from skills.json."""
    taxonomy = load_skills_taxonomy()
    skills: List[str] = []
    for category_skills in taxonomy.values():
        skills.extend(category_skills)
    return sorted(set(skills), key=len, reverse=True)


def extract_skills_from_text(text: str) -> List[str]:
    """Match taxonomy skills in job/resume text (longest match first)."""
    if not text:
        return []

    text_lower = text.lower()
    found: set = set()
    for skill in _all_taxonomy_skills():
        pattern = r"(?<![a-zA-Z0-9+#.])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9+#.])"
        if re.search(pattern, text_lower):
            found.add(skill)
    return sorted(found)


def _job_text(job: Dict[str, Any]) -> str:
    return " ".join(
        job.get(field, "")
        for field in ("title", "description", "requirements")
        if job.get(field)
    )


def compute_market_insights(
    jobs: List[Dict[str, Any]],
    recommendations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Aggregate analytics from fetched and recommended jobs."""
    recommendations = recommendations or []

    demanded_counter: Counter = Counter()
    for job in jobs:
        demanded_counter.update(extract_skills_from_text(_job_text(job)))

    missing_counter: Counter = Counter()
    for rec in recommendations:
        missing_counter.update(rec.get("missing_skills", []))

    source_counter: Counter = Counter()
    for job in jobs:
        source_counter[job.get("source", "Unknown")] += 1

    location_counter: Counter = Counter()
    for job in jobs:
        location = job.get("location", "Unknown").strip() or "Unknown"
        location_counter[location] += 1

    scores = [r.get("match_score", 0) for r in recommendations if r.get("match_score") is not None]
    avg_match = round(sum(scores) / len(scores), 1) if scores else None

    return {
        "demanded_skills": demanded_counter.most_common(10),
        "missing_skills": missing_counter.most_common(10),
        "jobs_by_source": source_counter.most_common(),
        "jobs_by_location": location_counter.most_common(10),
        "average_match_score": avg_match,
        "job_count": len(jobs),
        "recommendation_count": len(recommendations),
    }


# ============================================================================
# SECTION: Chart Helpers
# ============================================================================

def _dark_chart_style():
    """Apply a clean light matplotlib theme matching the portal design."""
    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#f8fafc",
        "axes.edgecolor": "#e2e8f0",
        "axes.labelcolor": "#64748b",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "text.color": "#1e293b",
        "grid.color": "#e2e8f0",
        "font.family": "DejaVu Sans",
    })


def _plot_horizontal_bar(
    title: str,
    labels: List[str],
    values: List[int],
    xlabel: str,
    color: str,
) -> plt.Figure:
    """Create a styled horizontal bar chart."""
    _dark_chart_style()
    fig, ax = plt.subplots(figsize=(9, max(3.5, len(labels) * 0.45)))
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color=color, edgecolor="none", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color="#1e293b")
    ax.set_xlabel(xlabel, color="#64748b")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def _plot_source_chart(title: str, labels: List[str], values: List[int]) -> plt.Figure:
    """Pie / bar chart for job sources."""
    _dark_chart_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(labels) <= 5:
        colors = ["#6366f1", "#0ea5e9", "#10b981", "#f59e0b", "#f43f5e"]
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.0f%%",
            startangle=90,
            colors=colors[: len(labels)],
        )
        for t in texts:
            t.set_color("#334155")
        for t in autotexts:
            t.set_color("#ffffff")
            t.set_fontweight("bold")
        ax.set_title(title, fontsize=12, fontweight="bold", color="#1e293b")
    else:
        ax.bar(labels, values, color="#6366f1", edgecolor="none")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color="#1e293b")
        ax.set_ylabel("Number of jobs", color="#64748b")
        ax.tick_params(axis="x", rotation=30, colors="#64748b")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    return fig


def _plot_feature_importance(
    feature_importances: Dict[str, float],
) -> plt.Figure:
    """Create a premium feature importance chart for the ML model."""
    _dark_chart_style()
    sorted_items = sorted(feature_importances.items(), key=lambda x: x[1])
    labels = [k.replace("_", " ").title() for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    # Colour gradient: indigo → sky blue
    max_val = max(values) if values else 1
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("cool")
    colors = [cmap(v / max_val * 0.8 + 0.1) for v in values]

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(labels) * 0.55)))
    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.55)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#334155",
        )

    ax.set_title("Feature Importances — Random Forest Classifier", fontsize=12, fontweight="bold", pad=12, color="#1e293b")
    ax.set_xlabel("Importance Score", color="#64748b")
    ax.set_xlim(0, max(values) * 1.2 if values else 1)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def _plot_score_distribution(scores: List[int]) -> plt.Figure:
    """Histogram of match scores for current recommendations."""
    _dark_chart_style()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(scores, bins=10, range=(0, 100), color="#6366f1", edgecolor="#ffffff", linewidth=0.8, alpha=0.85)
    ax.axvline(x=50, color="#f43f5e", linestyle="--", linewidth=1.5, label="50% threshold")
    ax.axvline(x=75, color="#10b981", linestyle="--", linewidth=1.5, label="75% threshold")
    ax.set_title("Match Score Distribution", fontsize=12, fontweight="bold", color="#1e293b")
    ax.set_xlabel("Match Score (%)", color="#64748b")
    ax.set_ylabel("Jobs", color="#64748b")
    ax.legend(framealpha=0.8, labelcolor="#334155", facecolor="#ffffff", edgecolor="#e2e8f0")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


# ============================================================================
# SECTION: Backend & API calls
# ============================================================================

def check_backend(timeout: int = 8, attempts: int = 1) -> Tuple[bool, str]:
    """Return (is_online, status_message)."""
    last_message = f"Backend not reachable at {BACKEND_URL}"
    for _ in range(max(attempts, 1)):
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=timeout)
            if response.status_code == 200:
                version = response.json().get("version", "unknown")
                return True, f"Backend connected (v{version})"
            last_message = f"Backend returned status {response.status_code}"
        except requests.RequestException as exc:
            last_message = f"Backend not reachable at {BACKEND_URL}: {exc}"
    return False, last_message


def parse_resume(uploaded_file) -> Optional[Dict[str, Any]]:
    """Parse resume PDF via backend."""
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf"),
    }
    response = requests.post(PARSE_ENDPOINT, files=files, timeout=30)
    if response.status_code == 200:
        return response.json()

    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    st.error(f"Resume parsing failed ({response.status_code}): {detail}")
    return None


def parse_pasted_resume(resume_text: str) -> Optional[Dict[str, Any]]:
    """Parse pasted resume text via backend."""
    payload = {
        "resume_text": resume_text,
        "filename": "pasted_resume.txt",
    }
    response = requests.post(PARSE_TEXT_ENDPOINT, json=payload, timeout=30)
    if response.status_code == 200:
        return response.json()

    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    st.error(f"Resume text parsing failed ({response.status_code}): {detail}")
    return None


def fetch_stored_jobs(limit: int = 500) -> List[Dict[str, Any]]:
    """Load jobs already stored in the project database."""
    response = requests.get(STORED_JOBS_ENDPOINT, params={"limit": limit}, timeout=30)
    if response.status_code == 200:
        return response.json().get("jobs", [])

    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    st.error(f"Could not load stored jobs ({response.status_code}): {detail}")
    return []


def seed_job_database(max_companies_per_source: int = 4) -> Dict[str, Any]:
    """Auto-populate the live database from all supported job board families."""
    payload = {
        "sources": ["greenhouse", "lever", "ashby"],
        "max_companies_per_source": max_companies_per_source,
    }
    response = requests.post(SEED_JOBS_ENDPOINT, json=payload, timeout=180)
    if response.status_code == 200:
        return response.json()

    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    st.error(f"Automatic job fetch failed ({response.status_code}): {detail}")
    return {"jobs": [], "count": 0, "sources": [], "errors": []}


def get_stored_recommendations(
    resume_text: str,
    resume_skills: List[str],
    use_ml: bool = True,
    job_limit: int = 500,
) -> Tuple[List[Dict[str, Any]], int]:
    """Rank database jobs against the parsed resume."""
    payload = {
        "resume_text": resume_text,
        "resume_skills": resume_skills,
        "use_ml": use_ml,
        "job_limit": job_limit,
    }
    response = requests.post(RECOMMEND_STORED_ENDPOINT, json=payload, timeout=120)
    if response.status_code == 200:
        data = response.json()
        return data.get("recommendations", []), int(data.get("jobs_analyzed", 0) or 0)

    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    st.error(f"Database recommendation failed ({response.status_code}): {detail}")
    return [], 0


def fetch_jobs(source: str, company: str) -> List[Dict[str, Any]]:
    """Fetch jobs from a live job board via backend."""
    params = {"source": source, "company": company.strip()}
    response = requests.get(FETCH_JOBS_ENDPOINT, params=params, timeout=20)
    if response.status_code == 200:
        return response.json()

    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    st.error(f"Job fetch failed ({response.status_code}): {detail}")
    return []


def get_recommendations(
    resume_text: str,
    resume_skills: List[str],
    jobs: List[Dict[str, Any]],
    use_ml: bool = True,
) -> List[Dict[str, Any]]:
    """Call /recommend/jobs and return ranked job list."""
    payload = {
        "resume_text": resume_text,
        "resume_skills": resume_skills,
        "jobs": jobs,
        "use_ml": use_ml,
    }
    response = requests.post(RECOMMEND_ENDPOINT, json=payload, timeout=90)
    if response.status_code == 200:
        data = response.json()
        return data.get("recommendations", data)

    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    st.error(f"Recommendation failed ({response.status_code}): {detail}")
    return []


def get_official_searches(
    resume_text: str,
    resume_skills: List[str],
    location: str = "",
    include_amazon: bool = True,
) -> List[Dict[str, Any]]:
    """Call /jobs/official-search and return Big Tech career search links."""
    payload = {
        "resume_text": resume_text,
        "resume_skills": resume_skills,
        "location": location or None,
        "include_amazon": include_amazon,
    }
    response = requests.post(OFFICIAL_SEARCH_ENDPOINT, json=payload, timeout=15)
    if response.status_code == 200:
        data = response.json()
        return data.get("official_search_sources", [])

    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    st.error(f"Official search links failed ({response.status_code}): {detail}")
    return []


def save_job_bookmark(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """POST /jobs/save — bookmark a recommended job."""
    payload = {
        "title": job.get("title", ""),
        "company": job.get("company", ""),
        "location": job.get("location", ""),
        "apply_url": job.get("apply_url", ""),
        "match_score": int(job.get("match_score", 0) or 0),
        "missing_skills": job.get("missing_skills", []) or [],
    }
    response = requests.post(SAVE_JOB_ENDPOINT, json=payload, timeout=15)
    if response.status_code in (200, 201):
        return response.json()

    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    st.error(f"Could not save job ({response.status_code}): {detail}")
    return None


def fetch_saved_jobs() -> List[Dict[str, Any]]:
    """GET /jobs/saved — list bookmarked jobs."""
    response = requests.get(SAVED_JOBS_ENDPOINT, timeout=15)
    if response.status_code == 200:
        return response.json().get("saved_jobs", [])

    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    st.error(f"Could not load saved jobs ({response.status_code}): {detail}")
    return []


def remove_saved_job(saved_job_id: int) -> bool:
    """DELETE /jobs/saved/{id} — remove a bookmark."""
    response = requests.delete(f"{SAVED_JOBS_ENDPOINT}/{saved_job_id}", timeout=15)
    if response.status_code == 200:
        return True

    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    st.error(f"Could not remove saved job ({response.status_code}): {detail}")
    return False


def get_ml_model_status() -> Optional[Dict[str, Any]]:
    """GET /recommend/ml-model/status — fetch model metrics and importances."""
    try:
        response = requests.get(ML_STATUS_ENDPOINT, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def trigger_model_retrain() -> Optional[Dict[str, Any]]:
    """POST /recommend/ml-model/train — trigger classifier retraining."""
    try:
        response = requests.post(ML_TRAIN_ENDPOINT, timeout=120)
        if response.status_code == 200:
            return response.json()
        return {"error": response.json().get("detail", "Retrain failed")}
    except Exception as e:
        return {"error": str(e)}


def send_chat_message(
    messages: List[Dict[str, str]],
    resume_text: str = "",
    resume_skills: List[str] = [],
    missing_skills: List[str] = [],
    job_recommendations: List[Dict[str, Any]] = [],
    career_goal: str = "",
) -> Optional[str]:
    """POST /chat — send conversation history and candidate context for advisory answers."""
    payload = {
        "messages": messages,
        "resume_text": resume_text,
        "resume_skills": resume_skills,
        "missing_skills": missing_skills,
        "job_recommendations": job_recommendations,
        "career_goal": career_goal,
    }
    try:
        response = requests.post(CHAT_ENDPOINT, json=payload, timeout=45)
        if response.status_code == 200:
            return response.json().get("response")
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
    except Exception as e:
        detail = str(e)

    st.error(f"Career Advisor chatbot request failed: {detail}")
    return None


def match_resume_against_database(
    parsed_resume: Dict[str, Any],
    use_ml: bool,
    job_limit: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Ensure jobs exist, then rank them against the parsed resume."""
    jobs = fetch_stored_jobs(limit=job_limit)
    if not jobs:
        st.info("No stored roles yet. Fetching roles from Greenhouse, Lever, and Ashby automatically...")
        seed_result = seed_job_database(max_companies_per_source=4)
        jobs = seed_result.get("jobs", [])[:job_limit]
        fetched_count = sum(
            int(item.get("fetched", 0) or 0)
            for item in seed_result.get("sources", [])
        )
        if jobs:
            st.success(f"Loaded {len(jobs)} stored roles from {fetched_count} live job-board results.")
        elif seed_result.get("errors"):
            st.warning("Live job boards did not return stored roles yet. Try again or use Quick Tools with a company handle.")

    if jobs:
        st.session_state["jobs_db"] = jobs
    else:
        st.session_state["jobs_db"] = []
        st.session_state.pop("recommendations", None)
        st.session_state["last_jobs_analyzed"] = 0
        return [], 0

    recommendations, jobs_analyzed = get_stored_recommendations(
        resume_text=parsed_resume.get("raw_text", ""),
        resume_skills=parsed_resume.get("skills", []),
        use_ml=use_ml,
        job_limit=job_limit,
    )
    if recommendations:
        st.session_state["recommendations"] = recommendations
        st.session_state["last_jobs_analyzed"] = jobs_analyzed
    else:
        st.session_state.pop("recommendations", None)
        st.session_state["last_jobs_analyzed"] = jobs_analyzed
    return recommendations, jobs_analyzed


def get_top_missing_skills(
    recommendations: List[Dict[str, Any]],
    limit: int = 10,
) -> List[str]:
    """Collect the most common missing skills from the top recommendations."""
    missing_counter: Counter = Counter()
    for rec in recommendations[:10]:
        for skill in rec.get("missing_skills", []):
            missing_counter[skill] += 1
    return [skill for skill, _ in missing_counter.most_common(limit)]


def display_career_chatbot(backend_online: bool) -> None:
    """Render the AI Career Advisor as part of the main resume workflow."""
    st.markdown(
        """
        <div class="dashboard-panel">
            <div class="panel-title">Career Advisor</div>
            <div class="panel-subtitle">Ask about matched roles, missing skills, resume positioning, or a learning plan.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not backend_online:
        st.warning("Start the backend to chat with the Career Advisor.")
        return

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Paste or upload your resume, run matching, and I can help compare roles, "
                    "improve resume positioning, and plan skills to build next."
                ),
            }
        ]

    parsed_resume = st.session_state.get("parsed_resume", {})
    resume_text = parsed_resume.get("raw_text", "")
    resume_skills = parsed_resume.get("skills", [])
    recommendations = st.session_state.get("recommendations", [])
    missing_skills = get_top_missing_skills(recommendations)

    career_goal = st.text_input(
        "Target role or career goal",
        placeholder="e.g. Backend Engineer, ML Intern, Full Stack Developer",
        key="advisor_career_goal",
    )

    with st.expander("Advisor context", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Resume skills", len(resume_skills))
        c2.metric("Job matches", len(recommendations))
        c3.metric("Skill gaps", len(missing_skills))

        if resume_skills:
            st.write("**Extracted skills:** " + ", ".join(resume_skills[:20]))
        else:
            st.caption("No resume parsed yet.")

        if missing_skills:
            st.write("**Top gaps:** " + ", ".join(missing_skills))
        elif recommendations:
            st.caption("No major gaps detected in the current top matches.")

    if st.button("Clear chat history", key="clear_advisor_chat"):
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": "Chat cleared. Ask me about your resume, matches, or skills to improve.",
            }
        ]
        st.rerun()

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your best jobs, skill gaps, or resume improvements..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})

        messages_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state["chat_messages"]
        ]

        with st.chat_message("assistant"):
            with st.spinner("Career Advisor is preparing guidance..."):
                ai_response = send_chat_message(
                    messages=messages_history,
                    resume_text=resume_text,
                    resume_skills=resume_skills,
                    missing_skills=missing_skills,
                    job_recommendations=recommendations,
                    career_goal=career_goal,
                )
                if ai_response:
                    st.markdown(ai_response)
                    st.session_state["chat_messages"].append(
                        {"role": "assistant", "content": ai_response}
                    )
                    st.rerun()



# ============================================================================
# SECTION: Display Components
# ============================================================================

def display_parsed_resume(parsed: Dict[str, Any]) -> None:
    """Show extracted skills, categories, and resume sections."""
    skills = parsed.get("skills", [])
    skill_count = parsed.get("skill_count", len(skills))
    sections = parsed.get("sections", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Skills Found", skill_count)
    c2.metric("Skills Section", "Yes" if sections.get("skills", "").strip() else "No")
    c3.metric("Experience Section", "Yes" if sections.get("experience", "").strip() else "No")

    st.markdown("#### Extracted Skills")
    if skills:
        tags = " ".join(f'<span class="skill-tag">{s}</span>' for s in skills)
        st.markdown(tags, unsafe_allow_html=True)
    else:
        st.info("No skills matched the taxonomy yet. Try a resume with clearer skill keywords.")

    grouped = group_skills_by_category(skills)
    if grouped:
        st.markdown("#### Skills by Category")
        for category, cat_skills in grouped.items():
            with st.expander(f"{category} ({len(cat_skills)})", expanded=len(grouped) <= 4):
                tags = " ".join(f'<span class="skill-tag">{s}</span>' for s in cat_skills)
                st.markdown(tags, unsafe_allow_html=True)

    st.markdown("#### Resume Sections")
    tab_names = [SECTION_LABELS.get(k, k.title()) for k in sections]
    if sections:
        tabs = st.tabs(tab_names)
        for tab, (key, content) in zip(tabs, sections.items()):
            with tab:
                if content.strip():
                    st.markdown(content)
                else:
                    st.caption(f"No {SECTION_LABELS.get(key, key)} content detected.")


def _fit_badge(fit_label: str) -> str:
    lf = fit_label.lower()
    if "high" in lf:
        return f'<span class="badge-high">🟢 {fit_label}</span>'
    if "medium" in lf:
        return f'<span class="badge-medium">🟡 {fit_label}</span>'
    return f'<span class="badge-low">🔴 {fit_label}</span>'


def display_job_card(
    rec: Dict[str, Any],
    card_key: str = "job",
    backend_online: bool = True,
) -> None:
    """Render a single ranked job recommendation card."""
    score = rec.get("match_score", 0)
    fit = rec.get("fit_label") or get_fit_label(score)
    title = rec.get("title", "Unknown Title")
    company = rec.get("company", "—")
    location = rec.get("location", "—")
    source = rec.get("source", "Unknown")
    reason = rec.get("reason", "No explanation available.")

    with st.container():
        st.markdown(
            f"""
            <div class='glass-card'>
                <div class='job-card-head'>
                    <div>
                        <div class='job-card-title'>{title}</div>
                        <div class='job-card-meta'>{company} · {location}</div>
                    </div>
                    <div>{_fit_badge(fit)}</div>
                </div>
                <span class='source-chip'>Source: {source}</span>
            """,
            unsafe_allow_html=True,
        )

        metric_a, metric_b, metric_c = st.columns(3)
        with metric_a:
            st.metric("Match Score", f"{score}%")
        with metric_b:
            skill_pct = int(rec.get("skill_match_score", 0) * 100)
            st.metric("Skill Match", f"{skill_pct}%")
        with metric_c:
            sem_pct = int(rec.get("semantic_similarity_score", 0) * 100)
            st.metric("Semantic Fit", f"{sem_pct}%")

        st.progress(min(score, 100) / 100)

        st.markdown(
            f"<div class='job-card-summary'>{reason}</div>",
            unsafe_allow_html=True,
        )

        matched = rec.get("matched_skills", [])
        missing = rec.get("missing_skills", [])

        m_col, x_col = st.columns(2)
        with m_col:
            st.markdown("**Matched skills**")
            if matched:
                tags = " ".join(f'<span class="skill-tag">{s}</span>' for s in matched[:6])
                if len(matched) > 6:
                    tags += f' <span class="skill-tag">+{len(matched)-6} more</span>'
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.caption("None detected")
        with x_col:
            st.markdown("**Missing skills**")
            if missing:
                tags = " ".join(
                    f'<span class="missing-tag">{s}</span>'
                    for s in missing[:6]
                )
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.caption("None - great coverage.")

        action_col1, action_col2 = st.columns(2)
        apply_url = rec.get("apply_url", "")
        with action_col1:
            if apply_url and apply_url != "#":
                st.link_button("🚀 Apply Now", apply_url, use_container_width=True)
            else:
                st.caption("Apply link not available.")
        with action_col2:
            if st.button("🔖 Save Job", key=f"save_job_{card_key}", use_container_width=True):
                if not backend_online:
                    st.error("Backend offline — cannot save jobs.")
                else:
                    result = save_job_bookmark(rec)
                    if result:
                        st.session_state["saved_jobs"] = fetch_saved_jobs()
                        st.success(result.get("message", "Job saved!"))
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")


def display_official_search_card(source: Dict[str, Any]) -> None:
    """Render an official Big Tech career search link card."""
    with st.container():
        st.markdown(
            f"""
            <div class='glass-card'>
                <div class='job-card-title'>{source.get('title', 'Official Careers Search')}</div>
                <div class='job-card-meta'>{source.get('company', '—')} · {source.get('location', 'Any location')} · {source.get('source', 'Official Search')}</div>
                <div class='job-card-summary'>{source.get('reason', 'Open the official careers page to browse current openings.')}</div>
            """,
            unsafe_allow_html=True,
        )
        search_url = source.get("search_url") or source.get("apply_url", "")
        if search_url:
            st.link_button(
                f"Open {source.get('company', 'Careers')} Search",
                search_url,
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def display_saved_job_card(saved: Dict[str, Any], backend_online: bool = True) -> None:
    """Render a bookmarked job with option to remove."""
    with st.container():
        st.markdown(
            f"""
            <div class='glass-card'>
                <div class='job-card-head'>
                    <div>
                        <div class='job-card-title'>{saved.get('title', 'Unknown Title')}</div>
                        <div class='job-card-meta'>{saved.get('company', '—')} · {saved.get('location', '—')}</div>
                    </div>
                    <div class='source-chip'>Saved {saved.get('created_at', '')[:10] or 'recently'}</div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Match Score", f"{saved.get('match_score', 0)}%")

        missing = saved.get("missing_skills", [])
        if missing:
            st.markdown("**Missing skills**")
            tags = " ".join(
                f'<span class="missing-tag">{s}</span>'
                for s in missing
            )
            st.markdown(tags, unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns(2)
        apply_url = saved.get("apply_url", "")
        with btn_col1:
            if apply_url:
                st.link_button("Apply", apply_url, use_container_width=True)
        with btn_col2:
            job_id = saved.get("id")
            if st.button("Remove", key=f"remove_saved_{job_id}", use_container_width=True):
                if not backend_online:
                    st.error("Backend offline — cannot remove saved jobs.")
                elif job_id and remove_saved_job(job_id):
                    st.session_state["saved_jobs"] = fetch_saved_jobs()
                    st.success("Removed from saved jobs.")
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def display_saved_jobs_section(backend_online: bool) -> None:
    """Show all bookmarked jobs from SQLite."""
    st.markdown(
        """
        <div class="dashboard-panel">
            <div class="panel-title">Saved Jobs</div>
            <div class="panel-subtitle">Jobs you bookmarked from recommendations, stored locally in SQLite with no login required.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not backend_online:
        st.warning("Start the backend to save and view bookmarked jobs.")
        return

    if st.button("Refresh saved jobs", key="refresh_saved_jobs"):
        st.session_state["saved_jobs"] = fetch_saved_jobs()

    saved_jobs = st.session_state.get("saved_jobs")
    if saved_jobs is None:
        saved_jobs = fetch_saved_jobs()
        st.session_state["saved_jobs"] = saved_jobs

    if not saved_jobs:
        st.info("No saved jobs yet. Click **🔖 Save Job** on a recommendation to bookmark it.")
        return

    st.write(f"**{len(saved_jobs)}** saved job(s)")
    for saved in saved_jobs:
        display_saved_job_card(saved, backend_online=backend_online)


def display_skill_gap_summary(recommendations: List[Dict[str, Any]]) -> None:
    """Show most common missing skills from top 10 recommendations."""
    top_jobs = recommendations[:10]
    if not top_jobs:
        st.info("No recommendations yet — run job matching first.")
        return

    missing_counter: Counter = Counter()
    for job in top_jobs:
        for skill in job.get("missing_skills", []):
            missing_counter[skill] += 1

    if not missing_counter:
        st.success("Your resume covers the skills required by your top job matches!")
        return

    st.markdown("#### Most Common Skill Gaps (top 10 jobs)")
    most_common = missing_counter.most_common(8)

    for skill, count in most_common:
        pct = int(count / len(top_jobs) * 100)
        col_name, col_bar = st.columns([2, 3])
        with col_name:
            st.markdown(f"`{skill}`")
        with col_bar:
            st.progress(pct / 100, text=f"{count}/{len(top_jobs)} jobs ({pct}%)")

    top_skill, top_count = most_common[0]
    st.markdown(
        f"**💡 Suggested focus:** Start learning **{top_skill}** — "
        f"it appears in {top_count} of your best-matching roles."
    )


def display_resume_suggestions(
    parsed: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
) -> None:
    """Rule-based resume improvement tips."""
    suggestions: List[str] = []
    raw_text = parsed.get("raw_text", "")
    resume_skills = {s.lower() for s in parsed.get("skills", [])}

    top_jobs = recommendations[:10]
    missing_counter: Counter = Counter()
    for job in top_jobs:
        for skill in job.get("missing_skills", []):
            missing_counter[skill] += 1

    threshold = max(2, len(top_jobs) // 3) if top_jobs else 2

    if missing_counter.get("Docker", 0) >= threshold:
        suggestions.append(
            "🐳 **Docker** appears in many target jobs — add a containerized project "
            "or mention deployment with Docker in your experience section."
        )
    if missing_counter.get("SQL", 0) >= threshold:
        suggestions.append(
            "🗄️ **SQL** is frequently required — add a project using databases "
            "(e.g. PostgreSQL queries, schema design, or analytics with SQL)."
        )
    cloud_missing = sum(
        missing_counter.get(skill, 0)
        for skill in ["AWS", "Azure", "Google Cloud"]
    )
    if cloud_missing >= threshold:
        suggestions.append(
            "☁️ **Cloud skills** are in demand — deploy one project on AWS, Azure, or GCP "
            "and link the live demo in your resume."
        )

    if "github.com" not in raw_text.lower():
        suggestions.append(
            "🔗 **No GitHub link detected** — add your GitHub profile URL near the top of your resume."
        )

    if not LINK_PATTERN.search(raw_text):
        suggestions.append(
            "📎 **No project/repo links found** — add GitHub/GitLab repo URLs or live demo links "
            "for each project."
        )

    has_ml = any(s in resume_skills for s in {k.lower() for k in ML_SKILLS})
    if has_ml and not DEPLOYMENT_TERMS.search(raw_text):
        suggestions.append(
            "🚀 **ML but no deployment** — build a small ML app "
            "with Streamlit or FastAPI and mention it on your resume."
        )

    if not suggestions:
        suggestions.append(
            "✅ Your resume looks well-aligned with your top matches. "
            "Keep projects updated and tailor keywords to each application."
        )

    st.markdown("#### Resume Improvement Suggestions")
    for tip in suggestions:
        st.markdown(f"- {tip}")


def display_job_market_insights(
    jobs: List[Dict[str, Any]],
    recommendations: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Render Job Market Insights dashboard."""
    if not jobs:
        st.info("Load or fetch jobs to see market insights.")
        return

    insights = compute_market_insights(jobs, recommendations)

    st.markdown(
        """
        <div class="dashboard-panel">
            <div class="panel-title">Job Market Insights</div>
            <div class="panel-subtitle">Understand what employers are asking for and which skills you should learn next.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Jobs analyzed", insights["job_count"])
    m2.metric(
        "Average match score",
        f"{insights['average_match_score']}%" if insights["average_match_score"] is not None else "—",
    )
    m3.metric("Recommendations", insights["recommendation_count"])

    # Score distribution
    recs = recommendations or []
    scores = [r.get("match_score", 0) for r in recs]
    if scores:
        st.markdown("#### Match Score Distribution")
        fig = _plot_score_distribution(scores)
        st.pyplot(fig)
        plt.close(fig)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Top 10 Most Demanded Skills")
        if insights["demanded_skills"]:
            df_demanded = pd.DataFrame(insights["demanded_skills"], columns=["Skill", "Job Count"])
            st.dataframe(df_demanded, use_container_width=True, hide_index=True)
            fig = _plot_horizontal_bar(
                "Most Demanded Skills Across Jobs",
                [s for s, _ in insights["demanded_skills"]],
                [c for _, c in insights["demanded_skills"]],
                "Number of job postings mentioning skill",
                "#7c3aed",
            )
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("#### Jobs by Source")
        if insights["jobs_by_source"]:
            df_source = pd.DataFrame(insights["jobs_by_source"], columns=["Source", "Jobs"])
            st.dataframe(df_source, use_container_width=True, hide_index=True)
            fig = _plot_source_chart(
                "Job Postings by Source",
                [s for s, _ in insights["jobs_by_source"]],
                [c for _, c in insights["jobs_by_source"]],
            )
            st.pyplot(fig)
            plt.close(fig)

    with col_r:
        st.markdown("#### Top 10 Missing Skills (Your Resume Gaps)")
        if insights["missing_skills"]:
            df_missing = pd.DataFrame(insights["missing_skills"], columns=["Skill", "Times Missing"])
            st.dataframe(df_missing, use_container_width=True, hide_index=True)
            fig = _plot_horizontal_bar(
                "Skills Missing From Your Resume vs. Target Jobs",
                [s for s, _ in insights["missing_skills"]],
                [c for _, c in insights["missing_skills"]],
                "Number of jobs where skill is missing",
                "#ef4444",
            )
            st.pyplot(fig)
            plt.close(fig)
            top_skill = insights["missing_skills"][0][0]
            st.success(f"Learning **{top_skill}** next could improve your fit for multiple roles.")
        elif recommendations:
            st.success("No major skill gaps detected in your current recommendations.")

        st.markdown("#### Jobs by Location")
        if insights["jobs_by_location"]:
            df_location = pd.DataFrame(insights["jobs_by_location"], columns=["Location", "Jobs"])
            st.dataframe(df_location, use_container_width=True, hide_index=True)
            fig = _plot_horizontal_bar(
                "Job Postings by Location",
                [loc for loc, _ in insights["jobs_by_location"]],
                [c for _, c in insights["jobs_by_location"]],
                "Number of jobs",
                "#059669",
            )
            st.pyplot(fig)
            plt.close(fig)


# ============================================================================
# SECTION: ML Control Center
# ============================================================================

def display_ml_control_center(backend_online: bool) -> None:
    """Full ML model dashboard with metrics, features, and management controls."""

    st.markdown(
        '<p class="hero-title">🤖 ML Control Center</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="hero-subtitle">Monitor, retrain, and download the Random Forest fit classifier powering your recommendations.</p>',
        unsafe_allow_html=True,
    )

    if not backend_online:
        st.markdown(
            '<span class="status-offline">⛔ Backend Offline</span>',
            unsafe_allow_html=True,
        )
        st.warning("Start the FastAPI backend to access ML model controls.")
        return

    st.markdown('<span class="status-online">✅ Backend Connected</span>', unsafe_allow_html=True)
    st.write("")

    # --- Fetch model status ---
    with st.spinner("Loading model status..."):
        status_data = get_ml_model_status()

    if status_data is None:
        st.error("Could not fetch ML model status from backend.")
        return

    model_loaded = status_data.get("model_loaded", False)
    metrics = status_data.get("metrics") or {}
    feature_names = status_data.get("feature_names", [])
    feature_importances = status_data.get("feature_importances", {})

    # --- Model status indicator ---
    if model_loaded:
        st.markdown('<span class="status-online">🟢 Model Loaded — Random Forest Classifier</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-offline">🔴 Model Not Loaded — Heuristic Fallback Active</span>', unsafe_allow_html=True)

    st.write("")

    # --- Metrics cards ---
    if metrics:
        st.markdown('<p class="section-header">📈 Model Performance Metrics</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, (label, key, fmt) in zip(
            [c1, c2, c3, c4],
            [
                ("Accuracy", "accuracy", ".1%"),
                ("Precision", "precision", ".1%"),
                ("Recall", "recall", ".1%"),
                ("F1 Score", "f1_score", ".1%"),
            ],
        ):
            val = metrics.get(key, 0)
            formatted = format(val, fmt) if isinstance(val, float) else str(val)
            with col:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{formatted}</div>
                        <div class='metric-label'>{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.write("")
        train_n = metrics.get("train_samples", "?")
        test_n = metrics.get("test_samples", "?")
        st.caption(f"Trained on **{train_n}** samples · Evaluated on **{test_n}** test samples")

        # --- Classification report ---
        with st.expander("📋 Full Classification Report", expanded=False):
            report_text = metrics.get("classification_report", "")
            if report_text:
                st.code(report_text, language=None)
            else:
                st.caption("No classification report available.")

    # --- Feature Importances ---
    if feature_importances:
        st.markdown('<p class="section-header">🔬 Feature Importances</p>', unsafe_allow_html=True)
        st.caption("How much each input feature contributes to the Random Forest classifier's predictions.")

        fi_col, table_col = st.columns([3, 2])
        with fi_col:
            fig = _plot_feature_importance(feature_importances)
            st.pyplot(fig)
            plt.close(fig)

        with table_col:
            sorted_fi = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            df_fi = pd.DataFrame(sorted_fi, columns=["Feature", "Importance"])
            df_fi["Feature"] = df_fi["Feature"].str.replace("_", " ").str.title()
            df_fi["Importance"] = df_fi["Importance"].map("{:.4f}".format)
            st.dataframe(df_fi, use_container_width=True, hide_index=True)

    st.write("")

    # --- Model Management Controls ---
    st.markdown('<p class="section-header">⚙️ Model Management</p>', unsafe_allow_html=True)

    ctrl_col1, ctrl_col2 = st.columns(2)

    with ctrl_col1:
        st.markdown("**🔄 Retrain Model**")
        st.caption(
            "Regenerates the synthetic training dataset from sample jobs, retrains the Random Forest "
            "classifier, and saves the updated model."
        )
        if st.button("🔄 Retrain Model Now", use_container_width=True, type="primary"):
            with st.spinner("Training Random Forest classifier... (this may take ~30s)"):
                result = trigger_model_retrain()
            if result and "error" not in result:
                new_metrics = result.get("metrics", {})
                acc = new_metrics.get("accuracy", 0)
                f1 = new_metrics.get("f1_score", 0)
                st.success(
                    f"✅ Model retrained! Accuracy: **{acc:.1%}** · F1: **{f1:.1%}**"
                )
                st.rerun()
            else:
                st.error(f"Retraining failed: {result.get('error', 'Unknown error')}")

    with ctrl_col2:
        st.markdown("**⬇️ Download Model**")
        st.caption(
            "Download the serialized `fit_classifier.pkl` file for offline use, "
            "deployment inspection, or integration into other pipelines."
        )
        try:
            pkl_response = requests.get(ML_DOWNLOAD_ENDPOINT, timeout=15)
            if pkl_response.status_code == 200:
                st.download_button(
                    label="⬇️ Download fit_classifier.pkl",
                    data=pkl_response.content,
                    file_name="fit_classifier.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            else:
                st.warning("Model file not available for download. Train the model first.")
        except Exception:
            st.warning("Could not fetch model file from backend.")

    # --- Architecture info ---
    st.write("")
    with st.expander("ℹ️ Model Architecture & Feature Details", expanded=False):
        st.markdown("""
**Algorithm:** Random Forest Classifier (scikit-learn)

| Parameter | Value |
|-----------|-------|
| Estimators | 120 trees |
| Max depth | 8 |
| Class weights | Balanced |
| Output classes | Low Fit · Medium Fit · High Fit |

**Input Features:**

| Feature | Description |
|---------|-------------|
| `skill_match_score` | Fraction of required job skills present in resume (0–1) |
| `semantic_similarity_score` | Cosine similarity of sentence embeddings (0–1) |
| `number_of_matched_skills` | Raw count of matched skills |
| `number_of_missing_skills` | Raw count of missing skills |
| `has_project_keywords` | Binary — resume mentions projects/GitHub/deployed |
| `has_internship_keywords` | Binary — resume mentions intern/student/entry-level |

**Training Data:** Synthetic resume-job pair dataset generated from sample jobs and
annotated using heuristic composite scoring (45% skill overlap + 35% semantic + bonus terms).
        """)


# ============================================================================
# SECTION: Professional UI component overrides
# ============================================================================

def _fit_badge(fit_label: str) -> str:
    lf = fit_label.lower()
    if "high" in lf or "excellent" in lf:
        return f'<span class="badge-high">{fit_label}</span>'
    if "medium" in lf or "good" in lf or "moderate" in lf:
        return f'<span class="badge-medium">{fit_label}</span>'
    return f'<span class="badge-low">{fit_label}</span>'


def display_job_card(
    rec: Dict[str, Any],
    card_key: str = "job",
    backend_online: bool = True,
) -> None:
    """Render a professional ranked job card."""
    score = int(rec.get("match_score", 0) or 0)
    fit = rec.get("fit_label") or get_fit_label(score)
    skill_pct = int((rec.get("skill_match_score", 0) or 0) * 100)
    sem_pct = int((rec.get("semantic_similarity_score", 0) or 0) * 100)
    apply_url = rec.get("apply_url", "")

    with st.container(border=True):
        title_col, score_col = st.columns([4, 1])
        with title_col:
            st.markdown(f"**{rec.get('title', 'Unknown Title')}**")
            st.caption(
                f"{rec.get('company', '-')} | {rec.get('location', '-')} | "
                f"{rec.get('source', 'Unknown')}"
            )
        with score_col:
            st.markdown(_fit_badge(fit), unsafe_allow_html=True)
            st.metric("Match", f"{score}%")

        st.progress(min(max(score, 0), 100) / 100)
        st.caption(f"Skill alignment {skill_pct}% | Profile similarity {sem_pct}%")

        matched = rec.get("matched_skills", []) or []
        missing = rec.get("missing_skills", []) or []
        matched_col, missing_col = st.columns(2)
        with matched_col:
            st.markdown("**Matched skills**")
            if matched:
                tags = " ".join(f'<span class="skill-tag">{s}</span>' for s in matched[:7])
                if len(matched) > 7:
                    tags += f' <span class="skill-tag">+{len(matched) - 7} more</span>'
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.caption("No direct skill matches detected")
        with missing_col:
            st.markdown("**Skills to build**")
            if missing:
                tags = " ".join(f'<span class="missing-tag">{s}</span>' for s in missing[:7])
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.caption("No major gaps detected")

        st.caption(rec.get("reason", "Matched based on resume and job profile similarity."))

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if apply_url and apply_url != "#":
                st.link_button("View role", apply_url, use_container_width=True)
            else:
                st.caption("Apply link not available")
        with action_col2:
            if st.button("Save job", key=f"save_job_{card_key}", use_container_width=True):
                if not backend_online:
                    st.error("Backend offline; cannot save jobs.")
                else:
                    result = save_job_bookmark(rec)
                    if result:
                        st.session_state["saved_jobs"] = fetch_saved_jobs()
                        st.success(result.get("message", "Job saved."))
        st.write("")


def display_official_search_card(source: Dict[str, Any]) -> None:
    """Render an official career search link."""
    with st.container():
        st.markdown(f"**{source.get('title', 'Official careers search')}**")
        st.caption(
            f"{source.get('company', '-')} | {source.get('location', 'Any location')} | "
            f"{source.get('source', 'Official Search')}"
        )
        st.caption(source.get("reason", "Open the official careers page to browse current openings."))
        search_url = source.get("search_url") or source.get("apply_url", "")
        if search_url:
            st.link_button(
                f"Open {source.get('company', 'careers')} search",
                search_url,
                use_container_width=False,
            )
        st.divider()


def display_saved_job_card(saved: Dict[str, Any], backend_online: bool = True) -> None:
    """Render a saved job row."""
    with st.container():
        row_l, row_r = st.columns([3, 1])
        with row_l:
            st.markdown(f"**{saved.get('title', 'Unknown Title')}**")
            st.caption(
                f"{saved.get('company', '-')} | {saved.get('location', '-')} | "
                f"Saved {saved.get('created_at', '')[:10] or 'recently'}"
            )
        with row_r:
            st.metric("Match", f"{saved.get('match_score', 0)}%")

        missing = saved.get("missing_skills", []) or []
        if missing:
            tags = " ".join(f'<span class="missing-tag">{s}</span>' for s in missing[:8])
            st.markdown(tags, unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns(2)
        apply_url = saved.get("apply_url", "")
        with btn_col1:
            if apply_url:
                st.link_button("View role", apply_url, use_container_width=True)
        with btn_col2:
            job_id = saved.get("id")
            if st.button("Remove", key=f"remove_saved_{job_id}", use_container_width=True):
                if not backend_online:
                    st.error("Backend offline; cannot remove saved jobs.")
                elif job_id and remove_saved_job(job_id):
                    st.session_state["saved_jobs"] = fetch_saved_jobs()
                    st.success("Removed from saved jobs.")
                    st.rerun()
        st.divider()


def display_saved_jobs_section(backend_online: bool) -> None:
    """Show bookmarked jobs."""
    st.markdown("### Saved Jobs")
    st.caption("Roles bookmarked from your recommendations.")

    if not backend_online:
        st.warning("Start the backend to save and view bookmarked jobs.")
        return

    if st.button("Refresh saved jobs", key="refresh_saved_jobs"):
        st.session_state["saved_jobs"] = fetch_saved_jobs()

    saved_jobs = st.session_state.get("saved_jobs")
    if saved_jobs is None:
        saved_jobs = fetch_saved_jobs()
        st.session_state["saved_jobs"] = saved_jobs

    if not saved_jobs:
        st.info("No saved jobs yet. Save roles from the recommendations list to compare them here.")
        return

    st.write(f"**{len(saved_jobs)}** saved role(s)")
    for saved in saved_jobs:
        display_saved_job_card(saved, backend_online=backend_online)


def render_chatbot_launcher() -> None:
    """Render a floating chatbot popover in the bottom-right corner."""
    st.markdown(
        """
        <style>
            .skillhire-chat-launcher-shell {
                position: fixed;
                right: 24px;
                bottom: 24px;
                z-index: 9999;
                font-family: Inter, system-ui, sans-serif;
            }
            .skillhire-chat-launcher-shell * {
                box-sizing: border-box;
            }
            .skillhire-chat-launcher-toggle {
                position: absolute;
                opacity: 0;
                pointer-events: none;
            }
            .skillhire-chat-launcher-bubble {
                display: inline-flex;
                align-items: center;
                gap: 0.7rem;
                cursor: pointer;
                user-select: none;
            }
            .skillhire-chat-launcher-bubble-text,
            .skillhire-chat-launcher-panel {
                background: rgba(255, 255, 255, 0.97);
                border: 1px solid rgba(101, 146, 135, 0.22);
                box-shadow: 0 14px 32px rgba(15, 23, 42, 0.16);
                backdrop-filter: blur(14px);
            }
            .skillhire-chat-launcher-bubble-text {
                border-radius: 999px;
                padding: 0.7rem 1rem;
                color: #0f172a;
                font-size: 0.85rem;
                font-weight: 700;
                line-height: 1.1;
                white-space: nowrap;
            }
            .skillhire-chat-launcher-bubble-text span {
                display: block;
                font-size: 0.7rem;
                font-weight: 600;
                color: #64748b;
                margin-top: 0.15rem;
            }
            .skillhire-chat-launcher-icon {
                width: 58px;
                height: 58px;
                border-radius: 50%;
                background: radial-gradient(circle at 30% 30%, #e6f2dd, #88bda4 55%, #659287 100%);
                box-shadow: 0 16px 30px rgba(101, 146, 135, 0.28);
                display: grid;
                place-items: center;
                border: 3px solid rgba(255,255,255,0.95);
                position: relative;
            }
            .skillhire-chat-launcher-icon::before {
                content: "";
                width: 30px;
                height: 22px;
                border-radius: 12px;
                background: linear-gradient(180deg, #1f2937, #111827);
                position: absolute;
                top: 13px;
                left: 50%;
                transform: translateX(-50%);
                box-shadow: inset 0 -3px 0 rgba(255,255,255,0.1);
            }
            .skillhire-chat-launcher-icon::after {
                content: "";
                width: 8px;
                height: 4px;
                border-radius: 999px;
                background: #e2e8f0;
                position: absolute;
                top: 20px;
                left: 50%;
                transform: translateX(-50%);
                box-shadow: -6px 0 0 #e2e8f0, 6px 0 0 #e2e8f0;
            }
            .skillhire-chat-launcher-body {
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: rgba(236, 253, 245, 0.92);
                position: absolute;
                bottom: 8px;
                left: 50%;
                transform: translateX(-50%);
                box-shadow: 0 0 0 8px rgba(255,255,255,0.12);
            }
            .skillhire-chat-launcher-panel {
                position: absolute;
                right: 0;
                bottom: 72px;
                width: 320px;
                border-radius: 18px;
                padding: 0.95rem 0.95rem 0.9rem 0.95rem;
                color: #0f172a;
                opacity: 0;
                transform: translateY(10px) scale(0.98);
                pointer-events: none;
                transition: opacity 0.18s ease, transform 0.18s ease;
            }
            .skillhire-chat-launcher-toggle:checked ~ .skillhire-chat-launcher-panel {
                opacity: 1;
                transform: translateY(0) scale(1);
                pointer-events: auto;
            }
            .skillhire-chat-launcher-panel-title {
                font-size: 0.98rem;
                font-weight: 800;
                margin-bottom: 0.2rem;
            }
            .skillhire-chat-launcher-panel-copy {
                font-size: 0.82rem;
                line-height: 1.45;
                color: #475569;
                margin-bottom: 0.7rem;
            }
            .skillhire-chat-launcher-panel-chips {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-bottom: 0.75rem;
            }
            .skillhire-chat-launcher-panel-chip {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.42rem 0.65rem;
                border-radius: 999px;
                border: 1px solid #b1d3b9;
                background: #e6f2dd;
                color: #4f7f74;
                font-size: 0.75rem;
                font-weight: 700;
                text-decoration: none;
                white-space: nowrap;
            }
            .skillhire-chat-launcher-panel-chip:hover {
                background: #dcead2;
            }
            .skillhire-chat-launcher-panel-actions {
                display: flex;
                gap: 0.55rem;
                flex-wrap: wrap;
            }
            .skillhire-chat-launcher-panel-link,
            .skillhire-chat-launcher-panel-close {
                border: 0;
                border-radius: 999px;
                padding: 0.55rem 0.85rem;
                font-size: 0.8rem;
                font-weight: 700;
                cursor: pointer;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }
            .skillhire-chat-launcher-panel-link {
                background: #659287;
                color: #ffffff;
            }
            .skillhire-chat-launcher-panel-close {
                background: #e6f2dd;
                color: #4f7f74;
            }
            @media (max-width: 900px) {
                .skillhire-chat-launcher-shell {
                    right: 14px;
                    bottom: 14px;
                }
                .skillhire-chat-launcher-bubble-text {
                    display: none;
                }
                .skillhire-chat-launcher-panel {
                    width: min(88vw, 300px);
                    right: 0;
                }
            }
        </style>
        <div class="skillhire-chat-launcher-shell">
            <input class="skillhire-chat-launcher-toggle" id="skillhire-chat-launcher-toggle" type="checkbox" />
            <label class="skillhire-chat-launcher-bubble" for="skillhire-chat-launcher-toggle" aria-label="Open Career Advisor">
                <div class="skillhire-chat-launcher-bubble-text">Ask me anything<span>Career Advisor</span></div>
                <div class="skillhire-chat-launcher-icon">
                    <div class="skillhire-chat-launcher-body"></div>
                </div>
            </label>
            <div class="skillhire-chat-launcher-panel">
                <div class="skillhire-chat-launcher-panel-title">Career Advisor</div>
                <div class="skillhire-chat-launcher-panel-copy">Ask me anything about jobs, skills, resume improvements, or what to learn next.</div>
                <div class="skillhire-chat-launcher-panel-chips">
                    <a class="skillhire-chat-launcher-panel-chip" href="#career-advisor-panel">Review my resume</a>
                    <a class="skillhire-chat-launcher-panel-chip" href="#career-advisor-panel">Find better jobs</a>
                    <a class="skillhire-chat-launcher-panel-chip" href="#career-advisor-panel">Skill gaps</a>
                </div>
                <div class="skillhire-chat-launcher-panel-actions">
                    <a class="skillhire-chat-launcher-panel-link" href="#career-advisor-panel">Open advisor</a>
                    <label class="skillhire-chat-launcher-panel-close" for="skillhire-chat-launcher-toggle">Close</label>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_auth_topbar() -> None:
    """Render the account sign-in entry point in the top-right header."""
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None

    brand_col, auth_col = st.columns([0.72, 0.28], vertical_alignment="center")

    with brand_col:
        st.markdown(
            """
            <div class="auth-product-row">
                <div class="auth-product-mark">SH</div>
                <div>
                    <div class="auth-product-name">SkillHire AI</div>
                    <div class="auth-product-subtitle">Resume matching workspace</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with auth_col:
        auth_user = st.session_state.get("auth_user")
        if auth_user:
            with st.popover(
                auth_user.get("name") or "Account",
                type="secondary",
                icon=":material/verified_user:",
                use_container_width=True,
            ):
                st.markdown("**Signed in**")
                if auth_user.get("email"):
                    st.caption(auth_user["email"])
                st.caption("Your account is ready for saved roles, resume history, and skill progress.")
                if st.button("Sign out", key="auth_sign_out", use_container_width=True):
                    st.session_state["auth_user"] = None
                    st.session_state.pop("firebase_id_token", None)
                    st.rerun()
            return

        with st.popover(
            "Sign in",
            type="primary",
            icon=":material/account_circle:",
            use_container_width=True,
        ):
            st.markdown(
                """
                <div class="auth-popover-title">Track your career progress</div>
                <div class="auth-popover-copy">
                    Sign in with Google to keep your saved roles, resume analyses, and skill progress tied to your account.
                </div>
                <div class="auth-benefit-row">
                    <div class="auth-benefit-dot"></div>
                    <div>
                        <div class="auth-benefit-title">Resume history</div>
                        <div class="auth-benefit-copy">Compare older uploads with your latest resume improvements.</div>
                    </div>
                </div>
                <div class="auth-benefit-row">
                    <div class="auth-benefit-dot"></div>
                    <div>
                        <div class="auth-benefit-title">Saved jobs</div>
                        <div class="auth-benefit-copy">Keep shortlisted jobs available across devices and sessions.</div>
                    </div>
                </div>
                <div class="auth-benefit-row">
                    <div class="auth-benefit-dot"></div>
                    <div>
                        <div class="auth-benefit-title">Skill roadmap</div>
                        <div class="auth-benefit-copy">Track missing skills and learning progress from one account.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state.get("auth_error"):
                st.error(st.session_state["auth_error"])

            if FIREBASE_CONFIGURED:
                st.markdown(
                    """
                    <div class="firebase-status">
                        Firebase config detected. Google sign-in is active for this workspace.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                render_firebase_google_button()
            else:
                st.markdown(
                    """
                    <div class="firebase-status">
                        Firebase setup needed. Add your Firebase web config to the environment, then enable Google provider in Firebase Authentication.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    "Continue with Google",
                    key="firebase_google_missing",
                    disabled=True,
                    use_container_width=True,
                    help="Add Firebase config before enabling Google sign-in.",
                )


def decode_auth_payload(payload: str) -> Dict[str, Any]:
    """Decode a base64url JSON payload returned from the Firebase auth widget."""
    padded = payload + "=" * (-len(payload) % 4)
    decoded = base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8")
    return json.loads(decoded)


def verify_firebase_id_token(id_token: str) -> Optional[Dict[str, Any]]:
    """Verify a Firebase ID token using Firebase Auth's public REST API."""
    if not FIREBASE_API_KEY or not id_token:
        return None

    try:
        response = requests.post(
            f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={FIREBASE_API_KEY}",
            json={"idToken": id_token},
            timeout=12,
        )
        if response.status_code != 200:
            return None
        users = response.json().get("users", [])
        return users[0] if users else None
    except requests.RequestException:
        return None


def consume_firebase_auth_payload() -> None:
    """Consume Firebase auth redirect data and hydrate Streamlit session state."""
    payload = st.query_params.get("firebase_auth")
    error = st.query_params.get("firebase_auth_error")

    if error:
        st.session_state["auth_error"] = error
        del st.query_params["firebase_auth_error"]
        st.rerun()

    if not payload:
        return

    try:
        auth_payload = decode_auth_payload(payload)
        verified_user = verify_firebase_id_token(auth_payload.get("idToken", ""))
        if not verified_user:
            st.session_state["auth_error"] = "Firebase could not verify this Google sign-in. Try again."
        else:
            st.session_state["auth_user"] = {
                "uid": verified_user.get("localId"),
                "name": verified_user.get("displayName") or auth_payload.get("displayName") or "User",
                "email": verified_user.get("email") or auth_payload.get("email"),
                "photo_url": verified_user.get("photoUrl") or auth_payload.get("photoURL"),
                "provider": "firebase-google",
            }
            st.session_state["firebase_id_token"] = auth_payload.get("idToken")
            st.session_state.pop("auth_error", None)
    except (ValueError, json.JSONDecodeError, TypeError):
        st.session_state["auth_error"] = "Could not read the Firebase sign-in response. Try again."
    finally:
        del st.query_params["firebase_auth"]
        st.rerun()


def render_firebase_google_button() -> None:
    """Render a link to the standalone Firebase Google sign-in page."""
    login_url = f"{AUTH_BACKEND_URL.rstrip('/')}/firebase-login"
    st.link_button(
        "Continue with Google",
        login_url,
        type="primary",
        use_container_width=True,
        help="Opens Firebase Google sign-in on a normal localhost page.",
    )
    st.caption("Opens a Firebase sign-in page on localhost, then returns here.")


def render_top_career_links_section(backend_online: bool, compact: bool = False) -> None:
    """Render a top quick-links section for jobs, internships, and major career sites."""
    st.markdown(
        """
        <div class="dashboard-panel">
            <div class="panel-title">Jobs & Internships</div>
            <div class="panel-subtitle">Quick access to full-time jobs, internship roles, and official career sites.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    google_jobs = "https://careers.google.com/jobs/results/"
    google_internships = "https://careers.google.com/jobs/results/?q=intern"
    microsoft_jobs = "https://jobs.careers.microsoft.com/global/en/search"
    microsoft_internships = "https://jobs.careers.microsoft.com/global/en/search?keywords=intern"
    amazon_jobs = "https://www.amazon.jobs/en/"
    amazon_internships = "https://www.amazon.jobs/en/search?base_query=internship"

    if compact:
        st.markdown(
            f"""
            <style>
                .career-mini-grid {{
                    display: grid;
                    grid-template-columns: repeat(3, minmax(0, 1fr));
                    gap: 0.55rem;
                    margin-top: 0.65rem;
                    align-items: stretch;
                }}
                .career-mini-card {{
                    min-width: 0;
                    min-height: 220px;
                    background: #ffffff;
                    border: 1px solid #d9dee7;
                    border-radius: 8px;
                    padding: 0.72rem;
                    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
                    display: flex;
                    flex-direction: column;
                }}
                .career-mini-card__title {{
                    color: #172033;
                    font-size: 0.82rem;
                    font-weight: 800;
                    line-height: 1.2;
                    margin-bottom: 0.25rem;
                }}
                .career-mini-card__copy {{
                    color: #667085;
                    font-size: 0.68rem;
                    line-height: 1.3;
                    min-height: 2.6rem;
                    margin-bottom: 0.55rem;
                    flex: 1;
                }}
                .career-mini-card__actions {{
                    display: grid;
                    grid-template-columns: 1fr;
                    gap: 0.35rem;
                    margin-top: auto;
                }}
                .career-mini-card__action {{
                    display: block;
                    text-align: center;
                    text-decoration: none;
                    color: #172033 !important;
                    background: #f8fafc;
                    border: 1px solid #d9dee7;
                    border-radius: 6px;
                    padding: 0.42rem 0.35rem;
                    font-size: 0.68rem;
                    font-weight: 750;
                    line-height: 1.15;
                    white-space: normal;
                }}
                .career-mini-card__action:hover {{
                    background: #eff6ff;
                    border-color: #bfdbfe;
                    color: #1d4ed8 !important;
                }}
                @media (max-width: 900px) {{
                    .career-mini-grid {{
                        grid-template-columns: 1fr;
                    }}
                    .career-mini-card {{
                        min-height: 150px;
                    }}
                }}
            </style>
            <div class="career-mini-grid">
                <div class="career-mini-card">
                    <div class="career-mini-card__title">Google</div>
                    <div class="career-mini-card__copy">Official jobs and internships.</div>
                    <div class="career-mini-card__actions">
                        <a class="career-mini-card__action" href="{google_jobs}" target="_blank" rel="noopener noreferrer">Jobs</a>
                        <a class="career-mini-card__action" href="{google_internships}" target="_blank" rel="noopener noreferrer">Internships</a>
                    </div>
                </div>
                <div class="career-mini-card">
                    <div class="career-mini-card__title">Microsoft</div>
                    <div class="career-mini-card__copy">Roles and internship openings.</div>
                    <div class="career-mini-card__actions">
                        <a class="career-mini-card__action" href="{microsoft_jobs}" target="_blank" rel="noopener noreferrer">Jobs</a>
                        <a class="career-mini-card__action" href="{microsoft_internships}" target="_blank" rel="noopener noreferrer">Internships</a>
                    </div>
                </div>
                <div class="career-mini-card">
                    <div class="career-mini-card__title">Amazon</div>
                    <div class="career-mini-card__copy">Search jobs and internship roles.</div>
                    <div class="career-mini-card__actions">
                        <a class="career-mini-card__action" href="{amazon_jobs}" target="_blank" rel="noopener noreferrer">Jobs</a>
                        <a class="career-mini-card__action" href="{amazon_internships}" target="_blank" rel="noopener noreferrer">Internships</a>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    else:
        link_col1, link_col2, link_col3 = st.columns(3, gap="medium")

    with link_col1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="job-card-title">Google Careers</div>
                <div class="job-card-meta">Browse Google jobs and internships from the official careers site.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button("Open Google Jobs", google_jobs, use_container_width=True)
        st.link_button("Open Google Internships", google_internships, use_container_width=True)

    with link_col2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="job-card-title">Microsoft Careers</div>
                <div class="job-card-meta">Find Microsoft roles and internship opportunities directly.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button("Open Microsoft Jobs", microsoft_jobs, use_container_width=True)
        st.link_button("Open Microsoft Internships", microsoft_internships, use_container_width=True)

    with link_col3:
        st.markdown(
            """
            <div class="glass-card">
                <div class="job-card-title">Amazon Jobs</div>
                <div class="job-card-meta">Search Amazon jobs and internship roles on the official site.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button("Open Amazon Jobs", amazon_jobs, use_container_width=True)
        st.link_button("Open Amazon Internships", amazon_internships, use_container_width=True)

def display_skill_gap_summary(recommendations: List[Dict[str, Any]]) -> None:
    """Show common missing skills from top recommendations."""
    top_jobs = recommendations[:10]
    if not top_jobs:
        st.info("Run matching to generate skill gap analysis.")
        return

    missing_counter: Counter = Counter()
    for job in top_jobs:
        for skill in job.get("missing_skills", []):
            missing_counter[skill] += 1

    if not missing_counter:
        st.success("Your resume covers the core skills for the top matches.")
        return

    st.markdown("#### Common skill gaps")
    most_common = missing_counter.most_common(8)
    for skill, count in most_common:
        pct = int(count / len(top_jobs) * 100)
        col_name, col_bar = st.columns([2, 3])
        with col_name:
            st.markdown(f"`{skill}`")
        with col_bar:
            st.progress(pct / 100, text=f"{count}/{len(top_jobs)} roles ({pct}%)")

    top_skill, top_count = most_common[0]
    st.markdown(
        f"**Recommended focus:** Build evidence for **{top_skill}**; "
        f"it appears in {top_count} of your strongest role matches."
    )


def display_resume_suggestions(
    parsed: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
) -> None:
    """Rule-based resume improvement tips."""
    suggestions: List[str] = []
    raw_text = parsed.get("raw_text", "")

    top_jobs = recommendations[:10]
    missing_counter: Counter = Counter()
    for job in top_jobs:
        for skill in job.get("missing_skills", []):
            missing_counter[skill] += 1

    threshold = max(2, len(top_jobs) // 3) if top_jobs else 2

    if missing_counter.get("Docker", 0) >= threshold:
        suggestions.append("Add a deployment or containerization project that clearly mentions Docker.")
    if missing_counter.get("SQL", 0) >= threshold:
        suggestions.append("Add database experience with SQL, schema design, reporting, or PostgreSQL queries.")
    if sum(missing_counter.get(skill, 0) for skill in ["AWS", "Azure", "Google Cloud"]) >= threshold:
        suggestions.append("Show cloud exposure by deploying one project on AWS, Azure, or Google Cloud.")
    if "github.com" not in raw_text.lower():
        suggestions.append("Add your GitHub profile near the top of the resume.")
    if not LINK_PATTERN.search(raw_text):
        suggestions.append("Add project, repository, or live demo links for stronger evidence.")

    if not suggestions:
        suggestions.append("Your resume is aligned with the current top matches. Tailor keywords before applying.")

    st.markdown("#### Resume improvement actions")
    for tip in suggestions:
        st.markdown(f"- {tip}")


def display_job_market_insights(
    jobs: List[Dict[str, Any]],
    recommendations: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Render job market insights."""
    if not jobs:
        st.info("Load or fetch jobs to see market insights.")
        return

    insights = compute_market_insights(jobs, recommendations)

    st.markdown("### Job Market Insights")
    st.caption("Demand signals from the roles currently available in your database.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Jobs analyzed", insights["job_count"])
    m2.metric(
        "Average match",
        f"{insights['average_match_score']}%" if insights["average_match_score"] is not None else "-",
    )
    m3.metric("Recommendations", insights["recommendation_count"])

    scores = [r.get("match_score", 0) for r in recommendations or []]
    if scores:
        st.markdown("#### Match score distribution")
        fig = _plot_score_distribution(scores)
        st.pyplot(fig)
        plt.close(fig)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Most requested skills")
        if insights["demanded_skills"]:
            df_demanded = pd.DataFrame(insights["demanded_skills"], columns=["Skill", "Job Count"])
            st.dataframe(df_demanded, use_container_width=True, hide_index=True)
        st.markdown("#### Jobs by source")
        if insights["jobs_by_source"]:
            df_source = pd.DataFrame(insights["jobs_by_source"], columns=["Source", "Jobs"])
            st.dataframe(df_source, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("#### Missing skills in top matches")
        if insights["missing_skills"]:
            df_missing = pd.DataFrame(insights["missing_skills"], columns=["Skill", "Times Missing"])
            st.dataframe(df_missing, use_container_width=True, hide_index=True)
        elif recommendations:
            st.success("No major skill gaps detected in your current recommendations.")
        st.markdown("#### Jobs by location")
        if insights["jobs_by_location"]:
            df_location = pd.DataFrame(insights["jobs_by_location"], columns=["Location", "Jobs"])
            st.dataframe(df_location, use_container_width=True, hide_index=True)


def display_ml_control_center(backend_online: bool) -> None:
    """Professional ML model dashboard."""
    st.markdown('<div class="app-title">Model Operations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Monitor and retrain the classifier used to label fit quality.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if not backend_online:
        st.markdown('<span class="status-offline">Backend offline</span>', unsafe_allow_html=True)
        st.warning("Start the FastAPI backend to access model controls.")
        return

    st.markdown('<span class="status-online">Backend connected</span>', unsafe_allow_html=True)
    st.write("")

    with st.spinner("Loading model status..."):
        status_data = get_ml_model_status()

    if status_data is None:
        st.error("Could not fetch model status from backend.")
        return

    model_loaded = status_data.get("model_loaded", False)
    metrics = status_data.get("metrics") or {}
    feature_importances = status_data.get("feature_importances", {})

    if model_loaded:
        st.markdown('<span class="badge-high">Random Forest model loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-low">Heuristic fallback active</span>', unsafe_allow_html=True)

    if metrics:
        st.markdown('<p class="section-header">Model Performance</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, (label, key, fmt) in zip(
            [c1, c2, c3, c4],
            [
                ("Accuracy", "accuracy", ".1%"),
                ("Precision", "precision", ".1%"),
                ("Recall", "recall", ".1%"),
                ("F1 score", "f1_score", ".1%"),
            ],
        ):
            val = metrics.get(key, 0)
            formatted = format(val, fmt) if isinstance(val, float) else str(val)
            with col:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{formatted}</div>
                        <div class='metric-label'>{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        train_n = metrics.get("train_samples", "?")
        test_n = metrics.get("test_samples", "?")
        st.caption(f"Training samples: {train_n} | Test samples: {test_n}")

    if feature_importances:
        st.markdown('<p class="section-header">Feature Importance</p>', unsafe_allow_html=True)
        fi_col, table_col = st.columns([3, 2])
        with fi_col:
            fig = _plot_feature_importance(feature_importances)
            st.pyplot(fig)
            plt.close(fig)
        with table_col:
            sorted_fi = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            df_fi = pd.DataFrame(sorted_fi, columns=["Feature", "Importance"])
            df_fi["Feature"] = df_fi["Feature"].str.replace("_", " ").str.title()
            df_fi["Importance"] = df_fi["Importance"].map("{:.4f}".format)
            st.dataframe(df_fi, use_container_width=True, hide_index=True)

    st.markdown('<p class="section-header">Model Management</p>', unsafe_allow_html=True)
    ctrl_col1, ctrl_col2 = st.columns(2)

    with ctrl_col1:
        st.markdown("**Retrain model**")
        st.caption("Regenerate synthetic training data and update the fit classifier.")
        if st.button("Retrain now", use_container_width=True, type="primary"):
            with st.spinner("Training Random Forest classifier..."):
                result = trigger_model_retrain()
            if result and "error" not in result:
                new_metrics = result.get("metrics", {})
                st.success(
                    f"Model retrained. Accuracy {new_metrics.get('accuracy', 0):.1%}; "
                    f"F1 {new_metrics.get('f1_score', 0):.1%}."
                )
                st.rerun()
            else:
                st.error(f"Retraining failed: {result.get('error', 'Unknown error')}")

    with ctrl_col2:
        st.markdown("**Download model**")
        st.caption("Export the serialized classifier for inspection or deployment.")
        try:
            pkl_response = requests.get(ML_DOWNLOAD_ENDPOINT, timeout=15)
            if pkl_response.status_code == 200:
                st.download_button(
                    label="Download fit_classifier.pkl",
                    data=pkl_response.content,
                    file_name="fit_classifier.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            else:
                st.warning("Model file is not available. Train the model first.")
        except Exception:
            st.warning("Could not fetch model file from backend.")


# ============================================================================
# SECTION: Sidebar
# ============================================================================

backend_online, backend_msg = check_backend()

with st.sidebar:
    st.markdown(
        '<div class="brand-block"><div class="brand-title">SkillHire AI</div><div class="brand-subtitle">Professional job search workspace</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Search, rank, and track job opportunities from one workspace.")
    st.markdown("---")

    sidebar_jobs = len(st.session_state.get("jobs_db", []))
    sidebar_saved = len(st.session_state.get("saved_jobs", []))
    sidebar_recs = len(st.session_state.get("recommendations", []))
    st.markdown(
        f"""
        <div class="dashboard-panel">
            <div class="panel-title">Workspace snapshot</div>
            <div class="panel-subtitle">A quick status view before you run the next search.</div>
            <div class="workflow-strip" style="grid-template-columns: 1fr; gap: 0.55rem; margin-top: 0;">
                <div class="workflow-step">
                    <div class="workflow-step__index">LIVE</div>
                    <div class="workflow-step__title">Backend {"online" if backend_online else "offline"}</div>
                    <div class="workflow-step__copy">Resume parsing, fetches, and save actions run through FastAPI.</div>
                </div>
                <div class="workflow-step">
                    <div class="workflow-step__index">DATA</div>
                    <div class="workflow-step__title">{sidebar_jobs} jobs cached</div>
                    <div class="workflow-step__copy">Rank stored roles or add more from a company career board.</div>
                </div>
                <div class="workflow-step">
                    <div class="workflow-step__index">TRACKING</div>
                    <div class="workflow-step__title">{sidebar_saved} saved / {sidebar_recs} ranked</div>
                    <div class="workflow-step__copy">Bookmark promising jobs and revisit shortlisted matches later.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Search Controls")
    match_threshold = st.slider("Minimum match score (%)", 10, 100, 40, 5)
    max_recommendations = st.number_input("Max jobs to display", 1, 20, 5)
    job_database_limit = st.number_input("Database jobs to scan", 50, 1000, 500, 50)
    preferred_location = st.text_input(
        "Preferred location (optional)",
        placeholder="e.g. India, Remote, Seattle",
        help="Used for official Google/Microsoft/Amazon career search links.",
    )
    include_amazon = st.checkbox("Include Amazon Careers search", value=True)

    st.markdown("---")
    st.markdown("### Matching Mode")
    use_ml = st.toggle(
        "Use ML Classifier",
        value=True,
        help="ON = Random Forest model. OFF = heuristic scoring.",
    )
    if use_ml:
        st.success("ML classifier active")
    else:
        st.warning("Heuristic mode active")

    st.markdown("---")
    st.markdown("### Quick Tools")
    with st.expander("Add roles to database", expanded=False):
        st.caption("Fetch roles from public company boards and include them in future matching.")
        sidebar_job_source = st.radio(
            "Company board",
            [JOB_SOURCE_GREENHOUSE, JOB_SOURCE_LEVER, JOB_SOURCE_ASHBY],
            index=0,
            key="sidebar_database_job_source",
        )
        sidebar_board_map = {
            JOB_SOURCE_GREENHOUSE: "greenhouse",
            JOB_SOURCE_LEVER: "lever",
            JOB_SOURCE_ASHBY: "ashby",
        }
        sidebar_board = sidebar_board_map[sidebar_job_source]
        sidebar_popular_handles = {
            "greenhouse": ["Stripe", "Figma", "Reddit", "Airbnb", "HashiCorp", "GitHub", "Cloudflare", "Amplitude", "Custom..."],
            "lever": ["Vercel", "Figma", "Palantir", "Hotjar", "Lever", "Netflix", "Custom..."],
            "ashby": ["Linear", "Clerk", "Replicate", "Duolingo", "Custom..."],
        }
        sidebar_selected_company = st.selectbox(
            "Company",
            sidebar_popular_handles.get(sidebar_board, ["Custom..."]),
            key=f"sidebar_company_select_{sidebar_board}",
        )
        if sidebar_selected_company == "Custom...":
            sidebar_company_slug = st.text_input(
                "Company handle/slug",
                placeholder="e.g. stripe, figma, vercel",
                key=f"sidebar_company_custom_{sidebar_board}",
            )
        else:
            sidebar_company_slug = sidebar_selected_company.lower()

        if st.button(
            f"Fetch {sidebar_board.title()} roles",
            key=f"sidebar_fetch_roles_{sidebar_board}",
            use_container_width=True,
        ):
            if not backend_online:
                st.error("Backend is offline. Cannot fetch live jobs.")
            elif not sidebar_company_slug.strip():
                st.warning("Enter a company handle before fetching.")
            else:
                with st.spinner(f"Fetching roles from {sidebar_board.title()}..."):
                    jobs = fetch_jobs(sidebar_board, sidebar_company_slug)
                if jobs:
                    stored_jobs = fetch_stored_jobs(limit=int(job_database_limit))
                    st.session_state["jobs_db"] = stored_jobs or jobs
                    st.success(f"Saved {len(jobs)} roles from {sidebar_board.title()}.")
                else:
                    st.warning("No roles returned for that company board.")

    st.markdown("---")
    st.caption(
        "SkillHire AI matches your resume to jobs using skill overlap, "
        "semantic similarity, and role keywords."
    )
    st.caption("FastAPI | Streamlit | scikit-learn | Sentence-Transformers")


# ============================================================================
# SECTION: Main App Layout
# ============================================================================
parsed_resume_snapshot = st.session_state.get("parsed_resume")
recommendations_snapshot = st.session_state.get("recommendations", [])
saved_jobs_snapshot = st.session_state.get("saved_jobs", [])
jobs_snapshot = st.session_state.get("jobs_db", [])

consume_firebase_auth_payload()
render_auth_topbar()

# Application header
status_class = "status-online" if backend_online else "status-offline"
resume_state = "Parsed" if parsed_resume_snapshot else "Not parsed yet"
search_state = f"{len(recommendations_snapshot)} ranked roles" if recommendations_snapshot else "No ranked roles yet"
saved_state = f"{len(saved_jobs_snapshot)} saved jobs"
jobs_state = f"{len(jobs_snapshot)} jobs in cache"

st.markdown(
    f"""
    <div class="hero-shell">
        <div class="app-eyebrow">Career operations dashboard</div>
        <div class="app-title">Job Search Dashboard</div>
        <div class="hero-shell__meta">
            Use a single workspace to parse your resume, search live company boards, rank stored jobs, and keep a shortlist of the strongest opportunities.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4, gap="small")
with kpi_col1:
    st.markdown(
        f"""
        <div class="kpi-card {'kpi-card--success' if backend_online else 'kpi-card--warning'}">
            <div class="kpi-card__label">Platform status</div>
            <div class="kpi-card__value">{"Online" if backend_online else "Offline"}</div>
            <div class="kpi-card__hint">{backend_msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_col2:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card--accent">
            <div class="kpi-card__label">Resume state</div>
            <div class="kpi-card__value">{resume_state}</div>
            <div class="kpi-card__hint">Upload or paste a resume to activate recommendations.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_col3:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card--neutral">
            <div class="kpi-card__label">Search inventory</div>
            <div class="kpi-card__value">{len(jobs_snapshot)}</div>
            <div class="kpi-card__hint">Stored jobs available for ranking and filtering.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_col4:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card--success">
            <div class="kpi-card__label">Saved shortlist</div>
            <div class="kpi-card__value">{len(saved_jobs_snapshot)}</div>
            <div class="kpi-card__hint">Bookmark the strongest fits for later review.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not backend_online:
    st.warning(
        "The backend is not responding yet. On Render free tier it may be waking up; "
        "click Analyze to retry, or open the backend health URL once."
    )

st.write("")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_main, tab_ml = st.tabs(["Job Search", "Model Ops"])


# ============================================================================
# TAB 1: Job Search & Recommendations
# ============================================================================
with tab_main:
    profile_col, jobs_col = st.columns([1.05, 0.95], gap="large")
    main_col = st.container()

    if False:
        st.markdown('<p class="section-header">Quick Tools</p>', unsafe_allow_html=True)
        st.caption("Search controls, company boards, and utility actions.")

        st.markdown("<div class='dashboard-panel'>", unsafe_allow_html=True)
        st.markdown("**Search Controls**")
        match_threshold = st.slider("Minimum match score (%)", 10, 100, 40, 5)
        max_recommendations = st.number_input("Max jobs to display", 1, 20, 5)
        job_database_limit = st.number_input("Database jobs to scan", 50, 1000, 500, 50)
        preferred_location = st.text_input(
            "Preferred location (optional)",
            placeholder="e.g. India, Remote, Seattle",
            help="Used for official Google/Microsoft/Amazon career search links.",
        )
        include_amazon = st.checkbox("Include Amazon Careers search", value=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='dashboard-panel'>", unsafe_allow_html=True)
        st.markdown("**Matching Mode**")
        use_ml = st.toggle(
            "Use ML Classifier",
            value=True,
            help="ON = Random Forest model. OFF = heuristic scoring.",
        )
        if use_ml:
            st.success("ML classifier active")
        else:
            st.warning("Heuristic mode active")
        st.markdown("</div>", unsafe_allow_html=True)

        if False:
            st.caption("Fetch roles from public company boards. New roles are saved for future matching.")
            job_source = st.radio(
                "Company board",
                [JOB_SOURCE_GREENHOUSE, JOB_SOURCE_LEVER, JOB_SOURCE_ASHBY],
                index=0,
                key="database_job_source_legacy",
            )
            board_map = {
                JOB_SOURCE_GREENHOUSE: "greenhouse",
                JOB_SOURCE_LEVER: "lever",
                JOB_SOURCE_ASHBY: "ashby",
            }
            board = board_map[job_source]
            popular_handles = {
                "greenhouse": ["Stripe", "Figma", "Reddit", "Airbnb", "HashiCorp", "GitHub", "Cloudflare", "Amplitude", "Custom..."],
                "lever": ["Vercel", "Figma", "Palantir", "Hotjar", "Lever", "Netflix", "Custom..."],
                "ashby": ["Linear", "Clerk", "Replicate", "Duolingo", "Custom..."],
            }
            selected_option = st.selectbox(
                "Company",
                popular_handles.get(board, ["Custom..."]),
                key=f"database_company_select_legacy_{board}",
            )
            if selected_option == "Custom...":
                company_slug = st.text_input(
                    "Company handle/slug",
                    placeholder="e.g. stripe, figma, vercel",
                    key=f"database_company_custom_legacy_{board}",
                )
            else:
                company_slug = selected_option.lower()

            if st.button(
                f"Fetch {board.title()} roles into database",
                key=f"fetch_roles_legacy_{board}",
                use_container_width=True,
            ):
                if not backend_online:
                    st.error("Backend is offline. Cannot fetch live jobs.")
                elif not company_slug.strip():
                    st.warning("Enter a company handle before fetching.")
                else:
                    with st.spinner(f"Fetching jobs from {board.title()}..."):
                        jobs = fetch_jobs(board, company_slug)
                    if jobs:
                        stored_jobs = fetch_stored_jobs(limit=int(job_database_limit))
                        st.session_state["jobs_db"] = stored_jobs or jobs
                        st.success(f"Fetched and saved {len(jobs)} roles from {board.title()}.")
                    else:
                        st.warning("No roles returned for that company board.")

        render_top_career_links_section(backend_online)

    with profile_col:
        st.markdown('<p class="section-header">Candidate Profile</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF only)",
            type=["pdf"],
            help="We extract skills, sections, and text for job matching.",
        )
        st.caption("PDF upload is the supported input for this workspace.")
        if uploaded_file:
            st.caption(f"Selected: **{uploaded_file.name}**")

        mode_label = "ML Classifier" if use_ml else "Heuristic"
        if st.button(
            f"Analyze resume, fetch roles, and match ({mode_label})",
            type="primary",
            use_container_width=True,
        ):
            live_backend_online = backend_online
            if not live_backend_online:
                with st.spinner("Waking the backend service..."):
                    live_backend_online, backend_msg = check_backend(timeout=25, attempts=2)

            if not live_backend_online:
                st.error(
                    "Backend is still not reachable. Open the backend /health URL once, "
                    "wait for it to wake up, then try again."
                )
                st.caption(backend_msg)
            else:
                parsed = None
                with st.spinner("Parsing resume..."):
                    if uploaded_file is None:
                        st.warning("Upload a PDF resume first.")
                    else:
                        parsed = parse_resume(uploaded_file)

                if parsed:
                    st.session_state["parsed_resume"] = parsed
                    st.session_state.pop("recommendations", None)

                    with st.spinner("Ranking roles from the job database..."):
                        recs, jobs_analyzed = match_resume_against_database(
                            parsed_resume=parsed,
                            use_ml=use_ml,
                            job_limit=int(job_database_limit),
                        )

                    official = get_official_searches(
                        resume_text=parsed.get("raw_text", ""),
                        resume_skills=parsed.get("skills", []),
                        location=preferred_location,
                        include_amazon=include_amazon,
                    )
                    st.session_state["official_searches"] = official

                    if recs:
                        st.success(
                            f"Found {len(recs)} ranked matches from {jobs_analyzed} database jobs."
                        )
                    else:
                        st.warning(
                            "Resume parsed, but no database recommendations came back. "
                            "The automatic role fetch did not find usable listings yet. "
                            "Try again or use Quick Tools with a company handle."
                        )

        parsed_resume = st.session_state.get("parsed_resume")
        if parsed_resume:
            skills = parsed_resume.get("skills", [])
            st.caption(f"Current resume: **{len(skills)} extracted skills**")
            if st.button("Re-run role matching", use_container_width=True):
                with st.spinner("Re-ranking database roles..."):
                    recs, jobs_analyzed = match_resume_against_database(
                        parsed_resume=parsed_resume,
                        use_ml=use_ml,
                        job_limit=int(job_database_limit),
                    )
                if recs:
                    st.success(f"Updated matches from {jobs_analyzed} database jobs.")

        if False:
            st.caption("Fetch roles from public company boards. New roles are saved for future matching.")
            job_source = st.radio(
                "Company board",
                [JOB_SOURCE_GREENHOUSE, JOB_SOURCE_LEVER, JOB_SOURCE_ASHBY],
                index=0,
                key="database_job_source",
            )
            board_map = {
                JOB_SOURCE_GREENHOUSE: "greenhouse",
                JOB_SOURCE_LEVER: "lever",
                JOB_SOURCE_ASHBY: "ashby",
            }
            board = board_map[job_source]
            popular_handles = {
                "greenhouse": ["Stripe", "Figma", "Reddit", "Airbnb", "HashiCorp", "GitHub", "Cloudflare", "Amplitude", "Custom..."],
                "lever": ["Vercel", "Figma", "Palantir", "Hotjar", "Lever", "Netflix", "Custom..."],
                "ashby": ["Linear", "Clerk", "Replicate", "Duolingo", "Custom..."],
            }
            selected_option = st.selectbox(
                "Company",
                popular_handles.get(board, ["Custom..."]),
                key=f"database_company_select_{board}",
            )
            if selected_option == "Custom...":
                company_slug = st.text_input(
                    "Company handle/slug",
                    placeholder="e.g. stripe, figma, vercel",
                    key=f"database_company_custom_{board}",
                )
            else:
                company_slug = selected_option.lower()

            if st.button(f"Fetch {board.title()} roles into database", use_container_width=True):
                if not backend_online:
                    st.error("Backend is offline. Cannot fetch live jobs.")
                elif not company_slug.strip():
                    st.warning("Enter a company handle before fetching.")
                else:
                    with st.spinner(f"Fetching jobs from {board.title()}..."):
                        jobs = fetch_jobs(board, company_slug)
                    if jobs:
                        stored_jobs = fetch_stored_jobs(limit=int(job_database_limit))
                        st.session_state["jobs_db"] = stored_jobs or jobs
                        st.success(f"Fetched and saved {len(jobs)} roles from {board.title()}.")
                    else:
                        st.warning("No roles returned for that company board.")

    with jobs_col:
        render_top_career_links_section(backend_online, compact=True)

    with main_col:
        parsed_data = st.session_state.get("parsed_resume")
        recommendations = st.session_state.get("recommendations", [])

        if recommendations:
            jobs_analyzed = st.session_state.get("last_jobs_analyzed", len(st.session_state.get("jobs_db", [])))
            st.markdown('<p class="section-header">Recommended Roles</p>', unsafe_allow_html=True)
            st.caption(f"Ranked against {jobs_analyzed} roles from your database.")
            avg_score = int(sum(r.get("match_score", 0) for r in recommendations) / max(len(recommendations), 1))
            high_fit_count = sum(1 for r in recommendations if r.get("match_score", 0) >= 70)
            metric_1, metric_2, metric_3 = st.columns(3)
            metric_1.metric("Roles ranked", len(recommendations))
            metric_2.metric("Average match", f"{avg_score}%")
            metric_3.metric("Strong matches", high_fit_count)
            filtered = [r for r in recommendations if r.get("match_score", 0) >= match_threshold]
            filtered = filtered[:max_recommendations]

            if not filtered:
                st.warning(
                    f"No roles scored above {match_threshold}%. Lower the threshold in the sidebar."
                )
            else:
                for start in range(0, len(filtered), 2):
                    pair = filtered[start:start + 2]
                    pair_cols = st.columns(2, gap="large")
                    for col_index, rec in enumerate(pair):
                        with pair_cols[col_index]:
                            display_job_card(
                                rec,
                                card_key=f"rec_{start + col_index}",
                                backend_online=backend_online,
                            )

            with st.expander("Parsed resume profile", expanded=False):
                display_parsed_resume(parsed_data)
        elif parsed_data:
            st.markdown('<p class="section-header">Parsed Resume Profile</p>', unsafe_allow_html=True)
            display_parsed_resume(parsed_data)
            st.info("No role matches yet. Re-run matching to auto-fetch roles, or add a specific company from Quick Tools.")
        else:
            st.info("Paste a resume or upload a PDF to generate ranked roles from the database.")

        mode_label = "ML Classifier" if use_ml else "Heuristic"

        if False:
                st.markdown('<div id="candidate-profile-panel" class="dashboard-panel">', unsafe_allow_html=True)
                st.markdown('<div class="panel-title">Candidate Profile Summary</div>', unsafe_allow_html=True)
                if parsed_resume:
                    display_parsed_resume(parsed_resume)
                else:
                    st.info("Upload a resume to populate the candidate profile summary.")
                st.markdown('</div>', unsafe_allow_html=True)

                parsed_data = st.session_state.get("parsed_resume")
                recommendations = st.session_state.get("recommendations", [])

                if recommendations:
                    jobs_analyzed = st.session_state.get("last_jobs_analyzed", len(st.session_state.get("jobs_db", [])))
                    st.markdown('<p class="section-header">Recommended Roles</p>', unsafe_allow_html=True)
                    st.caption(f"Ranked against {jobs_analyzed} roles from your database.")
                    avg_score = int(sum(r.get("match_score", 0) for r in recommendations) / max(len(recommendations), 1))
                    high_fit_count = sum(1 for r in recommendations if r.get("match_score", 0) >= 70)
                    metric_1, metric_2, metric_3 = st.columns(3)
                    metric_1.metric("Roles ranked", len(recommendations))
                    metric_2.metric("Average match", f"{avg_score}%")
                    metric_3.metric("Strong matches", high_fit_count)
                    filtered = [r for r in recommendations if r.get("match_score", 0) >= match_threshold]
                    filtered = filtered[:max_recommendations]

                    if not filtered:
                        st.warning(
                            f"No roles scored above {match_threshold}%. Lower the threshold in the sidebar."
                        )
                    else:
                        for start in range(0, len(filtered), 2):
                            pair = filtered[start:start + 2]
                            pair_cols = st.columns(2, gap="large")
                            for col_index, rec in enumerate(pair):
                                with pair_cols[col_index]:
                                    display_job_card(
                                        rec,
                                        card_key=f"rec_{start + col_index}",
                                        backend_online=backend_online,
                                    )

                if st.session_state.get("recommendations") and st.session_state.get("parsed_resume"):
                    gap_col, suggest_col = st.columns(2, gap="large")

                    with gap_col:
                        st.markdown("### Skill Gap Summary")
                        display_skill_gap_summary(st.session_state["recommendations"])

                    with suggest_col:
                        st.markdown("### Resume Tips")
                        display_resume_suggestions(
                            st.session_state["parsed_resume"],
                            st.session_state["recommendations"],
                        )

                jobs_for_insights = st.session_state.get("jobs_db", [])
                recs_for_insights = st.session_state.get("recommendations")
                if jobs_for_insights:
                    st.markdown("---")
                    display_job_market_insights(jobs_for_insights, recs_for_insights)

                with st.expander("Career Advisor", expanded=True):
                    st.markdown('<div id="career-advisor-panel" class="chat-launcher-anchor"></div>', unsafe_allow_html=True)
                    display_career_chatbot(backend_online)

                st.markdown("---")
                display_saved_jobs_section(backend_online)
                for idx, rec in enumerate(filtered):
                    display_job_card(rec, card_key=f"rec_{idx}", backend_online=backend_online)

        elif parsed_data:
            st.markdown('<p class="section-header">👤 Parsed Resume Profile</p>', unsafe_allow_html=True)
            display_parsed_resume(parsed_data)
            st.info("Load jobs and click **⚡ Find Matching Jobs** to see recommendations.")
        else:
            st.info("Upload or paste a resume to generate role recommendations.")

    # ── Official Big Tech Search Links ────────────────────────────────────
    parsed_for_search = st.session_state.get("parsed_resume")
    if False and parsed_for_search:
        st.markdown("---")
        st.markdown("### 🏢 Official Big Tech Searches")
        st.caption(
            "These links open official Google, Microsoft, and Amazon career search pages "
            "pre-filled from your resume. No scraping — browse openings directly on each site."
        )

        search_col1, search_col2 = st.columns([3, 1])
        with search_col2:
            if st.button("🔄 Refresh search links", use_container_width=True):
                if backend_online:
                    st.session_state["official_searches"] = get_official_searches(
                        resume_text=parsed_for_search.get("raw_text", ""),
                        resume_skills=parsed_for_search.get("skills", []),
                        location=preferred_location,
                        include_amazon=include_amazon,
                    )
                else:
                    st.error("Backend offline — cannot refresh official search links.")

        official_searches = st.session_state.get("official_searches", [])
        if not official_searches and backend_online:
            official_searches = get_official_searches(
                resume_text=parsed_for_search.get("raw_text", ""),
                resume_skills=parsed_for_search.get("skills", []),
                location=preferred_location,
                include_amazon=include_amazon,
            )
            st.session_state["official_searches"] = official_searches

        if official_searches:
            search_cols = st.columns(min(len(official_searches), 3))
            for idx, source in enumerate(official_searches):
                with search_cols[idx % len(search_cols)]:
                    display_official_search_card(source)
        elif not backend_online:
            st.warning("Start the backend to generate official career search links.")
        else:
            st.info("Parse your resume to generate tailored Big Tech search links.")

    # ── Saved Jobs ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div id="career-advisor-panel" class="chat-launcher-anchor"></div>', unsafe_allow_html=True)
    with st.expander("Career Advisor", expanded=True):
        display_career_chatbot(backend_online)

    st.markdown("---")
    display_saved_jobs_section(backend_online)

    # ── Skill Gap + Resume Suggestions ───────────────────────────────────
    if st.session_state.get("recommendations") and st.session_state.get("parsed_resume"):
        st.markdown("---")
        gap_col, suggest_col = st.columns(2, gap="large")

        with gap_col:
            st.markdown("### 🎯 Skill Gap Summary")
            display_skill_gap_summary(st.session_state["recommendations"])

        with suggest_col:
            st.markdown("### 💡 Resume Tips")
            display_resume_suggestions(
                st.session_state["parsed_resume"],
                st.session_state["recommendations"],
            )

    # ── Market Insights ───────────────────────────────────────────────────
    jobs_for_insights = st.session_state.get("jobs_db", [])
    recs_for_insights = st.session_state.get("recommendations")
    if jobs_for_insights:
        st.markdown("---")
        display_job_market_insights(jobs_for_insights, recs_for_insights)


# ============================================================================
# TAB 2: AI Career Advisor Chatbot
# ============================================================================
if False:
    st.markdown('<p class="hero-title">💬 AI Career Advisor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Chat with our AI Advisor to decide on career choices, resume updates, and what skills to learn next.</p>',
        unsafe_allow_html=True,
    )

    if not backend_online:
        st.markdown('<span class="status-offline">⛔ Backend Offline</span>', unsafe_allow_html=True)
        st.warning("Please start the FastAPI backend to interact with the AI Career Advisor.")
    else:
        # Check environment and setup chat state
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": "Hello! I am your AI Career Advisor. Based on your parsed resume and job matching results, I can suggest roadmaps for skill gaps or help you decide which roles are the best fit. Ask me anything!"
                }
            ]

        # Context collection
        parsed_resume = st.session_state.get("parsed_resume", {})
        resume_text = parsed_resume.get("raw_text", "")
        resume_skills = parsed_resume.get("skills", [])
        recommendations = st.session_state.get("recommendations", [])
        career_goal = st.text_input(
            "Target role or career goal",
            placeholder="e.g. Backend Engineer, ML Intern, Full Stack Developer",
            key="advisor_career_goal",
        )
        
        # Calculate missing skills
        missing_skills = []
        if recommendations:
            missing_counter = Counter()
            for r in recommendations[:10]:
                for s in r.get("missing_skills", []):
                    missing_counter[s] += 1
            missing_skills = [s for s, _ in missing_counter.most_common(10)]

        # Display context overview in an expander
        with st.expander("💼 Advisor Context Overview", expanded=False):
            st.write("Below is the profile context the AI Advisor will use to personalize your advice:")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Extracted Skills:**")
                if resume_skills:
                    st.write(", ".join(resume_skills))
                else:
                    st.caption("No resume parsed yet.")
            with col2:
                st.write("**Top Skill Gaps:**")
                if missing_skills:
                    st.write(", ".join(missing_skills))
                else:
                    st.caption("No skill gaps calculated yet. Run job matching to detect gaps.")

        # Clear chat history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": "Hello! I am your AI Career Advisor. Based on your parsed resume and job matching results, I can suggest roadmaps for skill gaps or help you decide which roles are the best fit. Ask me anything!"
                }
            ]
            st.rerun()

        # Display chat messages
        for message in st.session_state["chat_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask me about career paths, resume tips, or skills..."):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})

            # Send messages history to backend
            messages_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["chat_messages"]
            ]

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("AI Advisor is thinking..."):
                    ai_response = send_chat_message(
                        messages=messages_history,
                        resume_text=resume_text,
                        resume_skills=resume_skills,
                        missing_skills=missing_skills,
                        job_recommendations=recommendations,
                        career_goal=career_goal,
                    )
                    if ai_response:
                        st.markdown(ai_response)
                        st.session_state["chat_messages"].append(
                            {"role": "assistant", "content": ai_response}
                        )
                        st.rerun()


# ============================================================================
# TAB 3: ML Control Center
# ============================================================================
with tab_ml:

    display_ml_control_center(backend_online)


render_chatbot_launcher()
