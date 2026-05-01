import json
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_curve

from nba_prescriptive_backend import (
    DEFAULT_MIN_GAMES,
    DEFAULT_MIN_TOTAL_MINUTES,
    DEFAULT_POSTSEASON_VALUE_USD,
    FEATURE_SETS_WITH_SALARY,
    PROJECT_DIR,
    PostseasonSwapEngine,
    SwapPolicy,
)


st.set_page_config(
    page_title="SwapIQ",
    page_icon=":basketball:",
    layout="wide",
)

BACKEND_CACHE_VERSION = "swapiq_v4"
PRIMARY_GOLD = "#AF9D74"
SECONDARY_GOLD = "#A49A87"
PAGE_BG = "#EBEBEB"
SURFACE = "#FFFFFF"
SURFACE_DARK = "#3F3F3F"
MUTED = "#968F83"
TEXT = "#1B1B1B"

st.markdown(
    f"""
    <style>
    :root {{
        --primary-gold: {PRIMARY_GOLD};
        --secondary-gold: {SECONDARY_GOLD};
        --page-bg: {PAGE_BG};
        --surface: {SURFACE};
        --surface-dark: {SURFACE_DARK};
        --muted: {MUTED};
        --text: {TEXT};
        --border: rgba(27, 27, 27, 0.18);
        --border-strong: rgba(27, 27, 27, 0.88);
        --radius: 6px;
        --font-ui: "Aptos", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        --font-display: "Georgia", "Times New Roman", serif;
    }}
    .stApp {{
        background: linear-gradient(180deg, rgba(175, 157, 116, 0.08) 0%, rgba(175, 157, 116, 0.00) 14%), var(--page-bg);
        color: var(--text);
        font-family: var(--font-ui);
    }}
    [data-testid="stHeader"] {{
        background: transparent;
        height: 0rem;
    }}
    [data-testid="stToolbar"] {{
        display: none;
    }}
    #MainMenu {{
        visibility: hidden;
    }}
    .block-container {{
        padding-top: 0.45rem;
        padding-bottom: 2rem;
        max-width: none;
        padding-left: 1.35rem;
        padding-right: 1.35rem;
    }}
    [data-testid="stSidebar"] {{
        display: none;
    }}
    .header-shell {{
        background: var(--surface);
        border: 1.5px solid var(--border-strong);
        border-radius: var(--radius);
        padding: 1rem 1.25rem 0.95rem 1.35rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 10px 24px rgba(27, 27, 27, 0.05);
    }}
    .header-kicker {{
        display: inline-block;
        background: var(--primary-gold);
        color: white;
        border-radius: 2px;
        padding: 0.2rem 0.55rem;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.55rem;
    }}
    .header-title {{
        font-family: var(--font-display);
        font-size: 2.15rem;
        font-weight: 800;
        line-height: 1.02;
        margin-bottom: 0.2rem;
    }}
    .header-subtitle {{
        color: rgba(27, 27, 27, 0.76);
        font-size: 0.95rem;
        line-height: 1.42;
        max-width: 58rem;
    }}
    .panel-shell {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.8rem 0.9rem;
        box-shadow: 0 8px 18px rgba(27, 27, 27, 0.04);
        margin-bottom: 0.8rem;
    }}
    .section-label {{
        display: inline-block;
        background: var(--primary-gold);
        color: white;
        border-radius: 2px;
        padding: 0.18rem 0.5rem;
        font-size: 0.76rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.6rem;
    }}
    .mini-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.9rem 0.95rem;
        box-shadow: 0 8px 18px rgba(27, 27, 27, 0.04);
        min-height: 132px;
    }}
    .mini-card h4 {{
        margin: 0 0 0.35rem 0;
        font-size: 0.95rem;
        font-weight: 800;
        color: var(--text);
    }}
    .mini-card p {{
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.48;
        color: rgba(27, 27, 27, 0.76);
    }}
    .summary-card {{
        background: linear-gradient(180deg, rgba(175, 157, 116, 0.08) 0%, rgba(175, 157, 116, 0.02) 100%), var(--surface);
        border: 1px solid rgba(27, 27, 27, 0.16);
        border-radius: var(--radius);
        padding: 0.9rem 0.95rem;
        box-shadow: 0 10px 22px rgba(27, 27, 27, 0.05);
        min-height: 122px;
    }}
    .summary-label {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 800;
        color: rgba(27, 27, 27, 0.58);
        margin-bottom: 0.35rem;
    }}
    .summary-value {{
        font-size: 1.6rem;
        font-weight: 800;
        line-height: 1.08;
        color: var(--text);
        margin-bottom: 0.28rem;
    }}
    .summary-copy {{
        font-size: 0.9rem;
        line-height: 1.46;
        color: rgba(27, 27, 27, 0.74);
    }}
    .page-note {{
        background: #F7F4ED;
        border-left: 4px solid var(--primary-gold);
        border-radius: 4px;
        padding: 0.55rem 0.75rem;
        margin: 0.15rem 0 0.75rem 0;
        color: rgba(27, 27, 27, 0.82);
        font-size: 0.88rem;
        line-height: 1.4;
    }}
    .kpi-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.8rem 0.85rem;
        box-shadow: 0 8px 18px rgba(27, 27, 27, 0.04);
    }}
    .kpi-label {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 800;
        color: rgba(27, 27, 27, 0.58);
        margin-bottom: 0.35rem;
    }}
    .kpi-value {{
        font-size: 1.82rem;
        font-weight: 800;
        line-height: 1.02;
        color: var(--text);
        margin-bottom: 0.15rem;
    }}
    .kpi-help {{
        font-size: 0.86rem;
        color: rgba(27, 27, 27, 0.68);
        line-height: 1.35;
    }}
    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stPopover"] button {{
        width: 100%;
        border-radius: 4px;
        border: 1.5px solid black !important;
        background: black !important;
        color: white !important;
        font-weight: 700;
        box-shadow: none !important;
    }}
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    div[data-testid="stPopover"] button:hover {{
        background: #111111 !important;
        color: white !important;
        border-color: black !important;
    }}
    .stButton > button:focus,
    .stDownloadButton > button:focus,
    div[data-testid="stPopover"] button:focus {{
        outline: none !important;
        box-shadow: 0 0 0 1px black !important;
    }}
    .stSelectbox label, .stNumberInput label, .stSlider label, .stCheckbox label, .stRadio label {{
        font-weight: 700;
    }}
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stNumberInput input,
    .stTextInput input {{
        background: var(--surface-dark) !important;
        color: white !important;
        border-radius: 4px !important;
        border: 1px solid black !important;
    }}
    .stSelectbox svg, .stNumberInput svg {{
        color: white !important;
        fill: white !important;
    }}
    .stSlider [role="slider"] {{
        background: black !important;
        border: 1px solid black !important;
    }}
    .stSlider [data-testid="stTickBar"] {{
        background: black !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        width: 100%;
        border: 1px solid rgba(27, 27, 27, 0.18);
        border-radius: 6px;
        overflow: hidden;
        background: white;
    }}
    .stTabs [data-baseweb="tab"] {{
        flex: 1 1 0;
        justify-content: center;
        min-height: 3.2rem;
        border-radius: 0;
        border: none;
        border-right: 1px solid rgba(27, 27, 27, 0.18);
        background: #F6F3EC;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.8rem 1rem;
    }}
    .stTabs [data-baseweb="tab"]:last-child {{
        border-right: none;
    }}
    .stTabs [aria-selected="true"] {{
        background: var(--primary-gold) !important;
        color: white !important;
        border-right: 1px solid black !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background: #E9E1D1;
    }}
    div[data-testid="stMetric"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.85rem 0.95rem;
        box-shadow: 0 8px 18px rgba(27, 27, 27, 0.04);
    }}
    .stAlert {{
        border-radius: 4px;
        border: 1px solid var(--border);
    }}
    .nav-active {{
        font-size: 0.82rem;
        color: rgba(27, 27, 27, 0.62);
        font-weight: 700;
        margin-bottom: 0.6rem;
    }}
    .settings-anchor {{
        display: none;
    }}
    div[data-testid="stVerticalBlock"]:has(.settings-anchor) {{
        position: sticky;
        top: 1rem;
        align-self: start;
        z-index: 20;
    }}
    .top-nav {{
        margin: 0.05rem 0 0.8rem 0;
        padding: 0;
        border-bottom: none;
    }}
    .context-rail {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.7rem 0.9rem 0.2rem 0.9rem;
        box-shadow: 0 8px 18px rgba(27, 27, 27, 0.04);
        margin-bottom: 0.8rem;
    }}
    .context-title {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 800;
        color: rgba(27, 27, 27, 0.58);
        margin-bottom: 0.45rem;
    }}
    div[data-testid="stSegmentedControl"] {{
        margin-bottom: 0.75rem;
    }}
    div[data-testid="stSegmentedControl"] button,
    div[data-testid="stSegmentedControl"] [role="button"] {{
        min-height: 3.1rem !important;
        border-radius: 0 !important;
        border: 1px solid rgba(27, 27, 27, 0.28) !important;
        background: #F6F3EC !important;
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
        box-shadow: none !important;
    }}
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[data-testid="stSegmentedControl"] button[aria-selected="true"],
    div[data-testid="stSegmentedControl"] button[kind="primary"],
    div[data-testid="stSegmentedControl"] [role="button"][aria-pressed="true"],
    div[data-testid="stSegmentedControl"] [role="button"][aria-selected="true"] {{
        background: var(--primary-gold) !important;
        color: white !important;
        -webkit-text-fill-color: white !important;
        border-color: black !important;
    }}
    div[data-testid="stSegmentedControl"] button:hover,
    div[data-testid="stSegmentedControl"] [role="button"]:hover {{
        background: #E9E1D1 !important;
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        border-color: black !important;
    }}
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"]:hover,
    div[data-testid="stSegmentedControl"] button[aria-selected="true"]:hover,
    div[data-testid="stSegmentedControl"] button[kind="primary"]:hover,
    div[data-testid="stSegmentedControl"] [role="button"][aria-pressed="true"]:hover,
    div[data-testid="stSegmentedControl"] [role="button"][aria-selected="true"]:hover {{
        background: var(--primary-gold) !important;
        color: white !important;
        -webkit-text-fill-color: white !important;
    }}
    @media (max-width: 700px) {{
        div[data-testid="stSegmentedControl"] button,
        div[data-testid="stSegmentedControl"] [role="button"] {{
            min-height: 2.85rem !important;
            background: #F6F3EC !important;
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
        }}
        div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
        div[data-testid="stSegmentedControl"] button[aria-selected="true"],
        div[data-testid="stSegmentedControl"] button[kind="primary"],
        div[data-testid="stSegmentedControl"] [role="button"][aria-pressed="true"],
        div[data-testid="stSegmentedControl"] [role="button"][aria-selected="true"] {{
            background: var(--primary-gold) !important;
            color: white !important;
            -webkit-text-fill-color: white !important;
        }}
    }}
    .top-nav-row .stButton > button {{
        background: #F6F3EC !important;
        border-color: rgba(27, 27, 27, 0.22) !important;
        color: #1B1B1B !important;
        width: 100% !important;
        min-height: 3rem !important;
        font-size: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        border-radius: 0 !important;
    }}
    .top-nav-row .stButton > button:hover {{
        background: #E9E1D1 !important;
        border-color: #1B1B1B !important;
        color: #1B1B1B !important;
    }}
    .nav-tab-active {{
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 3rem;
        width: 100%;
        background: var(--primary-gold);
        color: white;
        border: 1.5px solid black;
        border-radius: 0;
        font-weight: 800;
        font-size: 1rem;
        box-sizing: border-box;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model and cleaned roster data...")
def load_engine(feature_set_name: str, postseason_value_usd: float, cache_version: str):
    return PostseasonSwapEngine(
        postseason_value_usd=postseason_value_usd,
        feature_set_name=feature_set_name,
    )


def init_state():
    defaults = {
        "active_page": "Model",
        "nav_page": "Model",
        "feature_set_name": "analyst_positional" if "analyst_positional" in FEATURE_SETS_WITH_SALARY else list(FEATURE_SETS_WITH_SALARY.keys())[0],
        "postseason_value_usd": float(DEFAULT_POSTSEASON_VALUE_USD),
        "top_n": 15,
        "min_games": DEFAULT_MIN_GAMES,
        "min_total_minutes": int(DEFAULT_MIN_TOTAL_MINUTES),
        "allow_multi_team": False,
        "season": None,
        "team": None,
        "model_plot": "Confusion matrix",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def make_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def read_binary_if_exists(path: Path):
    return path.read_bytes() if path.exists() else None


def load_share_links():
    config_path = PROJECT_DIR / "share_links.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def filter_by_position(df: pd.DataFrame, position_filter: str, column_name: str) -> pd.DataFrame:
    if df.empty or position_filter == "All positions":
        return df.copy()
    return df[df[column_name] == position_filter].reset_index(drop=True)


def format_feature_set_name(name: str) -> str:
    return name.replace("_", " ").title()


def render_kpi_card(column, label: str, value: str, help_text: str):
    column.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_card(column, label: str, value: str, copy: str):
    column.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">{label}</div>
            <div class="summary-value">{value}</div>
            <div class="summary-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def themed_altair(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(stroke="black", strokeWidth=1.0)
        .configure_axis(
            labelColor=TEXT,
            titleColor=TEXT,
            grid=False,
            domainColor="black",
            tickColor="black",
        )
        .configure_legend(
            labelColor=TEXT,
            titleColor=TEXT,
            fillColor=SURFACE,
            strokeColor="black",
        )
        .configure_title(color=TEXT, fontSize=18)
    )


def render_model_plot(engine: PostseasonSwapEngine, plot_name: str):
    validation = engine.validation
    folds = pd.DataFrame(validation.get("folds", []))
    preds = pd.DataFrame(validation.get("predictions", []))

    if plot_name == "Confusion matrix":
        cm = np.array([[validation["tn"], validation["fp"]], [validation["fn"], validation["tp"]]])
        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        ax.set_facecolor(SURFACE)
        cell_colors = np.array([[SECONDARY_GOLD, SURFACE], [SURFACE, SECONDARY_GOLD]], dtype=object)
        for i in range(2):
            for j in range(2):
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cell_colors[i, j], edgecolor="black", linewidth=1.5)
                ax.add_patch(rect)
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=15, fontweight="bold", color="black")
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted No", "Predicted Yes"], fontweight="bold")
        ax.set_yticklabels(["Actual No", "Actual Yes"], fontweight="bold")
        ax.set_title("LOSO-CV Confusion Matrix", fontsize=14, fontweight="bold", pad=8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("black")
        ax.tick_params(length=0)
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout(pad=0.6)
        plot_col, _ = st.columns([1.35, 2.65], gap="small")
        with plot_col:
            st.pyplot(fig, width="content")
        plt.close(fig)
        return

    if plot_name == "Coefficient chart":
        coef_df = pd.DataFrame(
            {
                "feature": engine.model.params.index,
                "coefficient": engine.model.params.values,
                "p_value": engine.model.pvalues.values,
            }
        )
        coef_df = coef_df[coef_df["feature"] != "const"].copy()
        coef_df["abs_coef"] = coef_df["coefficient"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        ax.barh(coef_df["feature"], coef_df["coefficient"], color=SECONDARY_GOLD, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=1.4)
        ax.invert_yaxis()
        ax.set_title("Final Logistic Coefficients", fontsize=18, fontweight="bold", pad=12)
        ax.set_xlabel("Coefficient", fontweight="bold")
        ax.set_facecolor(SURFACE)
        fig.patch.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_linewidth(1.3)
            spine.set_color("black")
        ax.tick_params(axis="y", labelsize=10)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        return

    if plot_name == "Lift chart":
        if preds.empty or len(preds) < 10:
            st.info("This backend does not expose enough held-out predictions for a lift chart.")
            return
        lift_df = preds.copy().sort_values("pred_prob", ascending=False).reset_index(drop=True)
        lift_df["decile"] = pd.qcut(np.arange(len(lift_df)), q=10, labels=[f"D{i}" for i in range(1, 11)])
        baseline_rate = lift_df["postseason"].mean()
        lift_summary = (
            lift_df.groupby("decile", observed=False)
            .agg(actual_rate=("postseason", "mean"), avg_pred=("pred_prob", "mean"))
            .reset_index()
        )
        lift_summary["lift"] = lift_summary["actual_rate"] / baseline_rate
        lift_summary["decile_num"] = lift_summary["decile"].str.replace("D", "", regex=False).astype(int)
        chart = (
            alt.Chart(lift_summary)
            .mark_bar(color=SECONDARY_GOLD, stroke="black", strokeWidth=0.5)
            .encode(
                x=alt.X(
                    "decile:N",
                    title="Predicted probability decile (D1 = highest)",
                    sort=alt.SortField(field="decile_num", order="ascending"),
                ),
                y=alt.Y("lift:Q", title="Lift vs overall postseason rate"),
                tooltip=[
                    alt.Tooltip("decile:N", title="Decile"),
                    alt.Tooltip("actual_rate:Q", title="Observed rate", format=".3f"),
                    alt.Tooltip("avg_pred:Q", title="Average predicted probability", format=".3f"),
                    alt.Tooltip("lift:Q", title="Lift", format=".2f"),
                ],
            )
            .properties(height=280, title="Lift Chart from Held-Out LOSO Predictions")
        )
        rule = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color="black", strokeDash=[6, 4]).encode(y="y:Q")
        st.altair_chart(themed_altair(chart + rule), width="stretch")
        return

    if plot_name == "Calibration curve":
        if preds.empty:
            st.info("This backend does not expose held-out predictions for calibration.")
            return
        calib_df = preds.copy()
        calib_df["bin"] = pd.qcut(calib_df["pred_prob"], q=min(10, calib_df["pred_prob"].nunique()), duplicates="drop")
        calib_summary = (
            calib_df.groupby("bin", observed=False)
            .agg(avg_pred=("pred_prob", "mean"), actual_rate=("postseason", "mean"), n=("postseason", "size"))
            .dropna()
            .reset_index(drop=True)
        )
        curve = (
            alt.Chart(calib_summary)
            .mark_line(color=PRIMARY_GOLD, point=alt.OverlayMarkDef(filled=True, fill=PRIMARY_GOLD, stroke="black"))
            .encode(
                x=alt.X("avg_pred:Q", title="Average predicted probability", axis=alt.Axis(format=".0%")),
                y=alt.Y("actual_rate:Q", title="Observed postseason rate", axis=alt.Axis(format=".0%")),
                tooltip=[
                    alt.Tooltip("avg_pred:Q", title="Average predicted probability", format=".3f"),
                    alt.Tooltip("actual_rate:Q", title="Observed rate", format=".3f"),
                    alt.Tooltip("n:Q", title="Teams"),
                ],
            )
            .properties(height=280, title="Calibration Curve from Held-Out LOSO Predictions")
        )
        diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="black", strokeDash=[6, 4]).encode(x="x:Q", y="y:Q")
        st.altair_chart(themed_altair(curve + diagonal), width="stretch")
        return

    if plot_name == "Probability distribution":
        if preds.empty:
            st.info("This backend does not expose held-out predictions for a probability distribution.")
            return
        hist = (
            alt.Chart(preds)
            .mark_bar(color=SECONDARY_GOLD, stroke="black", strokeWidth=0.5)
            .encode(
                x=alt.X("pred_prob:Q", bin=alt.Bin(maxbins=12), title="Held-out predicted probability"),
                y=alt.Y("count():Q", title="Team-seasons"),
                tooltip=[alt.Tooltip("count():Q", title="Count")],
            )
            .properties(height=280, title="Distribution of Held-Out Predicted Probabilities")
        )
        st.altair_chart(themed_altair(hist), width="stretch")
        return

    if plot_name == "Top false positives / negatives":
        if preds.empty:
            st.info("This backend does not expose held-out predictions for misclassification diagnostics.")
            return
        error_df = preds.copy()
        error_df["error_type"] = np.select(
            [
                (error_df["postseason"] == 0) & (error_df["pred_class"] == 1),
                (error_df["postseason"] == 1) & (error_df["pred_class"] == 0),
            ],
            ["False positive", "False negative"],
            default="Correct",
        )
        false_pos = (
            error_df[error_df["error_type"] == "False positive"]
            .sort_values("pred_prob", ascending=False)
            .head(8)
            .copy()
        )
        false_neg = (
            error_df[error_df["error_type"] == "False negative"]
            .sort_values("pred_prob", ascending=True)
            .head(8)
            .copy()
        )
        fp_col, fn_col = st.columns(2, gap="large")
        with fp_col:
            st.markdown("**Top False Positives**")
            if false_pos.empty:
                st.caption("No false positives in held-out predictions.")
            else:
                st.dataframe(
                    false_pos[["Season", "Team", "pred_prob"]]
                    .rename(columns={"Season": "Season", "Team": "Team", "pred_prob": "Predicted probability"})
                    .style.format({"Predicted probability": "{:.3f}"}),
                    width="stretch",
                )
        with fn_col:
            st.markdown("**Top False Negatives**")
            if false_neg.empty:
                st.caption("No false negatives in held-out predictions.")
            else:
                st.dataframe(
                    false_neg[["Season", "Team", "pred_prob"]]
                    .rename(columns={"Season": "Season", "Team": "Team", "pred_prob": "Predicted probability"})
                    .style.format({"Predicted probability": "{:.3f}"}),
                    width="stretch",
                )
        return

    if folds.empty:
        st.info("This backend does not expose season-level validation folds for this plot.")
        return

    if plot_name == "Accuracy over seasons":
        chart = (
            alt.Chart(folds)
            .mark_line(color=PRIMARY_GOLD, point=alt.OverlayMarkDef(filled=True, fill=PRIMARY_GOLD, stroke="black"))
            .encode(
                x=alt.X("season:N", title="Holdout season"),
                y=alt.Y("accuracy:Q", title="Accuracy", axis=alt.Axis(format=".0%")),
                tooltip=[alt.Tooltip("season:N", title="Season"), alt.Tooltip("accuracy:Q", title="Accuracy", format=".3f")],
            )
            .properties(height=280, title="LOSO Accuracy by Season")
        )
        st.altair_chart(themed_altair(chart), width="stretch")
        return

    if plot_name == "AUC over seasons":
        if "auc" not in folds.columns or folds["auc"].dropna().empty:
            st.info("This backend does not expose season-level AUC values for this plot.")
            return
        chart = (
            alt.Chart(folds.dropna(subset=["auc"]))
            .mark_line(color=SECONDARY_GOLD, point=alt.OverlayMarkDef(filled=True, fill=SECONDARY_GOLD, stroke="black"))
            .encode(
                x=alt.X("season:N", title="Holdout season"),
                y=alt.Y("auc:Q", title="AUC"),
                tooltip=[alt.Tooltip("season:N", title="Season"), alt.Tooltip("auc:Q", title="AUC", format=".3f")],
            )
            .properties(height=280, title="LOSO AUC by Season")
        )
        st.altair_chart(themed_altair(chart), width="stretch")
        return

    if plot_name == "Brier over seasons":
        chart = (
            alt.Chart(folds)
            .mark_line(color=MUTED, point=alt.OverlayMarkDef(filled=True, fill=MUTED, stroke="black"))
            .encode(
                x=alt.X("season:N", title="Holdout season"),
                y=alt.Y("brier:Q", title="Brier score"),
                tooltip=[alt.Tooltip("season:N", title="Season"), alt.Tooltip("brier:Q", title="Brier", format=".3f")],
            )
            .properties(height=280, title="LOSO Brier Score by Season")
        )
        st.altair_chart(themed_altair(chart), width="stretch")
        return

    if plot_name == "Observed vs predicted rate":
        rate_folds = folds.rename(columns={"actual_rate": "Observed rate", "predicted_rate": "Average predicted probability"})
        chart = (
            alt.Chart(rate_folds)
            .transform_fold(["Observed rate", "Average predicted probability"], as_=["Series", "Rate"])
            .mark_line(point=alt.OverlayMarkDef(filled=True, stroke="black"))
            .encode(
                x=alt.X("season:N", title="Holdout season"),
                y=alt.Y("Rate:Q", title="Rate", axis=alt.Axis(format=".0%")),
                color=alt.Color(
                    "Series:N",
                    scale=alt.Scale(
                        domain=["Observed rate", "Average predicted probability"],
                        range=[PRIMARY_GOLD, SECONDARY_GOLD],
                    ),
                    legend=alt.Legend(title=None),
                ),
                tooltip=[
                    alt.Tooltip("season:N", title="Season"),
                    alt.Tooltip("Series:N", title="Series"),
                    alt.Tooltip("Rate:Q", title="Rate", format=".3f"),
                ],
            )
            .properties(height=280, title="Observed vs Predicted Postseason Rate")
        )
        st.altair_chart(themed_altair(chart), width="stretch")
        return

    if plot_name == "Training fit by season":
        chart = (
            alt.Chart(folds)
            .mark_bar(color=SECONDARY_GOLD, stroke="black", strokeWidth=0.5)
            .encode(
                x=alt.X("pseudo_r2_train:Q", title="Training pseudo R2"),
                y=alt.Y("season:N", sort="-x", title="Holdout season"),
                tooltip=[
                    alt.Tooltip("season:N", title="Season"),
                    alt.Tooltip("pseudo_r2_train:Q", title="Pseudo R2", format=".3f"),
                    alt.Tooltip("aic_train:Q", title="Train AIC", format=".1f"),
                ],
            )
            .properties(height=280, title="Training Fit During LOSO-CV")
        )
        st.altair_chart(themed_altair(chart), width="stretch")
        return


def get_available_model_plots(engine: PostseasonSwapEngine) -> list[str]:
    validation = engine.validation
    available = ["Confusion matrix", "Coefficient chart"]
    folds = pd.DataFrame(validation.get("folds", []))
    preds = pd.DataFrame(validation.get("predictions", []))
    if not preds.empty:
        available.append("Lift chart")
    if not folds.empty:
        if "accuracy" in folds.columns:
            available.append("Accuracy over seasons")
        if "auc" in folds.columns and folds["auc"].dropna().shape[0] > 0:
            available.append("AUC over seasons")
        if "brier" in folds.columns:
            available.append("Brier over seasons")
        if {"actual_rate", "predicted_rate", "season"}.issubset(folds.columns):
            available.append("Observed vs predicted rate")
        if {"pseudo_r2_train", "aic_train", "season"}.issubset(folds.columns):
            available.append("Training fit by season")
    return available


MODEL_PLOT_NOTES = {
    "Confusion matrix": "Shows how often the model correctly and incorrectly classifies postseason qualification on held-out seasons.",
    "Coefficient chart": "Ranks the final model's feature effects by magnitude so we can see which inputs most strongly move postseason odds.",
    "Lift chart": "Shows whether the highest predicted-probability groups actually make the postseason at higher rates than the overall sample.",
    "Calibration curve": "Checks whether predicted probabilities match observed postseason rates, which matters because the tool converts probability into expected value.",
    "Probability distribution": "Shows how widely the held-out predicted probabilities are spread across team-seasons, indicating how decisively the model separates teams.",
    "Top false positives / negatives": "Highlights the biggest held-out misses so we can see which teams the model was most overconfident or underconfident about.",
    "Accuracy over seasons": "Tracks held-out classification accuracy by season to show whether performance stays stable over time.",
    "AUC over seasons": "Tracks the model's ranking quality by season, independent of the specific probability threshold used for classification.",
    "Brier over seasons": "Tracks probability error by season, where lower values indicate better-calibrated postseason probabilities.",
    "Observed vs predicted rate": "Compares the actual postseason rate to the model's average predicted rate in each holdout season.",
    "Training fit by season": "Shows how well each training fold fit the training data, which helps contextualize variation across LOSO splits.",
}


def build_policy() -> SwapPolicy:
    return SwapPolicy(
        min_games=int(st.session_state["min_games"]),
        min_total_minutes=float(st.session_state["min_total_minutes"]),
        allow_multi_team_replacements=bool(st.session_state["allow_multi_team"]),
        strict_position_match=True,
    )


def render_header_and_settings(engine: PostseasonSwapEngine):
    share_links = load_share_links()
    methodology_colab_url = share_links.get("methodology_colab_url")
    judge_colab_url = share_links.get("judge_colab_url")
    readme_path = PROJECT_DIR / "README.md"
    readme_bytes = read_binary_if_exists(readme_path)

    header_cols = st.columns([6.3, 1.5])
    with header_cols[0]:
        st.markdown(
            """
            <div class="header-shell">
                <div class="header-kicker">NBA Decision Support</div>
                <div class="header-title">SwapIQ</div>
                <div class="header-subtitle">
                    A refined roster-optimization dashboard for testing whether same-position player swaps
                    improve postseason probability enough to justify their salary cost.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with header_cols[1]:
        settings_box = st.container()
        with settings_box:
            st.markdown('<div class="settings-anchor"></div>', unsafe_allow_html=True)
        with st.popover("Search Settings", width="stretch"):
            st.caption("Shared search and business settings")
            st.number_input(
                "Postseason value (USD)",
                min_value=0.0,
                step=500_000.0,
                format="%.0f",
                key="postseason_value_usd",
            )
            st.slider("Top rows to display", min_value=5, max_value=50, step=5, key="top_n")
            st.slider("Minimum candidate games", min_value=0, max_value=82, step=1, key="min_games")
            st.slider("Minimum candidate total minutes", min_value=0, max_value=2000, step=50, key="min_total_minutes")
            st.checkbox("Allow multi-team replacement candidates", key="allow_multi_team")

            st.markdown("---")
            if methodology_colab_url:
                st.link_button("Open methodology notebook", methodology_colab_url, width="stretch")
            if judge_colab_url:
                st.link_button("Open judge notebook", judge_colab_url, width="stretch")
            if readme_bytes:
                st.download_button(
                    "Download README",
                    data=readme_bytes,
                    file_name="README.md",
                    mime="text/markdown",
                    width="stretch",
                )


def render_top_navigation():
    st.markdown('<div class="top-nav">', unsafe_allow_html=True)
    pages = ["Model", "Threats", "Opportunities"]
    if st.session_state.get("nav_page") not in pages:
        st.session_state["nav_page"] = st.session_state["active_page"] if st.session_state["active_page"] in pages else "Model"
    if hasattr(st, "segmented_control"):
        selected_page = st.segmented_control(
            "Primary navigation",
            options=pages,
            key="nav_page",
            label_visibility="collapsed",
            width="stretch",
        )
        if selected_page in pages and selected_page != st.session_state["active_page"]:
            st.session_state["active_page"] = selected_page
            st.rerun()
    else:
        nav_cols = st.columns(3, gap="small")
        for col, page_name in zip(nav_cols, pages):
            with col:
                if st.session_state["active_page"] == page_name:
                    st.markdown(f'<div class="nav-tab-active">{page_name}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="top-nav-row">', unsafe_allow_html=True)
                    if st.button(page_name, key=f"nav_{page_name.lower()}"):
                        st.session_state["active_page"] = page_name
                        st.session_state["nav_page"] = page_name
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_context_rail(engine: PostseasonSwapEngine):
    available_seasons = sorted(engine.team_season["Season"].unique().tolist())
    if st.session_state["season"] not in available_seasons:
        st.session_state["season"] = "'15-'16" if "'15-'16" in available_seasons else available_seasons[-1]

    teams_for_season = sorted(
        engine.team_season.loc[engine.team_season["Season"] == st.session_state["season"], "Team"].unique().tolist()
    )
    if st.session_state["team"] not in teams_for_season:
        st.session_state["team"] = "IND" if "IND" in teams_for_season else teams_for_season[0]

    with st.container(border=True):
        st.markdown('<div class="context-title">Search Context</div>', unsafe_allow_html=True)
        rail_cols = st.columns([1, 1, 6], gap="small")
        with rail_cols[0]:
            st.selectbox("Season", available_seasons, key="season")
        with rail_cols[1]:
            refreshed_teams = sorted(
                engine.team_season.loc[engine.team_season["Season"] == st.session_state["season"], "Team"].unique().tolist()
            )
            if st.session_state["team"] not in refreshed_teams:
                st.session_state["team"] = "IND" if "IND" in refreshed_teams else refreshed_teams[0]
            st.selectbox("Team", refreshed_teams, key="team")


def render_model_page(engine: PostseasonSwapEngine):
    validation = engine.validation
    available_plots = get_available_model_plots(engine)
    if st.session_state["model_plot"] not in available_plots:
        st.session_state["model_plot"] = available_plots[0]
    st.markdown('<div class="section-label">Model Overview</div>', unsafe_allow_html=True)
    st.subheader("Model diagnostics")
    st.markdown(
        '<div class="page-note">Use this page to check whether the probability engine is stable enough to support roster decisions before turning it into swap recommendations.</div>',
        unsafe_allow_html=True,
    )
    control_cols = st.columns([2.4, 1.6], gap="large")
    with control_cols[0]:
        st.selectbox(
            "Model feature set",
            options=list(FEATURE_SETS_WITH_SALARY.keys()),
            index=list(FEATURE_SETS_WITH_SALARY.keys()).index(st.session_state["feature_set_name"]),
            format_func=format_feature_set_name,
            key="feature_set_name",
            help="Changing the feature set reloads the scoring model for all pages.",
        )
    with control_cols[1]:
        st.markdown(
            f"""
            <div class="summary-card" style="min-height:88px;padding:0.75rem 0.85rem;">
                <div class="summary-label">Current model</div>
                <div class="summary-value" style="font-size:1.15rem;">{len(engine.model_features)} standardized features</div>
                <div class="summary-copy">{format_feature_set_name(st.session_state["feature_set_name"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    metric_cols = st.columns(4)
    render_kpi_card(metric_cols[0], "LOSO Accuracy", f"{validation['accuracy']:.3f}", "Held-out season classification accuracy")
    render_kpi_card(metric_cols[1], "LOSO F1", f"{validation['f1']:.3f}", "Balance of precision and recall")
    render_kpi_card(metric_cols[2], "AUC", f"{validation['auc']:.3f}" if "auc" in validation else "N/A", "Ranking quality for postseason probability")
    render_kpi_card(metric_cols[3], "Brier", f"{validation['brier']:.3f}" if "brier" in validation else "N/A", "Probability calibration error")

    plot_cols = st.columns([1.05, 3.2], gap="large")
    with plot_cols[0]:
        st.markdown('<div class="section-label">Plot Generator</div>', unsafe_allow_html=True)
        st.selectbox(
            "Select performance plot",
            options=available_plots,
            key="model_plot",
        )
        with st.expander("Model details", expanded=False):
            st.write("Features:", engine.model_features)
            st.write("LOSO-CV confusion matrix:", [[validation["tn"], validation["fp"]], [validation["fn"], validation["tp"]]])

    with plot_cols[1]:
        with st.container(border=True):
            st.markdown(
                f'<div class="summary-copy" style="margin-bottom:0.55rem;">{MODEL_PLOT_NOTES.get(st.session_state["model_plot"], "")}</div>',
                unsafe_allow_html=True,
            )
            render_model_plot(engine, st.session_state["model_plot"])


def render_opportunities_page(engine: PostseasonSwapEngine, player_data: pd.DataFrame):
    season = st.session_state["season"]
    team = st.session_state["team"]
    policy = build_policy()

    positions_for_team = sorted(
        player_data.loc[
            (player_data["Season"] == season) & (player_data["Team"] == team),
            "Position",
        ].unique().tolist()
    )
    position = st.selectbox("Position", positions_for_team, key="opportunity_position")

    report = engine.simulate_swaps(season=season, team=team, position=position, top_n=st.session_state["top_n"], policy=policy)
    baseline = report["baseline"]
    top_swaps = report["top_swaps"].copy()
    all_swaps = report["all_swaps"].copy()
    incumbent_vulnerability = report["incumbent_vulnerability"].copy()

    st.markdown('<div class="section-label">Opportunities</div>', unsafe_allow_html=True)
    st.subheader(f"Opportunity scan | {team} {season} | {position}")
    st.markdown(
        '<div class="page-note">Use this page to find realistic same-position upgrades, then compare probability lift, expected value, and salary impact.</div>',
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(2)
    render_kpi_card(kpi_cols[0], "Baseline probability", format_percent(baseline["baseline_prob"]), "Current postseason probability before any simulated swap")
    render_kpi_card(kpi_cols[1], "Baseline EV", format_currency(baseline["baseline_ev_usd"]), "Current postseason value under the selected business assumption")

    position_players = sorted(all_swaps["incumbent_name"].dropna().unique().tolist())
    view_mode = st.radio("View mode", options=["All players at this position", "Target one player"], horizontal=True)
    selected_player = None
    if view_mode == "Target one player" and position_players:
        selected_player = st.selectbox("Player to evaluate", position_players, key="opportunity_player")
        top_swaps = top_swaps[top_swaps["incumbent_name"] == selected_player].reset_index(drop=True)
        all_swaps = all_swaps[all_swaps["incumbent_name"] == selected_player].reset_index(drop=True)
        incumbent_vulnerability = incumbent_vulnerability[incumbent_vulnerability["incumbent_name"] == selected_player].reset_index(drop=True)

    summary_cols = st.columns(3, gap="large")
    if not top_swaps.empty:
        best_net = top_swaps.sort_values("net_value_usd", ascending=False).iloc[0]
        best_prob = top_swaps.sort_values("delta_prob", ascending=False).iloc[0]
        render_summary_card(
            summary_cols[0],
            "Best value swap",
            f"{best_net['incumbent_name']} -> {best_net['candidate_name']}",
            f"Adds {format_currency(best_net['net_value_usd'])} of net value with a {best_net['delta_prob']:.2%} probability lift.",
        )
        render_summary_card(
            summary_cols[1],
            "Biggest playoff lift",
            f"{best_prob['delta_prob']:.2%}",
            f"{best_prob['incumbent_name']} -> {best_prob['candidate_name']} creates the largest raw postseason-probability gain.",
        )
    else:
        render_summary_card(summary_cols[0], "Best value swap", "No valid swaps", "The current filters leave no eligible candidate set.")
        render_summary_card(summary_cols[1], "Biggest playoff lift", "No valid swaps", "Relax the candidate filters or choose another position.")

    if not incumbent_vulnerability.empty:
        vuln_sorted = incumbent_vulnerability.sort_values("best_net_value_usd", ascending=False)
        vuln_row = vuln_sorted.iloc[0]
        render_summary_card(
            summary_cols[2],
            "Most vulnerable incumbent",
            vuln_row["incumbent_name"],
            f"Best available replacement is worth {format_currency(vuln_row['best_net_value_usd'])} under the current assumptions.",
        )
    else:
        render_summary_card(summary_cols[2], "Most vulnerable incumbent", "Unavailable", "No incumbent vulnerability estimates were produced for this context.")

    display_swaps = top_swaps[
        [
            "incumbent_name",
            "candidate_name",
            "candidate_team",
            "delta_prob",
            "delta_ev_usd",
            "delta_salary_usd",
            "net_value_usd",
            "candidate_total_minutes_projected",
        ]
    ].copy()
    display_swaps["Recommendation"] = pd.cut(
        display_swaps["net_value_usd"],
        bins=[-float("inf"), 0, 250000, float("inf")],
        labels=["Avoid", "Marginal", "Strong"],
    )
    display_swaps = display_swaps.rename(
        columns={
            "incumbent_name": "Incumbent",
            "candidate_name": "Candidate",
            "candidate_team": "Candidate team",
            "Recommendation": "Recommendation",
            "delta_prob": "Delta probability",
            "delta_ev_usd": "Delta EV (USD)",
            "delta_salary_usd": "Delta salary (USD)",
            "net_value_usd": "Net value (USD)",
            "candidate_total_minutes_projected": "Projected minutes",
        }
    )
    st.markdown("### Opportunity visuals")
    chart_cols = st.columns(2)
    scatter = (
        alt.Chart(top_swaps.copy())
        .mark_circle(size=70)
        .encode(
            x=alt.X("delta_salary_usd:Q", title="Delta salary (USD)"),
            y=alt.Y("delta_ev_usd:Q", title="Delta EV (USD)"),
            color=alt.Color(
                "net_value_usd:Q",
                title="Net value (USD)",
                scale=alt.Scale(range=[MUTED, SECONDARY_GOLD, PRIMARY_GOLD]),
            ),
            tooltip=[
                alt.Tooltip("incumbent_name:N", title="Incumbent"),
                alt.Tooltip("candidate_name:N", title="Candidate"),
                alt.Tooltip("candidate_team:N", title="Candidate team"),
                alt.Tooltip("delta_prob:Q", title="Delta prob", format=".4%"),
                alt.Tooltip("delta_ev_usd:Q", title="Delta EV", format=",.0f"),
                alt.Tooltip("delta_salary_usd:Q", title="Delta salary", format=",.0f"),
                alt.Tooltip("net_value_usd:Q", title="Net value", format=",.0f"),
            ],
        )
        .properties(title="EV gain vs salary change", height=260)
    )
    chart_cols[0].altair_chart(themed_altair(scatter), width="stretch")

    bar_data = top_swaps.head(10).copy()
    bar_data["label"] = bar_data["incumbent_name"] + " -> " + bar_data["candidate_name"]
    bar = (
        alt.Chart(bar_data)
        .mark_bar(color=SECONDARY_GOLD, stroke="black", strokeWidth=0.5)
        .encode(
            x=alt.X("net_value_usd:Q", title="Net value (USD)"),
            y=alt.Y("label:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("label:N", title="Swap"),
                alt.Tooltip("net_value_usd:Q", title="Net value", format=",.0f"),
                alt.Tooltip("delta_prob:Q", title="Delta prob", format=".4%"),
                alt.Tooltip("delta_salary_usd:Q", title="Delta salary", format=",.0f"),
            ],
        )
        .properties(title="Top swaps by net value", height=260)
    )
    chart_cols[1].altair_chart(themed_altair(bar), width="stretch")

    st.markdown("### Ranked swap table")
    st.dataframe(
        display_swaps.style.format(
            {
                "Delta probability": "{:.4%}",
                "Delta EV (USD)": "${:,.0f}",
                "Delta salary (USD)": "${:,.0f}",
                "Net value (USD)": "${:,.0f}",
                "Projected minutes": "{:,.1f}",
            }
        ),
        width="stretch",
    )

    st.markdown("### Position vulnerability")
    vuln_display = incumbent_vulnerability.rename(
        columns={
            "incumbent_name": "Incumbent",
            "incumbent_salary": "Salary",
            "incumbent_total_minutes": "Total minutes",
            "best_net_value_usd": "Best replacement net value (USD)",
        }
    )
    st.dataframe(
        vuln_display.style.format(
            {
                "Salary": "${:,.0f}",
                "Total minutes": "{:,.1f}",
                "Best replacement net value (USD)": "${:,.0f}",
            }
        ),
        width="stretch",
    )

    download_cols = st.columns(2)
    download_cols[0].download_button(
        "Download top swaps CSV",
        data=make_download_bytes(top_swaps),
        file_name=f"top_swaps_{team}_{season}_{position}.csv",
        mime="text/csv",
    )
    download_cols[1].download_button(
        "Download full swap universe CSV",
        data=make_download_bytes(all_swaps),
        file_name=f"all_swaps_{team}_{season}_{position}.csv",
        mime="text/csv",
    )


def render_threats_page(engine: PostseasonSwapEngine, player_data: pd.DataFrame):
    if not hasattr(engine, "analyze_threats"):
        st.warning("This backend does not include Threats analysis yet.")
        return

    season = st.session_state["season"]
    team = st.session_state["team"]
    policy = build_policy()
    threat_report = engine.analyze_threats(season=season, team=team, top_n=st.session_state["top_n"], policy=policy)

    threat_positions = ["All positions"] + sorted(
        player_data.loc[
            (player_data["Season"] == season) & (player_data["Team"] == team),
            "Position",
        ].unique().tolist()
    )
    threat_position_filter = st.selectbox(
        "Optional position filter",
        threat_positions,
        key="threat_position_filter",
        help="Threat results are computed across the full roster first, then narrowed for display.",
    )

    value_threats = filter_by_position(threat_report["value_threats"], threat_position_filter, "source_position")
    top_competitive_swaps = filter_by_position(threat_report["top_competitive_swaps"], threat_position_filter, "source_position").head(st.session_state["top_n"])
    full_competitive_threats = filter_by_position(threat_report["competitive_threats"], threat_position_filter, "source_position")
    top_value_swaps = filter_by_position(threat_report["top_value_swaps"], threat_position_filter, "source_position").head(st.session_state["top_n"])
    player_contributions = filter_by_position(threat_report["player_contributions"], threat_position_filter, "position")

    st.markdown('<div class="section-label">Threats</div>', unsafe_allow_html=True)
    st.subheader(f"Threat scan | {team} {season}")
    st.markdown(
        '<div class="page-note">Use this page to see which players are most valuable to keep. Competitive threat measures another team\'s probability gain; value threat converts that gain into net expected value.</div>',
        unsafe_allow_html=True,
    )

    threat_summary_cols = st.columns(3, gap="large")
    if not top_competitive_swaps.empty:
        top_comp = top_competitive_swaps.sort_values("delta_prob", ascending=False).iloc[0]
        render_summary_card(
            threat_summary_cols[0],
            "Top competitive threat",
            top_comp["source_player"],
            f"{top_comp['target_team']} gains {top_comp['delta_prob']:.2%} in postseason probability by taking him.",
        )
    else:
        render_summary_card(
            threat_summary_cols[0],
            "Top competitive threat",
            "Unavailable",
            "No competitive threat scenarios survived the current filters.",
        )

    if not top_value_swaps.empty:
        top_value = top_value_swaps.sort_values("net_value_usd", ascending=False).iloc[0]
        render_summary_card(
            threat_summary_cols[1],
            "Top value threat",
            top_value["source_player"],
            f"{top_value['target_team']} gains {format_currency(top_value['net_value_usd'])} of net value under the current assumptions.",
        )
    else:
        render_summary_card(
            threat_summary_cols[1],
            "Top value threat",
            "Unavailable",
            "No value threat scenarios survived the current filters.",
        )

    if not player_contributions.empty:
        top_contributor = player_contributions.sort_values("probability_contribution", ascending=False).iloc[0]
        render_summary_card(
            threat_summary_cols[2],
            "Most important current player",
            top_contributor["player_name"],
            f"Estimated to contribute {top_contributor['probability_contribution']:.2%} of team postseason probability versus an average same-position replacement.",
        )
    else:
        render_summary_card(
            threat_summary_cols[2],
            "Most important current player",
            "Unavailable",
            "Contribution estimates were not available for this context.",
        )

    threat_tabs = st.tabs(["Competitive", "Value"])

    with threat_tabs[0]:
        st.caption(
            "Contribution is measured against an average same-position external replacement. "
            "Threat columns populate only when another team improves under the current filters."
        )
        competitive_columns = (
            full_competitive_threats.sort_values("delta_prob", ascending=False)
            .drop_duplicates("source_player")
            .loc[:, ["source_player", "target_team", "replaced_player", "delta_prob"]]
            .rename(
                columns={
                    "source_player": "player_name",
                    "target_team": "best_threatened_team",
                    "replaced_player": "best_replaced_player",
                    "delta_prob": "best_threat_delta_probability",
                }
            )
        )
        competitive_contribution = player_contributions.merge(competitive_columns, on="player_name", how="left")
        competitive_contribution["best_threatened_team"] = competitive_contribution["best_threatened_team"].fillna("No eligible threat")
        competitive_contribution["best_replaced_player"] = competitive_contribution["best_replaced_player"].fillna("No eligible threat")
        competitive_contribution["best_threat_delta_probability"] = competitive_contribution["best_threat_delta_probability"].fillna(0.0)
        threat_summary = competitive_contribution[
            [
                "player_name",
                "position",
                "probability_contribution",
                "best_threatened_team",
                "best_replaced_player",
                "best_threat_delta_probability",
                "ev_contribution_usd",
                "surplus_value_usd",
                "salary",
                "total_minutes",
            ]
        ].rename(
            columns={
                "player_name": "Player",
                "position": "Position",
                "probability_contribution": "Probability contribution",
                "best_threatened_team": "Top threatened team",
                "best_replaced_player": "Replaced player",
                "best_threat_delta_probability": "Best threat delta probability",
                "ev_contribution_usd": "EV contribution (USD)",
                "surplus_value_usd": "Estimated surplus value (USD)",
                "salary": "Salary",
                "total_minutes": "Total minutes",
            }
        )
        st.dataframe(
            threat_summary.style.format(
                {
                    "Probability contribution": "{:.4%}",
                    "Best threat delta probability": "{:.4%}",
                    "EV contribution (USD)": "${:,.0f}",
                    "Estimated surplus value (USD)": "${:,.0f}",
                    "Salary": "${:,.0f}",
                    "Total minutes": "{:,.1f}",
                }
            ),
            width="stretch",
        )
        if not top_competitive_swaps.empty:
            competitive_chart = top_competitive_swaps.head(10).copy()
            competitive_chart["label"] = competitive_chart["source_player"] + " -> " + competitive_chart["target_team"]
            chart = (
                alt.Chart(competitive_chart)
                .mark_bar(color=SECONDARY_GOLD, stroke="black", strokeWidth=0.5)
                .encode(
                    x=alt.X("delta_prob:Q", title="Probability lift for other team"),
                    y=alt.Y("label:N", sort="-x", title=None),
                    tooltip=[
                        alt.Tooltip("source_player:N", title="Our player"),
                        alt.Tooltip("target_team:N", title="Threatened team"),
                        alt.Tooltip("delta_prob:Q", title="Delta prob", format=".4%"),
                    ],
                )
                .properties(title="Top competitive threat swaps", height=260)
            )
            st.altair_chart(themed_altair(chart), width="stretch")

    with threat_tabs[1]:
        st.info("Value threats are highly sensitive to the postseason value assumption in Search Settings.")
        value_summary = value_threats[
            ["source_player", "source_position", "target_team", "replaced_player", "delta_prob", "delta_ev_usd", "delta_salary_usd", "net_value_usd"]
        ].rename(
            columns={
                "source_player": "Our player",
                "source_position": "Position",
                "target_team": "Threatened team",
                "replaced_player": "Replaced player",
                "delta_prob": "Threat delta probability",
                "delta_ev_usd": "Threat delta EV (USD)",
                "delta_salary_usd": "Threat delta salary (USD)",
                "net_value_usd": "Threat net value (USD)",
            }
        )
        st.dataframe(
            value_summary.style.format(
                {
                    "Threat delta probability": "{:.4%}",
                    "Threat delta EV (USD)": "${:,.0f}",
                    "Threat delta salary (USD)": "${:,.0f}",
                    "Threat net value (USD)": "${:,.0f}",
                }
            ),
            width="stretch",
        )
        if not top_value_swaps.empty:
            threat_map = (
                alt.Chart(top_value_swaps.copy())
                .mark_circle(size=85, opacity=0.9, stroke="black", strokeWidth=0.6)
                .encode(
                    x=alt.X("delta_prob:Q", title="Probability gain for other team"),
                    y=alt.Y("net_value_usd:Q", title="Net EV gain for other team (USD)"),
                    color=alt.Color(
                        "source_position:N",
                        title="Position",
                        scale=alt.Scale(domain=["PG", "SG", "SF", "PF", "C"], range=[PRIMARY_GOLD, SECONDARY_GOLD, MUTED, "#C9C9C9", "#5F5F5F"]),
                    ),
                    tooltip=[
                        alt.Tooltip("source_player:N", title="Our player"),
                        alt.Tooltip("target_team:N", title="Threatened team"),
                        alt.Tooltip("delta_prob:Q", title="Delta probability", format=".4%"),
                        alt.Tooltip("net_value_usd:Q", title="Net value", format=",.0f"),
                    ],
                )
                .properties(title="Threat map: competitive impact vs value impact", height=255)
            )
            chart = (
                alt.Chart(top_value_swaps.head(10))
                .mark_circle(size=70)
                .encode(
                    x=alt.X("delta_salary_usd:Q", title="Salary change for other team (USD)"),
                    y=alt.Y("net_value_usd:Q", title="Net EV gain for other team (USD)"),
                    color=alt.Color(
                        "source_position:N",
                        title="Position",
                        scale=alt.Scale(domain=["PG", "SG", "SF", "PF", "C"], range=[PRIMARY_GOLD, SECONDARY_GOLD, MUTED, "#C9C9C9", "#5F5F5F"]),
                    ),
                    tooltip=[
                        alt.Tooltip("source_player:N", title="Our player"),
                        alt.Tooltip("target_team:N", title="Threatened team"),
                        alt.Tooltip("net_value_usd:Q", title="Net value", format=",.0f"),
                    ],
                )
                .properties(title="Top value threat swaps", height=260)
            )
            threat_chart_cols = st.columns(2, gap="large")
            threat_chart_cols[0].altair_chart(themed_altair(threat_map), width="stretch")
            threat_chart_cols[1].altair_chart(themed_altair(chart), width="stretch")


def main():
    init_state()

    engine = load_engine(
        st.session_state["feature_set_name"],
        st.session_state["postseason_value_usd"],
        BACKEND_CACHE_VERSION,
    )
    render_header_and_settings(engine)
    render_top_navigation()

    team_season = engine.team_season.copy()
    player_data = engine.player_data.copy()
    available_seasons = sorted(team_season["Season"].unique().tolist())
    if st.session_state["season"] not in available_seasons:
        st.session_state["season"] = "'15-'16" if "'15-'16" in available_seasons else available_seasons[-1]
    teams_for_season = sorted(team_season.loc[team_season["Season"] == st.session_state["season"], "Team"].unique().tolist())
    if st.session_state["team"] not in teams_for_season:
        st.session_state["team"] = "IND" if "IND" in teams_for_season else teams_for_season[0]

    if st.session_state["active_page"] in {"Threats", "Opportunities"}:
        render_context_rail(engine)

    if st.session_state["active_page"] == "Model":
        render_model_page(engine)
    elif st.session_state["active_page"] == "Threats":
        render_threats_page(engine, player_data)
    else:
        render_opportunities_page(engine, player_data)

    st.caption(
        "Interpretation note: SwapIQ is a probabilistic decision-support tool. Net value and surplus value depend on the postseason value assumption, and all scenarios use role-scaled same-position logic."
    )


main()
