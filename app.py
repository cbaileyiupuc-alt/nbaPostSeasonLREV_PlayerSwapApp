import altair as alt
import pandas as pd
import streamlit as st
from pathlib import Path
import json

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
    page_title="NBA Postseason Swap Tool",
    page_icon="🏀",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model and cleaned roster data...")
def load_engine(feature_set_name: str, postseason_value_usd: float):
    return PostseasonSwapEngine(
        postseason_value_usd=postseason_value_usd,
        feature_set_name=feature_set_name,
    )


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


st.title("NBA Postseason Swap Tool")
st.caption(
    "Interactive same-position swap simulator using the locked logistic postseason model. "
    "Probabilities are team-level postseason probabilities used for expected value comparisons."
)

methodology_path = PROJECT_DIR / "Postseason_Model_Methodology.ipynb"
judge_notebook_path = PROJECT_DIR / "Postseason_Model_For_Judges.ipynb"
readme_path = PROJECT_DIR / "README.md"
share_links = load_share_links()
methodology_colab_url = share_links.get("methodology_colab_url")
judge_colab_url = share_links.get("judge_colab_url")
repo_url = share_links.get("repo_url")

with st.container(border=True):
    st.subheader("What this project does")
    st.write(
        "This tool estimates a team's postseason probability, simulates same-position roster swaps, "
        "and compares the change in expected postseason value against the change in salary."
    )
    st.write(
        "Use the app to explore candidate swaps. Use the notebooks to review the modeling choices, "
        "validation design, calibration results, assumptions, and limitations."
    )
    info_cols = st.columns(3)
    readme_bytes = read_binary_if_exists(readme_path)
    if methodology_colab_url:
        info_cols[0].link_button(
            "Open full methodology in Colab",
            methodology_colab_url,
            use_container_width=True,
        )
    else:
        info_cols[0].button("Open full methodology in Colab", disabled=True, use_container_width=True)
    if judge_colab_url:
        info_cols[1].link_button(
            "Open judge notebook in Colab",
            judge_colab_url,
            use_container_width=True,
        )
    else:
        info_cols[1].button("Open judge notebook in Colab", disabled=True, use_container_width=True)
    if readme_bytes:
        info_cols[2].download_button(
            "Download project README",
            data=readme_bytes,
            file_name=readme_path.name,
            mime="text/markdown",
            use_container_width=True,
        )
    if not methodology_colab_url or not judge_colab_url:
        st.info(
            "Colab links are not configured yet. After pushing this project to GitHub, create "
            "`share_links.json` from `share_links.example.json` and paste the GitHub-based Colab URLs there."
        )
    if repo_url:
        st.markdown(f"[Open project repository]({repo_url})")

with st.sidebar:
    st.header("Controls")
    feature_set_name = st.selectbox(
        "Model feature set",
        options=list(FEATURE_SETS_WITH_SALARY.keys()),
        index=list(FEATURE_SETS_WITH_SALARY.keys()).index("analyst_positional")
        if "analyst_positional" in FEATURE_SETS_WITH_SALARY
        else 0,
        help="Use the locked production model by default, or inspect alternative benchmarked feature sets.",
    )
    postseason_value_usd = st.number_input(
        "Postseason value (USD)",
        min_value=0.0,
        step=500_000.0,
        value=float(DEFAULT_POSTSEASON_VALUE_USD),
        format="%.0f",
        help="Dollar value assigned to one unit of postseason probability.",
    )
    top_n = st.slider("Top swaps to display", min_value=5, max_value=50, value=15, step=5)
    min_games = st.slider("Minimum candidate games", min_value=0, max_value=82, value=DEFAULT_MIN_GAMES, step=1)
    min_total_minutes = st.slider(
        "Minimum candidate total minutes",
        min_value=0,
        max_value=2000,
        value=int(DEFAULT_MIN_TOTAL_MINUTES),
        step=50,
    )
    allow_multi_team = st.checkbox("Allow multi-team replacement candidates", value=False)

try:
    engine = load_engine(feature_set_name, postseason_value_usd)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

team_season = engine.team_season.copy()
player_data = engine.player_data.copy()

available_seasons = sorted(team_season["Season"].unique().tolist())
default_season = "'15-'16"
season_index = available_seasons.index(default_season) if default_season in available_seasons else max(len(available_seasons) - 1, 0)
season = st.selectbox("Season", available_seasons, index=season_index)

teams_for_season = sorted(team_season.loc[team_season["Season"] == season, "Team"].unique().tolist())
default_team = "IND"
team_index = teams_for_season.index(default_team) if default_team in teams_for_season else 0
team = st.selectbox("Team", teams_for_season, index=team_index)

positions_for_team = sorted(
    player_data.loc[
        (player_data["Season"] == season) & (player_data["Team"] == team),
        "Position",
    ].unique().tolist()
)
position = st.selectbox("Position", positions_for_team)

policy = SwapPolicy(
    min_games=min_games,
    min_total_minutes=float(min_total_minutes),
    allow_multi_team_replacements=allow_multi_team,
    strict_position_match=True,
)

try:
    report = engine.simulate_swaps(
        season=season,
        team=team,
        position=position,
        top_n=top_n,
        policy=policy,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

baseline = report["baseline"]
validation = report["validation"]
top_swaps = report["top_swaps"].copy()
all_swaps = report["all_swaps"].copy()
incumbent_vulnerability = report["incumbent_vulnerability"].copy()

st.subheader(f"{team} {season} | {position} swaps")

metric_cols = st.columns(5)
metric_cols[0].metric("Baseline postseason probability", format_percent(baseline["baseline_prob"]))
metric_cols[1].metric("Baseline EV", format_currency(baseline["baseline_ev_usd"]))
metric_cols[2].metric("Model LOSO-CV accuracy", f"{validation['accuracy']:.3f}")
metric_cols[3].metric("Model LOSO-CV F1", f"{validation['f1']:.3f}")
metric_cols[4].metric("Model feature count", str(len(engine.model_features)))

with st.expander("Model details", expanded=False):
    st.write("Feature set:", feature_set_name)
    st.write("Features:", engine.model_features)
    st.write("LOSO-CV confusion matrix:", [[validation["tn"], validation["fp"]], [validation["fn"], validation["tp"]]])
    st.write("Policy:", report["policy"])

st.markdown("### Top swap recommendations")

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

display_swaps = display_swaps.rename(
    columns={
        "incumbent_name": "Incumbent",
        "candidate_name": "Candidate",
        "candidate_team": "Candidate team",
        "delta_prob": "Delta probability",
        "delta_ev_usd": "Delta EV (USD)",
        "delta_salary_usd": "Delta salary (USD)",
        "net_value_usd": "Net value (USD)",
        "candidate_total_minutes_projected": "Projected minutes",
    }
)

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
    use_container_width=True,
)

chart_cols = st.columns(2)

scatter_data = top_swaps.copy()
scatter = (
    alt.Chart(scatter_data)
    .mark_circle(size=90)
    .encode(
        x=alt.X("delta_salary_usd:Q", title="Delta salary (USD)"),
        y=alt.Y("delta_ev_usd:Q", title="Delta EV (USD)"),
        color=alt.Color("net_value_usd:Q", title="Net value (USD)", scale=alt.Scale(scheme="blueorange")),
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
    .properties(title="EV gain vs salary change", height=360)
)
chart_cols[0].altair_chart(scatter, use_container_width=True)

bar_data = top_swaps.head(10).copy()
bar_data["label"] = bar_data["incumbent_name"] + " → " + bar_data["candidate_name"]
bar = (
    alt.Chart(bar_data)
    .mark_bar()
    .encode(
        x=alt.X("net_value_usd:Q", title="Net value (USD)"),
        y=alt.Y("label:N", sort="-x", title=None),
        color=alt.Color("net_value_usd:Q", legend=None, scale=alt.Scale(scheme="teals")),
        tooltip=[
            alt.Tooltip("label:N", title="Swap"),
            alt.Tooltip("net_value_usd:Q", title="Net value", format=",.0f"),
            alt.Tooltip("delta_prob:Q", title="Delta prob", format=".4%"),
            alt.Tooltip("delta_salary_usd:Q", title="Delta salary", format=",.0f"),
        ],
    )
    .properties(title="Top 10 swaps by net value", height=360)
)
chart_cols[1].altair_chart(bar, use_container_width=True)

st.markdown("### Roster vulnerability")

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
    use_container_width=True,
)

with st.expander("Current roster at selected position", expanded=False):
    roster_position = player_data[
        (player_data["Season"] == season) & (player_data["Team"] == team) & (player_data["Position"] == position)
    ][["Name", "Salary", "Games_played", "total_minutes_est", "Height", "Weight"]].copy()
    roster_position = roster_position.rename(
        columns={
            "Name": "Player",
            "Salary": "Salary",
            "Games_played": "Games",
            "total_minutes_est": "Total minutes",
        }
    )
    st.dataframe(
        roster_position.style.format(
            {"Salary": "${:,.0f}", "Games": "{:,.0f}", "Total minutes": "{:,.1f}"}
        ),
        use_container_width=True,
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

st.caption(
    "Interpretation note: this is a probabilistic decision-support tool. "
    "Net value depends on the postseason value assumption and uses role-scaled swap logic."
)
