import altair as alt
import json
from pathlib import Path

import pandas as pd
import streamlit as st

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


def filter_by_position(df: pd.DataFrame, position_filter: str, column_name: str) -> pd.DataFrame:
    if df.empty or position_filter == "All positions":
        return df.copy()
    return df[df[column_name] == position_filter].reset_index(drop=True)


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
        "Use Opportunities to look for inbound upgrades. Use Threats to see which of your players "
        "most help other teams or contribute to your own baseline probability."
    )
    info_cols = st.columns(3)
    readme_bytes = read_binary_if_exists(readme_path)
    if methodology_colab_url:
        info_cols[0].link_button(
            "Open full methodology in Colab",
            methodology_colab_url,
            width="stretch",
        )
    else:
        info_cols[0].button("Open full methodology in Colab", disabled=True, width="stretch")
    if judge_colab_url:
        info_cols[1].link_button(
            "Open judge notebook in Colab",
            judge_colab_url,
            width="stretch",
        )
    else:
        info_cols[1].button("Open judge notebook in Colab", disabled=True, width="stretch")
    if readme_bytes:
        info_cols[2].download_button(
            "Download project README",
            data=readme_bytes,
            file_name=readme_path.name,
            mime="text/markdown",
            width="stretch",
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
    top_n = st.slider("Top rows to display", min_value=5, max_value=50, value=15, step=5)
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
validation = engine.validation

available_seasons = sorted(team_season["Season"].unique().tolist())
default_season = "'15-'16"
season_index = available_seasons.index(default_season) if default_season in available_seasons else max(len(available_seasons) - 1, 0)
season = st.selectbox("Season", available_seasons, index=season_index)

teams_for_season = sorted(team_season.loc[team_season["Season"] == season, "Team"].unique().tolist())
default_team = "IND"
team_index = teams_for_season.index(default_team) if default_team in teams_for_season else 0
team = st.selectbox("Team", teams_for_season, index=team_index)

section = st.radio(
    "Section",
    options=["Opportunities", "Threats"],
    horizontal=True,
    help="Opportunities looks for same-position inbound swaps we might want. Threats looks at which of our players most improve other teams or our own baseline probability.",
)

policy = SwapPolicy(
    min_games=min_games,
    min_total_minutes=float(min_total_minutes),
    allow_multi_team_replacements=allow_multi_team,
    strict_position_match=True,
)

metric_cols = st.columns(5)
metric_cols[0].metric("Model LOSO-CV accuracy", f"{validation['accuracy']:.3f}")
metric_cols[1].metric("Model LOSO-CV F1", f"{validation['f1']:.3f}")
metric_cols[2].metric("Model AUC", f"{validation['auc']:.3f}" if "auc" in validation else "N/A")
metric_cols[3].metric("Model Brier score", f"{validation['brier']:.3f}" if "brier" in validation else "N/A")
metric_cols[4].metric("Model feature count", str(len(engine.model_features)))

with st.expander("Model details", expanded=False):
    st.write("Feature set:", feature_set_name)
    st.write("Features:", engine.model_features)
    st.write("LOSO-CV confusion matrix:", [[validation["tn"], validation["fp"]], [validation["fn"], validation["tp"]]])
    st.write("Policy defaults:", policy)

if section == "Opportunities":
    positions_for_team = sorted(
        player_data.loc[
            (player_data["Season"] == season) & (player_data["Team"] == team),
            "Position",
        ].unique().tolist()
    )
    position = st.selectbox("Position", positions_for_team)

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
    top_swaps = report["top_swaps"].copy()
    all_swaps = report["all_swaps"].copy()
    incumbent_vulnerability = report["incumbent_vulnerability"].copy()

    position_players = sorted(all_swaps["incumbent_name"].dropna().unique().tolist())
    view_mode = st.radio(
        "View mode",
        options=["All players at this position", "Target one player"],
        horizontal=True,
    )
    selected_player = None
    if view_mode == "Target one player" and position_players:
        selected_player = st.selectbox("Player to evaluate", position_players)
        top_swaps = top_swaps[top_swaps["incumbent_name"] == selected_player].reset_index(drop=True)
        all_swaps = all_swaps[all_swaps["incumbent_name"] == selected_player].reset_index(drop=True)
        incumbent_vulnerability = incumbent_vulnerability[
            incumbent_vulnerability["incumbent_name"] == selected_player
        ].reset_index(drop=True)

    st.subheader(f"Opportunities | {team} {season} | {position}")

    baseline_cols = st.columns(2)
    baseline_cols[0].metric("Baseline postseason probability", format_percent(baseline["baseline_prob"]))
    baseline_cols[1].metric("Baseline EV", format_currency(baseline["baseline_ev_usd"]))

    st.info(
        "Opportunities finds same-position inbound swaps from other teams that improve our team's postseason outlook. "
        "The vulnerability table below stays position-specific and shows which incumbents are easiest to improve on under the current policy."
    )

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

    chart_cols = st.columns(2)

    scatter = (
        alt.Chart(top_swaps.copy())
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
    chart_cols[0].altair_chart(scatter, width="stretch")

    bar_data = top_swaps.head(10).copy()
    bar_data["label"] = bar_data["incumbent_name"] + " -> " + bar_data["candidate_name"]
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
    chart_cols[1].altair_chart(bar, width="stretch")

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

    if selected_player:
        st.caption(
            f"Showing only swaps that replace {selected_player}. This is the direct player-targeting view within the selected position."
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

else:
    if not hasattr(engine, "analyze_threats"):
        st.subheader(f"Threats | {team} {season}")
        st.warning(
            "This deployed backend does not include the newer Threats analysis yet. "
            "Push the updated `nba_prescriptive_backend.py` to GitHub and redeploy to enable competitive threats, value threats, and player contribution rankings."
        )
        st.stop()

    try:
        threat_report = engine.analyze_threats(
            season=season,
            team=team,
            top_n=top_n,
            policy=policy,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    threat_positions = ["All positions"] + sorted(
        player_data.loc[
            (player_data["Season"] == season) & (player_data["Team"] == team),
            "Position",
        ].unique().tolist()
    )
    threat_position_filter = st.selectbox(
        "Optional position filter",
        threat_positions,
        help="Threat results are computed across the full roster first. Use this only to narrow the display after the analysis is done.",
    )

    competitive_threats = filter_by_position(
        threat_report["competitive_threats"], threat_position_filter, "source_position"
    )
    value_threats = filter_by_position(
        threat_report["value_threats"], threat_position_filter, "source_position"
    )
    top_competitive_swaps = filter_by_position(
        threat_report["top_competitive_swaps"], threat_position_filter, "source_position"
    ).head(top_n)
    top_value_swaps = filter_by_position(
        threat_report["top_value_swaps"], threat_position_filter, "source_position"
    ).head(top_n)
    player_contributions = filter_by_position(
        threat_report["player_contributions"], threat_position_filter, "position"
    )

    st.subheader(f"Threats | {team} {season}")
    st.info(
        "Threats looks outward instead of inward. Competitive threats show which of our players would improve other teams' postseason odds the most if stolen in a same-position swap. "
        "Value threats show which of our players create the largest net EV gain for other teams, which depends materially on the postseason value assumption."
    )

    threat_tabs = st.tabs(["Competitive threats", "Value threats", "Player contribution"])

    with threat_tabs[0]:
        threat_summary = competitive_threats[
            [
                "source_player",
                "source_position",
                "target_team",
                "replaced_player",
                "delta_prob",
                "delta_ev_usd",
                "delta_salary_usd",
            ]
        ].rename(
            columns={
                "source_player": "Our player",
                "source_position": "Position",
                "target_team": "Threatened team",
                "replaced_player": "Replaced player",
                "delta_prob": "Threat delta probability",
                "delta_ev_usd": "Threat delta EV (USD)",
                "delta_salary_usd": "Threat delta salary (USD)",
            }
        )
        st.dataframe(
            threat_summary.style.format(
                {
                    "Threat delta probability": "{:.4%}",
                    "Threat delta EV (USD)": "${:,.0f}",
                    "Threat delta salary (USD)": "${:,.0f}",
                }
            ),
            width="stretch",
        )

        if not top_competitive_swaps.empty:
            competitive_chart = top_competitive_swaps.head(10).copy()
            competitive_chart["label"] = competitive_chart["source_player"] + " -> " + competitive_chart["target_team"]
            st.altair_chart(
                alt.Chart(competitive_chart)
                .mark_bar()
                .encode(
                    x=alt.X("delta_prob:Q", title="Probability lift for other team"),
                    y=alt.Y("label:N", sort="-x", title=None),
                    color=alt.Color("source_position:N", title="Position"),
                    tooltip=[
                        alt.Tooltip("source_player:N", title="Our player"),
                        alt.Tooltip("source_position:N", title="Position"),
                        alt.Tooltip("target_team:N", title="Threatened team"),
                        alt.Tooltip("replaced_player:N", title="Replaced player"),
                        alt.Tooltip("delta_prob:Q", title="Delta prob", format=".4%"),
                    ],
                )
                .properties(title="Top competitive threat swaps", height=360),
                width="stretch",
            )

    with threat_tabs[1]:
        st.warning(
            "Value threat rankings depend materially on the postseason value assumption. "
            "If you change that dollar value, these rankings can move even when the competitive ranking does not."
        )
        value_summary = value_threats[
            [
                "source_player",
                "source_position",
                "target_team",
                "replaced_player",
                "delta_prob",
                "delta_ev_usd",
                "delta_salary_usd",
                "net_value_usd",
            ]
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
            value_chart = (
                alt.Chart(top_value_swaps.head(10))
                .mark_circle(size=90)
                .encode(
                    x=alt.X("delta_salary_usd:Q", title="Salary change for other team (USD)"),
                    y=alt.Y("net_value_usd:Q", title="Net EV gain for other team (USD)"),
                    color=alt.Color("source_position:N", title="Position"),
                    tooltip=[
                        alt.Tooltip("source_player:N", title="Our player"),
                        alt.Tooltip("target_team:N", title="Threatened team"),
                        alt.Tooltip("replaced_player:N", title="Replaced player"),
                        alt.Tooltip("delta_prob:Q", title="Delta prob", format=".4%"),
                        alt.Tooltip("net_value_usd:Q", title="Net value", format=",.0f"),
                    ],
                )
                .properties(title="Top value threat swaps", height=360)
            )
            st.altair_chart(value_chart, width="stretch")

    with threat_tabs[2]:
        st.caption(
            "Player contribution is measured against an average same-position external replacement. "
            "Probability contribution is the drop in our baseline probability if that player is replaced; surplus value additionally subtracts the player's salary."
        )
        st.info(
            "Negative estimated surplus value does not mean a player is bad. It means that, under the current postseason value assumption, "
            "the player's modeled postseason contribution is not large enough to outweigh his salary cost in this simplified EV framework. "
            "Expensive stars can still rank very highly on probability contribution while looking negative on surplus value."
        )
        contribution_display = player_contributions[
            [
                "player_name",
                "position",
                "probability_contribution",
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
                "ev_contribution_usd": "EV contribution (USD)",
                "surplus_value_usd": "Estimated surplus value (USD)",
                "salary": "Salary",
                "total_minutes": "Total minutes",
            }
        )
        st.dataframe(
            contribution_display.style.format(
                {
                    "Probability contribution": "{:.4%}",
                    "EV contribution (USD)": "${:,.0f}",
                    "Estimated surplus value (USD)": "${:,.0f}",
                    "Salary": "${:,.0f}",
                    "Total minutes": "{:,.1f}",
                }
            ),
            width="stretch",
        )

        if not player_contributions.empty:
            contribution_chart = player_contributions.head(10).copy()
            st.altair_chart(
                alt.Chart(contribution_chart)
                .mark_bar()
                .encode(
                    x=alt.X("probability_contribution:Q", title="Probability contribution"),
                    y=alt.Y("player_name:N", sort="-x", title=None),
                    color=alt.Color("position:N", title="Position"),
                    tooltip=[
                        alt.Tooltip("player_name:N", title="Player"),
                        alt.Tooltip("position:N", title="Position"),
                        alt.Tooltip("probability_contribution:Q", title="Prob contribution", format=".4%"),
                        alt.Tooltip("surplus_value_usd:Q", title="Estimated surplus value", format=",.0f"),
                    ],
                )
                .properties(title="Top player contributions", height=360),
                width="stretch",
            )

    threat_download_cols = st.columns(3)
    threat_download_cols[0].download_button(
        "Download competitive threats CSV",
        data=make_download_bytes(competitive_threats),
        file_name=f"competitive_threats_{team}_{season}.csv",
        mime="text/csv",
    )
    threat_download_cols[1].download_button(
        "Download value threats CSV",
        data=make_download_bytes(value_threats),
        file_name=f"value_threats_{team}_{season}.csv",
        mime="text/csv",
    )
    threat_download_cols[2].download_button(
        "Download contribution ranking CSV",
        data=make_download_bytes(player_contributions),
        file_name=f"player_contributions_{team}_{season}.csv",
        mime="text/csv",
    )

st.caption(
    "Interpretation note: this is a probabilistic decision-support tool. "
    "Net value and surplus value depend on the postseason value assumption, and all swaps use role-scaled same-position logic."
)
