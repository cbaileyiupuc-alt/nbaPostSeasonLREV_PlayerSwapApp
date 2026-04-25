import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "_NBA.xlsx"


def resolve_data_path(data_path=None):
    path = Path(data_path) if data_path is not None else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at '{path}'. Please upload '_NBA.xlsx' into the project folder "
            f"'{PROJECT_DIR}' or pass a valid file path."
        )
    return path

BASE_FEATURES_WITH_SALARY = [
    "total_salary",
    "avg_age",
    "avg_points_per_min",
    "avg_assists_per_min",
    "avg_steals_per_min",
    "avg_blocks_per_min",
    "avg_turnovers_per_min",
    "pct_minutes_returning",
]

ANALYST_FEATURES_WITH_SALARY = [
    "total_salary",
    "minutes_weighted_age",
    "assist_to_turnover_ratio",
    "top_5_minutes_share",
    "pct_minutes_returning",
    "avg_steals_per_min",
    "avg_blocks_per_min",
]

BASE_FEATURES_NO_SALARY = [
    "avg_age",
    "avg_points_per_min",
    "avg_assists_per_min",
    "avg_steals_per_min",
    "avg_blocks_per_min",
    "avg_turnovers_per_min",
    "pct_minutes_returning",
]

ANALYST_FEATURES_NO_SALARY = [
    "minutes_weighted_age",
    "assist_to_turnover_ratio",
    "top_5_minutes_share",
    "pct_minutes_returning",
    "avg_steals_per_min",
    "avg_blocks_per_min",
]

POSITION_FEATURES = [
    "pg_creation_ratio",
    "sg_size_presence",
    "sf_size_presence",
]

FEATURE_SETS_WITH_SALARY = {
    "old": BASE_FEATURES_WITH_SALARY,
    "old_plus_efg": BASE_FEATURES_WITH_SALARY + ["team_efg_pct"],
    "analyst": ANALYST_FEATURES_WITH_SALARY,
    "analyst_plus_efg": ANALYST_FEATURES_WITH_SALARY + ["team_efg_pct"],
    "positional": ["total_salary", "minutes_weighted_age", "pct_minutes_returning", "team_efg_pct"]
    + POSITION_FEATURES,
    "combined": BASE_FEATURES_WITH_SALARY
    + [
        "minutes_weighted_age",
        "assist_to_turnover_ratio",
        "top_5_minutes_share",
        "team_efg_pct",
    ]
    + POSITION_FEATURES,
    "analyst_positional": ANALYST_FEATURES_WITH_SALARY + ["team_efg_pct"] + POSITION_FEATURES,
}

FEATURE_SETS_NO_SALARY = {
    "old": BASE_FEATURES_NO_SALARY,
    "old_plus_efg": BASE_FEATURES_NO_SALARY + ["team_efg_pct"],
    "analyst": ANALYST_FEATURES_NO_SALARY,
    "analyst_plus_efg": ANALYST_FEATURES_NO_SALARY + ["team_efg_pct"],
    "positional": ["minutes_weighted_age", "pct_minutes_returning", "team_efg_pct"] + POSITION_FEATURES,
    "combined": BASE_FEATURES_NO_SALARY
    + [
        "minutes_weighted_age",
        "assist_to_turnover_ratio",
        "top_5_minutes_share",
        "team_efg_pct",
    ]
    + POSITION_FEATURES,
    "analyst_positional": ANALYST_FEATURES_NO_SALARY + ["team_efg_pct"] + POSITION_FEATURES,
}

DEFAULT_WITH_SALARY_FEATURE_SET = "analyst_positional"
DEFAULT_NO_SALARY_FEATURE_SET = "old"

FEATURES_WITH_SALARY = FEATURE_SETS_WITH_SALARY[DEFAULT_WITH_SALARY_FEATURE_SET]
FEATURES_NO_SALARY = FEATURE_SETS_NO_SALARY[DEFAULT_NO_SALARY_FEATURE_SET]

DEFAULT_POSTSEASON_VALUE_USD = 10_000_000.0
DEFAULT_MIN_GAMES = 20
DEFAULT_MIN_TOTAL_MINUTES = 400.0
ROTATION_PLAYER_MINUTES = 400.0


@dataclass
class SwapPolicy:
    min_games: int = DEFAULT_MIN_GAMES
    min_total_minutes: float = DEFAULT_MIN_TOTAL_MINUTES
    allow_multi_team_replacements: bool = False
    strict_position_match: bool = True


def parse_season_start_year(season_str):
    if isinstance(season_str, str):
        season_str = season_str.strip()
        if len(season_str) == 7 and season_str.startswith("'") and season_str[3] == "-":
            year_prefix = int(season_str[1:3])
            return 1900 + year_prefix if year_prefix >= 50 else 2000 + year_prefix
        if "-" in season_str:
            try:
                return int(season_str.split("-")[0])
            except ValueError:
                return np.nan
        if "/" in season_str:
            try:
                return int(season_str.split("/")[0])
            except ValueError:
                return np.nan
        if len(season_str) == 4 and season_str.isdigit():
            return int(season_str)
    return np.nan


def safe_div(num, denom):
    return np.where(denom > 0, num / denom, 0.0)


def weighted_mean(x, w):
    w_sum = np.sum(w)
    return np.sum(x * w) / w_sum if w_sum > 0 else 0.0


def previous_team_match(prev_teams, target_team):
    if isinstance(prev_teams, (list, tuple, set)):
        return int(target_team in prev_teams)
    return 0


def safe_ratio(num, denom):
    return num / denom if denom and denom > 0 else 0.0


def weighted_mean_for_position(player_rows, position, value_series):
    subset = player_rows[player_rows["Position"] == position].copy()
    if subset.empty:
        return 0.0
    return weighted_mean(value_series.loc[subset.index], subset["total_minutes_est"])


def positional_ratio(player_rows, position, numerator_col, denominator_col):
    subset = player_rows[player_rows["Position"] == position].copy()
    if subset.empty:
        return 0.0
    numerator = (subset[numerator_col] * subset["total_minutes_est"]).sum()
    denominator = (subset[denominator_col] * subset["total_minutes_est"]).sum()
    return safe_ratio(numerator, denominator)


def summarize_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate_pred": float(np.mean(y_pred)),
        "positive_rate_actual": float(np.mean(y_true)),
    }


def normalize_by_season(team_season, features):
    team_season_norm = team_season.copy()
    season_stats = {}

    for season, idx in team_season.groupby("Season").groups.items():
        subset = team_season.loc[idx, features].astype(float)
        means = subset.mean()
        stds = subset.std(ddof=0).replace(0, 1)

        for feature in features:
            z_col = f"z_{feature}"
            team_season_norm.loc[idx, z_col] = (team_season_norm.loc[idx, feature] - means[feature]) / stds[feature]

        season_stats[season] = {
            "means": means.to_dict(),
            "stds": stds.to_dict(),
        }

    return team_season_norm, season_stats


def normalize_with_training_stats(train_df, test_df, features):
    train_means = train_df[features].mean()
    train_stds = train_df[features].std(ddof=0).replace(0, 1)

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    for feature in features:
        z_col = f"z_{feature}"
        train_scaled[z_col] = (train_scaled[feature] - train_means[feature]) / train_stds[feature]
        test_scaled[z_col] = (test_scaled[feature] - train_means[feature]) / train_stds[feature]

    return train_scaled, test_scaled


def load_and_clean_player_data(data_path=DATA_PATH):
    resolved_data_path = resolve_data_path(data_path)
    nba = pd.read_excel(resolved_data_path)
    nba_clean = nba.drop_duplicates().copy()

    for col in ["Name", "Team", "Position"]:
        if col in nba_clean.columns:
            nba_clean[col] = nba_clean[col].astype(str).str.strip()

    nba_clean["Season"] = nba_clean["Season"].astype(str).str.strip()

    numeric_cols = [
        "Salary",
        "Games_played",
        "Games_started",
        "Minutes",
        "Points",
        "Assists",
        "Rebound_off",
        "Rebound_def",
        "Blocks",
        "Steals",
        "Turnovers",
        "Fouls",
        "FG_made",
        "FG_attempted",
        "FT_made",
        "FT_attempted",
        "3P_made",
        "3P_attempted",
        "Age",
        "Height",
        "Weight",
    ]

    nba_clean[numeric_cols] = nba_clean[numeric_cols].apply(pd.to_numeric, errors="coerce")
    nba_clean["Postseason"] = pd.to_numeric(nba_clean["Postseason"], errors="coerce")
    nba_clean["start_year"] = nba_clean["Season"].apply(parse_season_start_year)

    nba_clean = nba_clean.dropna(subset=["Team", "Season", "Name", "start_year"]).copy()
    nba_clean = nba_clean[nba_clean["start_year"] >= 2005].copy()
    team_postseason = (
        nba_clean.groupby(["Season", "Team"])["Postseason"]
        .max()
        .reset_index()
        .rename(columns={"Postseason": "team_postseason"})
    )
    nba_clean = (
        nba_clean.sort_values(
            ["Name", "Season", "Team", "Position", "Games_played", "Minutes"],
            ascending=[True, True, True, True, False, False],
        )
        .drop_duplicates(subset=["Name", "Season", "Team", "Position"], keep="first")
        .copy()
    )
    nba_clean = nba_clean.drop(columns=["Postseason"]).merge(team_postseason, on=["Season", "Team"], how="left")
    nba_clean["Postseason"] = nba_clean["team_postseason"].fillna(0)
    nba_clean = nba_clean.drop(columns=["team_postseason"])

    nba_clean["Season"] = nba_clean["Season"].astype(str)
    nba_clean["total_minutes_est"] = nba_clean["Games_played"] * nba_clean["Minutes"]
    nba_clean["points_per_min"] = safe_div(nba_clean["Points"], nba_clean["Minutes"])
    nba_clean["assists_per_min"] = safe_div(nba_clean["Assists"], nba_clean["Minutes"])
    nba_clean["steals_per_min"] = safe_div(nba_clean["Steals"], nba_clean["Minutes"])
    nba_clean["blocks_per_min"] = safe_div(nba_clean["Blocks"], nba_clean["Minutes"])
    nba_clean["turnovers_per_min"] = safe_div(nba_clean["Turnovers"], nba_clean["Minutes"])
    nba_clean["rebound_total_per_min"] = safe_div(nba_clean["Rebound_off"] + nba_clean["Rebound_def"], nba_clean["Minutes"])
    nba_clean["fg_made_per_min"] = safe_div(nba_clean["FG_made"], nba_clean["Minutes"])
    nba_clean["fg_attempted_per_min"] = safe_div(nba_clean["FG_attempted"], nba_clean["Minutes"])
    nba_clean["three_made_per_min"] = safe_div(nba_clean["3P_made"], nba_clean["Minutes"])
    nba_clean["three_attempted_per_min"] = safe_div(nba_clean["3P_attempted"], nba_clean["Minutes"])
    nba_clean["ft_made_per_min"] = safe_div(nba_clean["FT_made"], nba_clean["Minutes"])
    nba_clean["ft_attempted_per_min"] = safe_div(nba_clean["FT_attempted"], nba_clean["Minutes"])

    nba_clean["salary_80"] = nba_clean.groupby("Season")["Salary"].transform(lambda x: x.quantile(0.8))
    nba_clean["star_player"] = (nba_clean["Salary"] >= nba_clean["salary_80"]).astype(int)

    season_order = sorted(nba_clean["Season"].unique())
    season_map = dict(zip(season_order[1:], season_order[:-1]))
    nba_clean["prev_season"] = nba_clean["Season"].map(season_map).astype(str)

    prev_team_map = (
        nba_clean[["Name", "Season", "Team"]]
        .rename(columns={"Season": "prev_season", "Team": "prev_team"})
        .groupby(["Name", "prev_season"])["prev_team"]
        .agg(lambda s: sorted(set(t for t in s if pd.notna(t))))
        .reset_index()
        .rename(columns={"prev_team": "prev_teams"})
    )
    prev_team_map["prev_season"] = prev_team_map["prev_season"].astype(str)

    nba_clean = nba_clean.merge(prev_team_map, on=["Name", "prev_season"], how="left")
    nba_clean["prev_teams"] = nba_clean["prev_teams"].apply(lambda x: x if isinstance(x, list) else [])
    nba_clean["returning_same_team"] = nba_clean.apply(
        lambda row: previous_team_match(row["prev_teams"], row["Team"]),
        axis=1,
    )

    team_counts = (
        nba_clean.groupby(["Name", "Season"])["Team"]
        .nunique()
        .rename("team_count_in_season")
        .reset_index()
    )
    nba_clean = nba_clean.merge(team_counts, on=["Name", "Season"], how="left")
    nba_clean["is_multi_team_season"] = (nba_clean["team_count_in_season"] > 1).astype(int)

    return nba_clean


def aggregate_team_features(player_rows, target_team):
    total_minutes = player_rows["total_minutes_est"].sum()
    returning_flags = player_rows["prev_teams"].apply(lambda x: previous_team_match(x, target_team))
    returning_minutes = player_rows.loc[returning_flags == 1, "total_minutes_est"].sum()
    team_assists_est = (player_rows["assists_per_min"] * player_rows["total_minutes_est"]).sum()
    team_turnovers_est = (player_rows["turnovers_per_min"] * player_rows["total_minutes_est"]).sum()
    team_points_est = (player_rows["points_per_min"] * player_rows["total_minutes_est"]).sum()
    team_fg_made_est = (player_rows["fg_made_per_min"] * player_rows["total_minutes_est"]).sum()
    team_fg_attempted_est = (player_rows["fg_attempted_per_min"] * player_rows["total_minutes_est"]).sum()
    team_three_made_est = (player_rows["three_made_per_min"] * player_rows["total_minutes_est"]).sum()
    team_ft_attempted_est = (player_rows["ft_attempted_per_min"] * player_rows["total_minutes_est"]).sum()
    sorted_minutes = player_rows["total_minutes_est"].fillna(0).sort_values(ascending=False)

    top_5_minutes_share = safe_ratio(sorted_minutes.head(5).sum(), total_minutes)
    assist_to_turnover_ratio = safe_ratio(team_assists_est, team_turnovers_est)
    minutes_weighted_age = weighted_mean(player_rows["Age"], player_rows["total_minutes_est"])
    pg_creation_ratio = positional_ratio(player_rows, "PG", "assists_per_min", "turnovers_per_min")
    size_signal = player_rows["Height"] * player_rows["Weight"]
    sg_size_presence = weighted_mean_for_position(player_rows, "SG", size_signal)
    sf_size_presence = weighted_mean_for_position(player_rows, "SF", size_signal)

    team_efg_pct = safe_ratio(team_fg_made_est + 0.5 * team_three_made_est, team_fg_attempted_est)

    return pd.Series(
        {
            "total_salary": player_rows["Salary"].sum(),
            "avg_age": player_rows["Age"].mean(),
            "minutes_weighted_age": minutes_weighted_age,
            "avg_points_per_min": weighted_mean(player_rows["points_per_min"], player_rows["total_minutes_est"]),
            "avg_assists_per_min": weighted_mean(player_rows["assists_per_min"], player_rows["total_minutes_est"]),
            "avg_steals_per_min": weighted_mean(player_rows["steals_per_min"], player_rows["total_minutes_est"]),
            "avg_blocks_per_min": weighted_mean(player_rows["blocks_per_min"], player_rows["total_minutes_est"]),
            "avg_turnovers_per_min": weighted_mean(player_rows["turnovers_per_min"], player_rows["total_minutes_est"]),
            "assist_to_turnover_ratio": assist_to_turnover_ratio,
            "top_5_minutes_share": top_5_minutes_share,
            "pg_creation_ratio": pg_creation_ratio,
            "sg_size_presence": sg_size_presence,
            "sf_size_presence": sf_size_presence,
            "pct_minutes_returning": returning_minutes / total_minutes if total_minutes > 0 else 0.0,
            "team_efg_pct": team_efg_pct,
        }
    )


def build_team_season_table(player_data):
    team_season = (
        player_data.groupby(["Season", "Team"])
        .apply(
            lambda df: pd.Series(
                {
                    "postseason": int(df["Postseason"].max()),
                    **aggregate_team_features(df, df.name[1]).to_dict(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return team_season


def fit_predict(train_df, test_df, features):
    train_scaled, test_scaled = normalize_with_training_stats(train_df, test_df, features)
    x_cols = [f"z_{feature}" for feature in features]

    X_train = sm.add_constant(train_scaled[x_cols], has_constant="add")
    y_train = train_scaled["postseason"].astype(int)
    X_test = sm.add_constant(test_scaled[x_cols], has_constant="add")

    model = sm.Logit(y_train, X_train).fit(disp=False, maxiter=200)
    train_probs = model.predict(X_train)
    test_probs = model.predict(X_test)
    return model, train_probs, test_probs, x_cols


def run_in_sample(team_season, features):
    model, probs, _, x_cols = fit_predict(team_season, team_season, features)
    metrics = summarize_predictions(team_season["postseason"].astype(int), probs)
    metrics["n_obs"] = int(len(team_season))
    metrics["pseudo_r2"] = float(model.prsquared)
    metrics["aic"] = float(model.aic)
    metrics["features"] = x_cols
    metrics["coefficients"] = {k: float(v) for k, v in model.params.items()}
    metrics["pvalues"] = {k: float(v) for k, v in model.pvalues.items()}
    return metrics


def run_loso_cv(team_season, features):
    seasons = sorted(team_season["Season"].unique())
    fold_rows = []
    all_probs = []

    for season in seasons:
        train_df = team_season[team_season["Season"] != season].copy()
        test_df = team_season[team_season["Season"] == season].copy()

        model, _, test_probs, _ = fit_predict(train_df, test_df, features)
        fold_rows.append(
            {
                "season": season,
                "n_test": int(len(test_df)),
                "aic_train": float(model.aic),
                "pseudo_r2_train": float(model.prsquared),
            }
        )

        fold_output = test_df[["Season", "Team", "postseason"]].copy()
        fold_output["pred_prob"] = test_probs
        all_probs.append(fold_output)

    cv_preds = pd.concat(all_probs, ignore_index=True)
    metrics = summarize_predictions(cv_preds["postseason"].astype(int), cv_preds["pred_prob"])
    metrics["n_obs"] = int(len(cv_preds))
    metrics["n_folds"] = int(len(seasons))
    metrics["folds"] = fold_rows
    return metrics


class PostseasonSwapEngine:
    def __init__(
        self,
        data_path=DATA_PATH,
        postseason_value_usd=DEFAULT_POSTSEASON_VALUE_USD,
        feature_set_name=DEFAULT_WITH_SALARY_FEATURE_SET,
    ):
        self.data_path = resolve_data_path(data_path)
        self.postseason_value_usd = float(postseason_value_usd)
        self.feature_set_name = feature_set_name
        self.model_features = FEATURE_SETS_WITH_SALARY[feature_set_name]
        self.player_data = load_and_clean_player_data(self.data_path)
        self.team_season = build_team_season_table(self.player_data)
        self.team_season_norm, self.season_stats = normalize_by_season(self.team_season, self.model_features)
        x_cols = [f"z_{feature}" for feature in self.model_features]
        X = sm.add_constant(self.team_season_norm[x_cols], has_constant="add")
        y = self.team_season_norm["postseason"].astype(int)
        self.model = sm.Logit(y, X).fit(disp=False, maxiter=200)
        self.validation = run_loso_cv(self.team_season, self.model_features)
        self.model_x_cols = x_cols

    def score_team_features(self, season, features_row):
        season_stat = self.season_stats[season]
        design = {"const": 1.0}
        for feature in self.model_features:
            mean = season_stat["means"][feature]
            std = season_stat["stds"][feature]
            z_value = (features_row[feature] - mean) / std if std != 0 else 0.0
            design[f"z_{feature}"] = z_value
        design_df = pd.DataFrame([design])[["const"] + self.model_x_cols]
        return float(self.model.predict(design_df).iloc[0])

    def get_team_roster(self, season, team):
        roster = self.player_data[
            (self.player_data["Season"] == season) & (self.player_data["Team"] == team)
        ].copy()
        if roster.empty:
            raise ValueError(f"No roster rows found for team={team} season={season}.")
        return roster

    def get_candidate_pool(self, season, position, team, roster_names, policy):
        candidates = self.player_data[self.player_data["Season"] == season].copy()
        if policy.strict_position_match:
            candidates = candidates[candidates["Position"] == position]

        candidates = candidates[candidates["Name"].notna()].copy()
        candidates = candidates[~candidates["Name"].isin(roster_names)].copy()
        candidates = candidates[candidates["Team"] != team].copy()
        candidates = candidates[candidates["Games_played"].fillna(0) >= policy.min_games].copy()
        candidates = candidates[candidates["total_minutes_est"].fillna(0) >= policy.min_total_minutes].copy()

        if not policy.allow_multi_team_replacements:
            candidates = candidates[candidates["team_count_in_season"] == 1].copy()

        candidates = candidates.sort_values(
            ["Season", "Position", "total_minutes_est", "Salary"], ascending=[True, True, False, False]
        ).copy()
        return candidates

    def build_average_replacement_row(self, season, position, team, roster_names, policy):
        candidates = self.get_candidate_pool(season, position, team, roster_names, policy)
        if candidates.empty:
            return None
        numeric_cols = candidates.select_dtypes(include=[np.number]).columns.tolist()
        avg_row = candidates.iloc[0].copy()
        for col in numeric_cols:
            avg_row[col] = candidates[col].mean()
        avg_row["Name"] = f"Average {position} Replacement"
        avg_row["Team"] = team
        avg_row["Position"] = position
        avg_row["prev_teams"] = []
        avg_row["is_multi_team_season"] = 0
        avg_row["team_count_in_season"] = 1
        return avg_row

    def simulate_swaps(self, season, team, position, top_n=15, policy=None):
        if policy is None:
            policy = SwapPolicy()

        roster = self.get_team_roster(season, team)
        incumbents = roster[roster["Position"] == position].copy()
        if incumbents.empty:
            raise ValueError(f"No incumbents found for team={team} season={season} position={position}.")

        roster_names = set(roster["Name"].tolist())
        candidates = self.get_candidate_pool(season, position, team, roster_names, policy)
        if candidates.empty:
            raise ValueError("No replacement candidates remain after applying the current policy filters.")

        baseline_features = aggregate_team_features(roster, team)
        baseline_prob = self.score_team_features(season, baseline_features)
        baseline_ev_usd = baseline_prob * self.postseason_value_usd

        swap_rows = []
        incumbent_summaries = []

        for incumbent_idx, incumbent in incumbents.iterrows():
            incumbent_best_net = None

            for _, candidate in candidates.iterrows():
                simulated_roster = roster.drop(index=incumbent_idx).copy()
                simulated_row = candidate.copy()
                simulated_row["Team"] = team
                simulated_row["total_minutes_est"] = incumbent["total_minutes_est"]
                simulated_roster = pd.concat([simulated_roster, simulated_row.to_frame().T], ignore_index=True)

                new_features = aggregate_team_features(simulated_roster, team)
                new_prob = self.score_team_features(season, new_features)
                new_ev_usd = new_prob * self.postseason_value_usd

                delta_prob = new_prob - baseline_prob
                delta_ev_usd = new_ev_usd - baseline_ev_usd
                delta_salary = float(candidate["Salary"] - incumbent["Salary"])
                net_value_usd = delta_ev_usd - delta_salary

                row = {
                    "season": season,
                    "team": team,
                    "position": position,
                    "incumbent_name": incumbent["Name"],
                    "incumbent_team": incumbent["Team"],
                    "incumbent_salary": float(incumbent["Salary"]),
                    "incumbent_games": float(incumbent["Games_played"]),
                    "incumbent_total_minutes": float(incumbent["total_minutes_est"]),
                    "candidate_name": candidate["Name"],
                    "candidate_team": candidate["Team"],
                    "candidate_salary": float(candidate["Salary"]),
                    "candidate_games": float(candidate["Games_played"]),
                    "candidate_total_minutes_actual": float(candidate["total_minutes_est"]),
                    "candidate_total_minutes_projected": float(incumbent["total_minutes_est"]),
                    "candidate_multi_team_flag": int(candidate["is_multi_team_season"]),
                    "baseline_prob": baseline_prob,
                    "new_prob": new_prob,
                    "delta_prob": delta_prob,
                    "baseline_ev_usd": baseline_ev_usd,
                    "new_ev_usd": new_ev_usd,
                    "delta_ev_usd": delta_ev_usd,
                    "delta_salary_usd": delta_salary,
                    "net_value_usd": net_value_usd,
                }
                swap_rows.append(row)

                if incumbent_best_net is None or net_value_usd > incumbent_best_net:
                    incumbent_best_net = net_value_usd

            incumbent_summaries.append(
                {
                    "season": season,
                    "team": team,
                    "position": position,
                    "incumbent_name": incumbent["Name"],
                    "incumbent_salary": float(incumbent["Salary"]),
                    "incumbent_total_minutes": float(incumbent["total_minutes_est"]),
                    "best_net_value_usd": float(incumbent_best_net) if incumbent_best_net is not None else None,
                }
            )

        swaps = pd.DataFrame(swap_rows).sort_values(
            ["net_value_usd", "delta_prob", "delta_ev_usd"], ascending=[False, False, False]
        )
        incumbent_rankings = pd.DataFrame(incumbent_summaries).sort_values(
            "best_net_value_usd", ascending=False
        )

        policy_dict = {
            "min_games": policy.min_games,
            "min_total_minutes": policy.min_total_minutes,
            "allow_multi_team_replacements": policy.allow_multi_team_replacements,
            "strict_position_match": policy.strict_position_match,
        }

        return {
            "baseline": {
                "season": season,
                "team": team,
                "position": position,
                "baseline_prob": baseline_prob,
                "baseline_ev_usd": baseline_ev_usd,
                "postseason_value_usd": self.postseason_value_usd,
            },
            "policy": policy_dict,
            "validation": self.validation,
            "top_swaps": swaps.head(top_n).reset_index(drop=True),
            "all_swaps": swaps.reset_index(drop=True),
            "incumbent_vulnerability": incumbent_rankings.reset_index(drop=True),
        }

    def analyze_threats(self, season, team, top_n=15, policy=None):
        if policy is None:
            policy = SwapPolicy()

        source_roster = self.get_team_roster(season, team)
        target_teams = sorted(t for t in self.team_season.loc[self.team_season["Season"] == season, "Team"].unique() if t != team)

        threat_rows = []
        contribution_rows = []

        source_roster_names = set(source_roster["Name"].tolist())
        baseline_source_features = aggregate_team_features(source_roster, team)
        baseline_source_prob = self.score_team_features(season, baseline_source_features)

        for source_idx, source_player in source_roster.iterrows():
            player_position = source_player["Position"]
            player_name = source_player["Name"]

            avg_replacement = self.build_average_replacement_row(season, player_position, team, source_roster_names, policy)
            if avg_replacement is not None:
                replaced_roster = source_roster.drop(index=source_idx).copy()
                avg_row = avg_replacement.copy()
                avg_row["total_minutes_est"] = source_player["total_minutes_est"]
                replaced_roster = pd.concat([replaced_roster, avg_row.to_frame().T], ignore_index=True)
                replacement_features = aggregate_team_features(replaced_roster, team)
                replacement_prob = self.score_team_features(season, replacement_features)
                probability_contribution = baseline_source_prob - replacement_prob
                ev_contribution_usd = probability_contribution * self.postseason_value_usd
                surplus_value_usd = ev_contribution_usd - float(source_player["Salary"])
                contribution_rows.append(
                    {
                        "season": season,
                        "team": team,
                        "player_name": player_name,
                        "position": player_position,
                        "baseline_prob": baseline_source_prob,
                        "replacement_prob": replacement_prob,
                        "probability_contribution": probability_contribution,
                        "ev_contribution_usd": ev_contribution_usd,
                        "surplus_value_usd": surplus_value_usd,
                        "salary": float(source_player["Salary"]),
                        "total_minutes": float(source_player["total_minutes_est"]),
                    }
                )

            for target_team in target_teams:
                target_roster = self.get_team_roster(season, target_team)
                target_incumbents = target_roster[target_roster["Position"] == player_position].copy()
                if target_incumbents.empty:
                    continue

                target_baseline_features = aggregate_team_features(target_roster, target_team)
                target_baseline_prob = self.score_team_features(season, target_baseline_features)
                target_baseline_ev = target_baseline_prob * self.postseason_value_usd

                for target_idx, target_player in target_incumbents.iterrows():
                    simulated_roster = target_roster.drop(index=target_idx).copy()
                    simulated_row = source_player.copy()
                    simulated_row["Team"] = target_team
                    simulated_row["total_minutes_est"] = target_player["total_minutes_est"]
                    simulated_row["prev_teams"] = source_player["prev_teams"] if isinstance(source_player["prev_teams"], list) else []
                    simulated_roster = pd.concat([simulated_roster, simulated_row.to_frame().T], ignore_index=True)

                    new_features = aggregate_team_features(simulated_roster, target_team)
                    new_prob = self.score_team_features(season, new_features)
                    new_ev = new_prob * self.postseason_value_usd
                    delta_prob = new_prob - target_baseline_prob
                    delta_ev = new_ev - target_baseline_ev
                    delta_salary = float(source_player["Salary"] - target_player["Salary"])
                    net_value = delta_ev - delta_salary

                    threat_rows.append(
                        {
                            "season": season,
                            "source_team": team,
                            "source_player": player_name,
                            "source_position": player_position,
                            "source_salary": float(source_player["Salary"]),
                            "target_team": target_team,
                            "replaced_player": target_player["Name"],
                            "replaced_salary": float(target_player["Salary"]),
                            "projected_minutes": float(target_player["total_minutes_est"]),
                            "baseline_target_prob": target_baseline_prob,
                            "new_target_prob": new_prob,
                            "delta_prob": delta_prob,
                            "delta_ev_usd": delta_ev,
                            "delta_salary_usd": delta_salary,
                            "net_value_usd": net_value,
                        }
                    )

        threat_df = pd.DataFrame(threat_rows)
        contribution_df = pd.DataFrame(contribution_rows).sort_values(
            "probability_contribution", ascending=False
        ).reset_index(drop=True)

        if threat_df.empty:
            competitive = pd.DataFrame()
            value = pd.DataFrame()
            top_competitive_swaps = pd.DataFrame()
            top_value_swaps = pd.DataFrame()
        else:
            competitive = (
                threat_df.sort_values(["source_player", "delta_prob"], ascending=[True, False])
                .groupby("source_player", as_index=False)
                .head(1)
                .sort_values("delta_prob", ascending=False)
                .reset_index(drop=True)
            )
            value = (
                threat_df.sort_values(["source_player", "net_value_usd"], ascending=[True, False])
                .groupby("source_player", as_index=False)
                .head(1)
                .sort_values("net_value_usd", ascending=False)
                .reset_index(drop=True)
            )
            top_competitive_swaps = threat_df.sort_values("delta_prob", ascending=False).head(top_n).reset_index(drop=True)
            top_value_swaps = threat_df.sort_values("net_value_usd", ascending=False).head(top_n).reset_index(drop=True)

        return {
            "competitive_threats": competitive,
            "value_threats": value,
            "top_competitive_swaps": top_competitive_swaps,
            "top_value_swaps": top_value_swaps,
            "player_contributions": contribution_df,
        }

    def export_swap_report(self, season, team, position, output_path, top_n=15, policy=None):
        report = self.simulate_swaps(season=season, team=team, position=position, top_n=top_n, policy=policy)
        serializable = {
            "baseline": report["baseline"],
            "policy": report["policy"],
            "validation": report["validation"],
            "top_swaps": report["top_swaps"].to_dict(orient="records"),
            "incumbent_vulnerability": report["incumbent_vulnerability"].to_dict(orient="records"),
        }
        Path(output_path).write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        return serializable
