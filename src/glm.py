"""
Poisson GLM for the post-RC rate ratio.

    log(lambda_i) = alpha_competition + beta * post_RC_i + log(mu_i)

where mu_i is the dynamically expected goal count for the period (offset), so
R = exp(beta) is the rate ratio relative to natural game progression.

    S1  flat offset  (log minutes)                 - expected to overstate R
    S2  dynamic offset                             - controls for game progression
    S3  dynamic offset + RC side, RC-team game state, RC minute
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .baseline import expected_goals_in_window, get_window


def scoreline_at_rc(games: pd.DataFrame, goals: pd.DataFrame) -> pd.DataFrame:
    """Game state (winning / drawing / losing) of the carded team at the first RC."""
    rc_games = games[games["has_red_card"] == 1].dropna(subset=["first_rc_minute"])
    before = goals.merge(rc_games[["game_id"]], on="game_id")
    before = before[before["minute"] < before["first_rc_minute"]]
    counts = before.groupby(["game_id", "side"]).size().unstack(fill_value=0)
    counts = counts.reindex(rc_games["game_id"], fill_value=0)
    for s in ("home", "away"):
        if s not in counts:
            counts[s] = 0

    out = rc_games[["game_id", "rc_side"]].set_index("game_id")
    goal_diff = counts["home"] - counts["away"]
    rc_team_diff = np.select([out["rc_side"] == "home", out["rc_side"] == "away"],
                             [goal_diff, -goal_diff], default=0)
    out["rc_team_status"] = np.select([rc_team_diff > 0, rc_team_diff == 0], ["winning", "drawing"], "losing")
    return out.reset_index()[["game_id", "rc_team_status"]]


def prepare_glm_data(periods: pd.DataFrame, games: pd.DataFrame, goals: pd.DataFrame,
                     curve: pd.Series) -> pd.DataFrame:
    d = periods[periods["minutes"] > 0].copy()
    d["expected_goals_dynamic"] = d.apply(lambda r: expected_goals_in_window(*get_window(r), curve), axis=1)
    d["log_expected_dynamic"] = np.log(d["expected_goals_dynamic"].clip(lower=1e-6))

    d = d.merge(scoreline_at_rc(games, goals), on="game_id", how="left")
    d["rc_team_status"] = d["rc_team_status"].fillna("no_rc")
    d["rc_min_filled"] = d["first_rc_min"].fillna(0)
    d["has_rc_min"] = d["first_rc_min"].notna().astype(int)
    return d[d["expected_goals_dynamic"] > 0].copy()


def fit_models(d: pd.DataFrame) -> dict:
    fam = sm.families.Poisson()
    s1 = smf.glm("goals ~ is_after_rc + C(competition)", data=d, family=fam, offset=d["log_minutes"]).fit()
    s2 = smf.glm("goals ~ is_after_rc + C(competition)", data=d, family=fam, offset=d["log_expected_dynamic"]).fit()
    s3 = smf.glm("goals ~ is_after_rc + C(rc_side) + C(rc_team_status) + rc_min_filled + has_rc_min + C(competition)",
                 data=d, family=fam, offset=d["log_expected_dynamic"]).fit()
    return {"S1 (flat offset)": s1, "S2 (dynamic offset)": s2, "S3 (dynamic + covariates)": s3}


def summarise(models: dict) -> pd.DataFrame:
    rows = []
    for label, m in models.items():
        coef = m.params["is_after_rc"]
        lo, hi = m.conf_int().loc["is_after_rc"]
        rows.append({"model": label, "R": np.exp(coef), "CI_low": np.exp(lo), "CI_high": np.exp(hi),
                     "p_value": m.pvalues["is_after_rc"], "AIC": m.aic})
    return pd.DataFrame(rows)


def expected_additional_goals(rate_ratio: float, curve: pd.Series,
                              minutes=(10, 30, 45, 60, 75, 85)) -> pd.DataFrame:
    """Expected extra goals from a red card at minute m: (R - 1) * baseline expected goals over (m, 90)."""
    rows = []
    for m in minutes:
        base = expected_goals_in_window(m, 90, curve)
        rows.append({"rc_minute": m, "remaining_min": 90 - m, "baseline_expected_goals": base,
                     "additional_goals": (rate_ratio - 1) * base})
    return pd.DataFrame(rows)
