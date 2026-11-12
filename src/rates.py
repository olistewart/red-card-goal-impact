"""Descriptive goal-rate analysis: raw comparison, rate ratios by phase / side / competition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats

from .baseline import baseline_curve, expected_goals_in_window
from .data import PHASE_LABELS


def raw_comparison(games: pd.DataFrame) -> dict:
    """Total goals in RC vs no-RC games (crude baseline; includes pre-RC goals)."""
    rc = games.loc[games["has_red_card"] == 1, "total_goals"]
    no_rc = games.loc[games["has_red_card"] == 0, "total_goals"]
    u, p = stats.mannwhitneyu(rc, no_rc, alternative="two-sided")
    return {"mean_no_rc": no_rc.mean(), "mean_rc": rc.mean(), "u": u, "p_value": p}


def aggregate_rates(games: pd.DataFrame) -> dict:
    """Pooled goals/min for no-RC games, pre-RC and post-RC periods, plus the naive (flat-baseline) ratio."""
    no_rc = games[games["has_red_card"] == 0]
    rc = games[games["has_red_card"] == 1].dropna(subset=["first_rc_minute"])
    rate_no_rc = no_rc["total_goals"].sum() / (len(no_rc) * 90)
    rate_before = rc["total_goals_before_rc"].sum() / rc["minutes_before_rc"].clip(lower=1).sum()
    rate_after = rc["total_goals_after_rc"].sum() / rc["minutes_after_rc"].clip(lower=1).sum()
    rate_baseline = ((no_rc["total_goals"].sum() + rc["total_goals_before_rc"].sum())
                     / (len(no_rc) * 90 + rc["minutes_before_rc"].clip(lower=1).sum()))
    return {"no_rc": rate_no_rc, "before_rc": rate_before, "after_rc": rate_after,
            "naive_ratio": rate_after / rate_baseline}


def overall_rate_ratio(games: pd.DataFrame, curve: pd.Series) -> dict:
    """R = observed post-RC rate / dynamically expected post-RC rate (descriptive)."""
    rc = games[games["has_red_card"] == 1].dropna(subset=["first_rc_minute"])
    exp_goals = sum(expected_goals_in_window(m, 90, curve) for m in rc["first_rc_minute"])
    post_min = rc["minutes_after_rc"].clip(lower=1).sum()
    obs_rate = rc["total_goals_after_rc"].sum() / post_min
    exp_rate = exp_goals / post_min
    return {"obs_rate": obs_rate, "exp_rate": exp_rate, "rate_ratio": obs_rate / exp_rate}


def phase_rate_ratios(games: pd.DataFrame, curve: pd.Series) -> pd.DataFrame:
    """Rate ratio by RC phase (early / mid / late)."""
    phase_windows = dict(zip(PHASE_LABELS, [(1, 45), (46, 75), (76, 90)]))
    rows = []
    for label, (t0, t1) in phase_windows.items():
        g = games[(games["has_red_card"] == 1)
                  & (games["first_rc_minute"] >= t0) & (games["first_rc_minute"] <= t1)]
        if g.empty:
            continue
        minutes = g["minutes_after_rc"].clip(lower=1).sum()
        obs = g["total_goals_after_rc"].sum() / minutes
        exp = sum(expected_goals_in_window(m, 90, curve) for m in g["first_rc_minute"]) / minutes
        rows.append({"phase": label, "n": len(g), "obs_rate": obs, "exp_rate": exp, "rate_ratio": obs / exp})
    return pd.DataFrame(rows)


def side_rate_ratios(games: pd.DataFrame, goals: pd.DataFrame) -> dict:
    """
    Post-RC rate ratios for the team that received the card and for its
    opponent, against side-specific (home / away) baseline curves.
    Only games with a card for exactly one side are used.
    """
    no_rc_ids = games.loc[games["has_red_card"] == 0, "game_id"]
    curves = {s: baseline_curve(goals, no_rc_ids, side=s) for s in ("home", "away")}

    g = games[(games["has_red_card"] == 1) & games["rc_side"].isin(["home", "away"])
              & (games["minutes_after_rc"] > 0)].copy()
    mins = g["minutes_after_rc"].clip(lower=1)
    for s in ("home", "away"):
        g[f"exp_{s}_rate"] = g["first_rc_minute"].apply(lambda m: expected_goals_in_window(m, 90, curves[s])) / mins
        g[f"obs_{s}_rate"] = g[f"{s}_goals_after_rc"] / mins

    is_home = g["rc_side"] == "home"
    obs_rc = np.where(is_home, g["obs_home_rate"], g["obs_away_rate"]).mean()
    exp_rc = np.where(is_home, g["exp_home_rate"], g["exp_away_rate"]).mean()
    obs_opp = np.where(is_home, g["obs_away_rate"], g["obs_home_rate"]).mean()
    exp_opp = np.where(is_home, g["exp_away_rate"], g["exp_home_rate"]).mean()
    return {"rc_team": obs_rc / exp_rc, "opponent": obs_opp / exp_opp}


def competition_rate_ratios(games: pd.DataFrame, goals: pd.DataFrame, min_no_rc_games: int = 20) -> pd.DataFrame:
    """Rate ratio per competition, each against its own baseline curve."""
    rows = []
    for comp in sorted(games["competition"].unique()):
        ids = games.loc[(games["competition"] == comp) & (games["has_red_card"] == 0), "game_id"].values
        if len(ids) < min_no_rc_games:
            continue
        curve = baseline_curve(goals, ids)
        g = games[(games["competition"] == comp) & (games["has_red_card"] == 1)].dropna(subset=["first_rc_minute"])
        if g.empty:
            continue
        minutes = g["minutes_after_rc"].clip(lower=1).sum()
        obs = g["total_goals_after_rc"].sum() / minutes
        exp = sum(expected_goals_in_window(m, 90, curve) for m in g["first_rc_minute"]) / minutes
        rows.append({"competition": comp, "rate_ratio": obs / exp, "n_rc_games": len(g), "n_no_rc_games": len(ids)})
    return pd.DataFrame(rows).sort_values("rate_ratio", ascending=False).reset_index(drop=True)
