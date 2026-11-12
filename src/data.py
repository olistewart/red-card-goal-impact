"""Load the match/event data and build the game-level and period-level tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

PHASE_BINS = [0, 45, 75, 90]
PHASE_LABELS = ["Early (1–45)", "Mid (46–75)", "Late (76–90)"]


def load_data(games_path: str, events_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    games.csv  : game_id, date, competition, home_goals, away_goals
    events.csv : game_id, type ('goal' | 'red_card'), minute, side ('home' | 'away')
    """
    games = pd.read_csv(games_path)
    events = pd.read_csv(events_path)
    games["date"] = pd.to_datetime(games["date"])
    return games, events


def build_game_table(games: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (games, goals).

    games gains: red-card counts per side, first red card minute, RC side and
    phase, goals before/after the first red card (home/away/total), and the
    minutes played in each period. A goal at the same minute as the first red
    card is assigned to 'after'. goals gains an `after_rc` flag.
    """
    games = games.copy()
    goals = events[events["type"] == "goal"].copy()
    rc = events[events["type"] == "red_card"].copy()

    rc_agg = rc.groupby("game_id").agg(
        red_cards=("type", "count"),
        first_rc_minute=("minute", "min"),
        home_rc=("side", lambda x: (x == "home").sum()),
        away_rc=("side", lambda x: (x == "away").sum()),
    ).reset_index()

    games = games.merge(rc_agg, on="game_id", how="left")
    for col in ["red_cards", "home_rc", "away_rc"]:
        games[col] = games[col].fillna(0).astype(int)
    games["has_red_card"] = (games["red_cards"] > 0).astype(int)
    # first_rc_minute stays NaN for games without a red card (intentional)

    goals = goals.merge(games[["game_id", "first_rc_minute", "has_red_card"]], on="game_id", how="left")
    goals["after_rc"] = (goals["has_red_card"] == 1) & (goals["minute"] >= goals["first_rc_minute"])

    def split(side: str, after: bool) -> np.ndarray:
        return (goals[(goals["side"] == side) & (goals["after_rc"] == after)]
                .groupby("game_id").size()
                .reindex(games["game_id"]).fillna(0).astype(int).values)

    games["home_goals_before_rc"] = split("home", False)
    games["away_goals_before_rc"] = split("away", False)
    games["home_goals_after_rc"] = split("home", True)
    games["away_goals_after_rc"] = split("away", True)
    games["total_goals_before_rc"] = games["home_goals_before_rc"] + games["away_goals_before_rc"]
    games["total_goals_after_rc"] = games["home_goals_after_rc"] + games["away_goals_after_rc"]

    # Consistency check against the goal totals in games.csv
    mismatches = ((games["home_goals_before_rc"] + games["home_goals_after_rc"] != games["home_goals"]).sum()
                  + (games["away_goals_before_rc"] + games["away_goals_after_rc"] != games["away_goals"]).sum())
    if mismatches:
        print(f"WARNING: {mismatches} goal-count mismatches between games and events tables.")
    games["total_goals"] = games["total_goals_before_rc"] + games["total_goals_after_rc"]

    # Minutes available in each period (exposure)
    games["minutes_before_rc"] = games["first_rc_minute"].fillna(90).clip(upper=90)
    games["minutes_after_rc"] = (90 - games["first_rc_minute"]).clip(lower=0).fillna(0)

    games["rc_side"] = "none"
    games.loc[(games["home_rc"] > 0) & (games["away_rc"] == 0), "rc_side"] = "home"
    games.loc[(games["away_rc"] > 0) & (games["home_rc"] == 0), "rc_side"] = "away"
    games.loc[(games["home_rc"] > 0) & (games["away_rc"] > 0), "rc_side"] = "both"
    games["rc_phase"] = pd.cut(games["first_rc_minute"], bins=PHASE_BINS, labels=PHASE_LABELS)

    return games, goals


def build_periods(games: pd.DataFrame) -> pd.DataFrame:
    """
    One row per period: 'no_rc' (whole no-RC game), 'before_rc' and 'after_rc'
    (the two halves of each RC game), with goals, minutes and the post-RC flag.
    """
    no_rc = games[games["has_red_card"] == 0]
    rc_games = games[games["has_red_card"] == 1]

    def rows(df, period, goals, minutes, side):
        return pd.DataFrame({
            "game_id": df["game_id"].values,
            "competition": df["competition"].values,
            "rc_side": side,
            "period": period,
            "goals": goals,
            "minutes": minutes,
            "first_rc_min": df["first_rc_minute"].values,
        })

    periods = pd.concat([
        rows(no_rc, "no_rc", no_rc["total_goals"].values, 90, "none"),
        rows(rc_games, "before_rc", rc_games["total_goals_before_rc"].values,
             rc_games["minutes_before_rc"].values, rc_games["rc_side"].values),
        rows(rc_games, "after_rc", rc_games["total_goals_after_rc"].values,
             rc_games["minutes_after_rc"].values, rc_games["rc_side"].values),
    ], ignore_index=True)

    periods["rate"] = periods["goals"] / periods["minutes"].clip(lower=1)
    periods["is_after_rc"] = (periods["period"] == "after_rc").astype(int)
    periods["log_minutes"] = np.log(periods["minutes"].clip(lower=1))
    return periods
