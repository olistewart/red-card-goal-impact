"""Dynamic baseline goal-rate curve (natural game progression) from no-RC games."""

from __future__ import annotations

import numpy as np
import pandas as pd

BIN_WIDTH = 5


def baseline_curve(goals: pd.DataFrame, game_ids, bin_width: int = BIN_WIDTH,
                   side: str | None = None) -> pd.Series:
    """
    Goals per minute in each `bin_width`-minute bin, over the given (no-RC)
    games. Minute 90 is clipped into the final bin (events are capped at 45/90,
    so injury-time goals pile up there). Empty bins are linearly interpolated.
    Pass side='home' or 'away' for side-specific curves.
    """
    game_ids = pd.Index(game_ids)
    g = goals[goals["game_id"].isin(game_ids)].copy()
    if side is not None:
        g = g[g["side"] == side]
    g["minute_bin"] = ((g["minute"] // bin_width) * bin_width).clip(upper=90 - bin_width)

    bins = pd.Index(range(0, 90, bin_width), name="minute_bin")
    curve = g.groupby("minute_bin").size().reindex(bins, fill_value=0).astype(float)
    curve = curve / (len(game_ids) * bin_width)
    return curve.replace(0, np.nan).interpolate().bfill()


def expected_goals_in_window(t_start: float, t_end: float, curve: pd.Series,
                             bin_width: int = BIN_WIDTH) -> float:
    """Integrate the baseline curve over [t_start, t_end] -> expected goal COUNT."""
    expected = 0.0
    for bin_start, rate in curve.items():
        overlap = max(0.0, min(t_end, bin_start + bin_width) - max(t_start, bin_start))
        expected += rate * overlap
    return max(expected, 1e-9)


def get_window(row) -> tuple[float, float]:
    """(t_start, t_end) covered by a row of the periods table."""
    if row["period"] == "no_rc":
        return 0, 90
    if row["period"] == "before_rc":
        return 0, row["first_rc_min"]
    return row["first_rc_min"], 90
