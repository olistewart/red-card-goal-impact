"""
Generate SYNTHETIC games.csv / events.csv in the expected schema, so the
pipeline can be run end to end without the real (non-public) data.

The goal rate rises through the match, and is multiplied by a known factor
after a red card, so the analysis should recover roughly that factor.
The output has no relation to the real dataset; results are for testing only.

    python -m src.synthetic_data --out data/synthetic --n-games 6000 --effect 1.09
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COMPETITIONS = ["League A", "League B", "League C", "League D"]


def generate(n_games: int = 6000, effect: float = 1.09, rc_prob: float = 0.16, seed: int = 0):
    rng = np.random.default_rng(seed)
    minutes = np.arange(1, 91)
    base_rate = 0.022 + 0.03 * (minutes / 90) ** 2          # goals/min, rises through the match
    games, events = [], []

    for gid in range(1, n_games + 1):
        rc_minute = None
        if rng.random() < rc_prob:
            rc_minute = int(np.clip(rng.beta(3, 1.6) * 90, 1, 90))
        rate = base_rate.copy()
        if rc_minute is not None:
            rate[minutes >= rc_minute] *= effect
            events.append((gid, "red_card", rc_minute, rng.choice(["home", "away"])))
        n = rng.poisson(rate)
        h = a = 0
        for minute, k in zip(minutes, n):
            for _ in range(k):
                side = "home" if rng.random() < 0.55 else "away"
                events.append((gid, "goal", int(minute), side))
                h += side == "home"
                a += side == "away"
        games.append((gid, pd.Timestamp("2020-08-01") + pd.Timedelta(days=int(gid // 10)),
                      COMPETITIONS[gid % len(COMPETITIONS)], h, a))

    games = pd.DataFrame(games, columns=["game_id", "date", "competition", "home_goals", "away_goals"])
    events = pd.DataFrame(events, columns=["game_id", "type", "minute", "side"])
    return games, events


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-games", type=int, default=6000)
    ap.add_argument("--effect", type=float, default=1.09)
    args = ap.parse_args(argv)

    games, events = generate(args.n_games, args.effect)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    games.to_csv(out / "games.csv", index=False)
    events.to_csv(out / "events.csv", index=False)
    print(f"Wrote synthetic data to {out}/ (planted post-RC rate ratio = {args.effect})")


if __name__ == "__main__":
    main()
