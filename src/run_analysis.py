"""
Run the full analysis and print the key results.

    python -m src.run_analysis --games data/games.csv --events data/events.csv
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd

from . import baseline, data, glm, rates

warnings.filterwarnings("ignore")
pd.set_option("display.width", 140)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Do red cards lead to more goals? (Poisson GLM analysis)")
    ap.add_argument("--games", default="data/games.csv")
    ap.add_argument("--events", default="data/events.csv")
    args = ap.parse_args(argv)

    games_raw, events = data.load_data(args.games, args.events)
    games, goals = data.build_game_table(games_raw, events)
    periods = data.build_periods(games)

    print(f"Games: {len(games):,} | with a red card: {games['has_red_card'].sum():,} "
          f"({100 * games['has_red_card'].mean():.1f}%) | competitions: {games['competition'].nunique()}")

    raw = rates.raw_comparison(games)
    print(f"\n1. Raw total goals - no RC: {raw['mean_no_rc']:.3f}, RC: {raw['mean_rc']:.3f} "
          f"(Mann-Whitney p = {raw['p_value']:.3f})")

    agg = rates.aggregate_rates(games)
    print(f"\n2. Goal rate (goals/min) - no RC: {agg['no_rc']:.4f}, pre-RC: {agg['before_rc']:.4f}, "
          f"post-RC: {agg['after_rc']:.4f}  | naive ratio vs flat baseline: {agg['naive_ratio']:.2f}")

    no_rc_ids = games.loc[games["has_red_card"] == 0, "game_id"]
    curve = baseline.baseline_curve(goals, no_rc_ids)
    overall = rates.overall_rate_ratio(games, curve)
    print(f"\n3. Dynamic rate ratio (observed / expected post-RC): R = {overall['rate_ratio']:.3f}")
    print(rates.phase_rate_ratios(games, curve).round(3).to_string(index=False))

    side = rates.side_rate_ratios(games, goals)
    print(f"\n4. Post-RC rate ratio - carded team: {side['rc_team']:.2f}, opponent: {side['opponent']:.2f}")

    print("\n5. Rate ratio by competition:")
    print(rates.competition_rate_ratios(games, goals).round(3).to_string(index=False))

    d = glm.prepare_glm_data(periods, games, goals, curve)
    models = glm.fit_models(d)
    results = glm.summarise(models)
    print("\n6. Poisson GLM (R = exp(beta) on the post-RC indicator):")
    print(results.round(4).to_string(index=False))

    r_final = results.iloc[-1]["R"]
    print(f"\n7. Expected additional goals from a red card (S3, R = {r_final:.3f}):")
    print(glm.expected_additional_goals(r_final, curve).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
