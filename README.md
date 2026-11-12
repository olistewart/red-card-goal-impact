# Do Red Cards Lead to More Goals?

A football data analysis project using a Poisson GLM to estimate how a red card changes the rate of goal scoring.

## Question

> Based on the data in this dataset, do red cards seem to lead to more goals? We mean more goals in general here - we might expect the team with the red card to get fewer goals than we would have otherwise expected, and their opponent to get more, but the question is whether these effects combined lead to more goals overall.

The data covered 20,370 games across 10 European competitions over five seasons (2018-19 to 2022-23): a fixtures file (`games.csv`) and a file of goal and red card times (`events.csv`).

## Data

**The data is not included in this repository.** It was provided externally for a task and it is not my right to make it public. The code expects the following schema, so the analysis can be reproduced on any comparable dataset:

| File | Columns |
|---|---|
| `games.csv` | `game_id`, `date`, `competition`, `home_goals`, `away_goals` |
| `events.csv` | `game_id`, `type` (`goal` or `red_card`), `minute`, `side` (`home` or `away`) |

`src/synthetic_data.py` generates a synthetic dataset in this format (with a known planted effect) so the pipeline can be run end to end. Its output has no relation to the real data and is for testing only.

## Approach

1. **Raw comparison.** Total goals in games with and without a red card (Mann-Whitney U test).
2. **Timing and side.** Red card phase (early 1-45', mid 46-75', late 76-90') and which side received the card.
3. **Goal rates.** Goals per minute before and after the first red card, to account for the time available in each period.
4. **Dynamic baseline.** A goal rate curve (5-minute bins) from no-red-card games, capturing the natural rise in scoring through a match. The rate ratio *R* is the observed post-red-card goal count divided by the count expected from this curve over the same window.
5. **Poisson GLM.** Goals in each period are modelled as Poisson with a log link and an offset of the log dynamically expected goals, so *R* = e<sup>β</sup> on the post-red-card indicator. Three specifications:
   - **S1:** flat offset (log minutes), competition fixed effect
   - **S2:** dynamic offset, competition fixed effect
   - **S3:** dynamic offset, plus red card side, game state of the carded team and red card minute

## Key results

| Model | R | 95% CI | p-value | AIC |
|---|---|---|---|---|
| S1 (flat offset) | 1.309 | [1.264, 1.356] | <0.001 | 80,784 |
| S2 (dynamic offset) | 1.095 | [1.057, 1.134] | <0.001 | 80,650 |
| S3 (dynamic + covariates) | 1.087 | [1.031, 1.147] | 0.002 | 80,471 |

- **Yes, but the effect is marginal.** After controlling for natural game progression, the goal rate rises by roughly 9% after a red card, robust across specifications. The flat-baseline estimate (~31%) is largely explained by goals becoming more frequent late in matches regardless.
- **The two sides largely offset.** The carded team scores less and its opponent more, with a small net positive effect. This is why raw whole-game totals look almost flat (2.70 goals with a red card vs 2.64 without, p = 0.13).
- **Timing matters.** A red card adds between about 0.02 and 0.20 expected goals, with early cards having the largest impact.
- **League variation.** R ranges from 1.06 (2. Bundesliga) to 1.21 (Primera).

This is relevant to in-play over/under markets: an early red card is more likely to tip a borderline line than a late one.

**Limitations:** event times are capped at 45 and 90 minutes, which inflates late-phase rates; timing is anchored to the first card (multi-card games are about 2% of the data); and there are no pre-match team strength ratings. See the report for details and future work.

## Repository structure

```
├── notebooks/oliverstewart_redcard_analysis.ipynb   # original exploratory analysis, with figures
├── reports/RC_Goal_Analysis.pdf                     # full write-up
├── src/
│   ├── data.py             # load data, build game- and period-level tables
│   ├── baseline.py         # dynamic baseline curve and expected goals in a window
│   ├── rates.py            # raw comparison, rate ratios by phase / side / competition
│   ├── glm.py              # Poisson GLM (S1-S3), scoreline covariate, expected extra goals
│   ├── run_analysis.py     # runs everything and prints the key results
│   └── synthetic_data.py   # synthetic data generator for testing
├── data/                   # place games.csv and events.csv here (not included)
├── requirements.txt
└── LICENSE
```

## Usage

```bash
pip install -r requirements.txt

# with your own data in the schema above
python -m src.run_analysis --games data/games.csv --events data/events.csv

# or test the pipeline on synthetic data
python -m src.synthetic_data --out data/synthetic --n-games 8000 --effect 1.09
python -m src.run_analysis --games data/synthetic/games.csv --events data/synthetic/events.csv
```

The notebook was written in Google Colab. To run it elsewhere, replace the Google Drive mount and the two `shutil.copy` lines in the data loading cell with local paths to `games.csv` and `events.csv`.

## Reference

Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football betting market. *Journal of the Royal Statistical Society: Series C*, 46(2), 265-280.

## Licence

Code and report are released under the [MIT Licence](LICENSE), © 2026 Oliver Stewart. The licence does not cover the underlying data, which is not included and is not mine to share.
