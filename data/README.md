# Data

The data used in this project was provided externally for a task and is not mine to make public, so it is not included here.

To reproduce the analysis, place `games.csv` and `events.csv` in this folder with the following schema:

| File | Columns |
|---|---|
| `games.csv` | `game_id`, `date`, `competition`, `home_goals`, `away_goals` |
| `events.csv` | `game_id`, `type` (`goal` or `red_card`), `minute`, `side` (`home` or `away`) |

CSV files in this folder are git-ignored. To test the pipeline without real data, generate a synthetic set:

```bash
python -m src.synthetic_data --out data/synthetic
```
