# zombieAPI

Single-file Python synthetic dataset generator for API metadata (zombie/active/deprecated/orphaned endpoints).

## Run

```bash
python generate_api_dataset.py             # generate 100k records → data/zombie.csv
python generate_api_dataset.py 100000      # specify record count
```

- No dependencies beyond Python stdlib (3.14).
- RNG seeded with `42` for reproducibility.
- Sequential generation in batches of 100K files (only when total >= 1M).
- After all batch files are written, they are merged into `data/zombie.csv` and deleted.
- If interrupted mid-run, partial `data/batch_*.csv` files remain.

## Structure

- `generate_api_dataset.py` — entrypoint and only source file.
- `data/` — output directory; `zombie.csv` is the final merged result.
- `MLMODEL/` — placeholder for future ML classification work.

## Gotchas

- No `.gitignore` — `data/*.csv`, `__pycache__/`, `.DS_Store` are all untracked. (Added .gitignore ver=python)
- No tests, no linting, no typechecking configured. (To configure in the new session)
- Default 100M records is expensive (~several minutes, large CSV). (Downgraded to 100k)
- Profile distribution: Active 41.7%, Deprecated 25.0%, Orphaned 16.6%, Zombie 16.7%.
