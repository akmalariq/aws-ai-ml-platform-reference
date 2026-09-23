"""Synthetic demand data, so the platform runs with no external dataset.

In a real deployment this is replaced by a Glue ETL job that lands the client's
data in the raw zone; the schema here matches what the lake house, the model,
and Athena expect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_STORES = 30
DAYS = 365
START = "2024-01-01"


def generate(seed: int = 42, n_stores: int = N_STORES, days: int = DAYS) -> pd.DataFrame:
    """Return a (store, event_date, units, promo) demand frame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=days, freq="D")
    rows = []
    for store in range(1, n_stores + 1):
        base = float(80 + rng.integers(0, 160))
        promo_days = set(int(i) for i in rng.choice(days, size=int(days * 0.12), replace=False))
        for i, date in enumerate(dates):
            weekly = 1.0 + (0.35 if date.dayofweek >= 5 else 0.0)
            annual = 1.0 + 0.25 * np.sin(2 * np.pi * date.dayofyear / 365)
            promo = 1 if i in promo_days else 0
            mean = base * weekly * annual * (1.6 if promo else 1.0)
            rows.append(
                {
                    "store": store,
                    "event_date": date.date(),
                    "units": int(rng.poisson(mean)),
                    "promo": promo,
                }
            )
    return pd.DataFrame(rows)
