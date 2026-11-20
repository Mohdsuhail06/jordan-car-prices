"""Cleaning rules: decide which raw listings are trustworthy enough to learn from.

Every rule here comes from a finding in notebooks/01_eda.ipynb.
"""
import pandas as pd

from jocars.config import DATA_YEAR, TARGET

MAX_PRICE = 200_000      # >= this: placeholders such as 999,999 / 1,000,000 / 7,000,000
MIN_PRICE = 1_000        # <= this: "call me" prices (306 listings are exactly 1,000)
RECENT_AGE = 6           # a car this young (or younger) ...
RECENT_MIN_PRICE = 3_000  # ... is never really sold below this: it's a down payment
OLDEST_YEAR = 1965       # value used for "Older than 1970"


def parse_year(year: pd.Series) -> pd.Series:
    """Turn the text year column into numbers ('Older than 1970' -> 1965)."""
    return pd.to_numeric(year.replace("Older than 1970", str(OLDEST_YEAR)), errors="coerce")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with only plausible, unique, priced listings."""
    out = df.dropna(how="all").dropna(subset=[TARGET]).drop_duplicates()

    price = out[TARGET]
    age = DATA_YEAR - parse_year(out["year"])
    plausible = (price > MIN_PRICE) & (price < MAX_PRICE)
    down_payment = (age <= RECENT_AGE) & (price < RECENT_MIN_PRICE)

    return out[plausible & ~down_payment].reset_index(drop=True)
