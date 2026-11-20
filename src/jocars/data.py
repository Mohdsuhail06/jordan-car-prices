"""Loading the raw JUCars-2024 file."""
import pandas as pd

from jocars.config import RAW_CSV


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV exactly as published (the file starts with a UTF-8 BOM)."""
    return pd.read_csv(path, encoding="utf-8-sig")
