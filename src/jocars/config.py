"""Project paths and constants, so every script and notebook agrees on them."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "cars_jordan.csv"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

TARGET = "price_value_JOD"
DATA_YEAR = 2024  # listings were collected during 2024
RANDOM_STATE = 42
