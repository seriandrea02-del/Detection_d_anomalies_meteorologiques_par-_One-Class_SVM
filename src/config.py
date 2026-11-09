from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "weatherHistory.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DATE_COLUMN = "Formatted Date"
FEATURES = [
    "Temperature (C)",
    "Apparent Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Visibility (km)",
    "Pressure (millibars)",
]

TRAIN_RATIO = 0.80
TRAIN_SAMPLE_SIZE = 6000
RANDOM_STATE = 42

# Paramètres initiaux du modèle : à comparer lors d'une étude complémentaire.
NU = 0.05
GAMMA = "scale"
