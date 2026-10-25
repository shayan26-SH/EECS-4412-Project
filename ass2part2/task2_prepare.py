from pathlib import Path
import pandas as pd
## count of 129696
# Locate the dataset relative to this script.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "hhpub_202609201846.csv"

PREDICTORS = [
    "HHSIZE",
    "DRVRCNT",
    "WRKCOUNT",
    "YOUNGCHILD",
    "HHFAMINC",
    "HOMEOWN",
    "URBAN",
]

# Read codes as text first so we can clean them explicitly.
households = pd.read_csv(DATA_PATH, dtype=str)

selected_columns = ["HOUSEID", "HHVEHCNT"] + PREDICTORS
data = households[selected_columns].copy()

print("Households loaded:", len(data))
print(data.head())