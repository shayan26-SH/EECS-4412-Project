from pathlib import Path
import pandas as pd
## households removed: 4398
## Households remaining: 125298

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "hhpub_202609201846.csv"

# Documented missing codes, including the apostrophe-prefixed
# versions found in our CSV.
MISSING_CODES = {
    "HHFAMINC": ["-7", "-8", "-9", "'-7", "'-8", "'-9"],
    "HOMEOWN": ["-7", "-8", "'-7", "'-8"],
}

households = pd.read_csv(
    DATA_PATH,
    dtype=str,
    na_values=MISSING_CODES,
)

# go through each row and check if data is missing
PREDICTORS = [
    "HHSIZE",
    "DRVRCNT",
    "WRKCOUNT",
    "YOUNGCHILD",
    "HHFAMINC",
    "HOMEOWN",
    "URBAN",
]

selected_columns = ["HOUSEID", "HHVEHCNT"] + PREDICTORS
data = households[selected_columns].copy()

rows_before = len(data)

# Remove households missing any selected predictor.
data = data.dropna(subset=PREDICTORS).copy()

print("Households removed:", rows_before - len(data))
print("Households remaining:", len(data))