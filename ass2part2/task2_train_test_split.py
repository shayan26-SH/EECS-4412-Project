from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent

data = pd.read_csv(
    SCRIPT_DIR / "task2_labelled_data.csv",
    dtype={"HOUSEID": str, "VEHICLE_CLASS": str},
)

# We have the class label, so exclude the original vehicle count.
data = data.drop(columns=["HHVEHCNT"])

# Create a reproducible, stratified 80/20 split.
#explained in the document
#80% is training data, 20% is test data

train_data, test_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["VEHICLE_CLASS"],
)

test_data.to_csv(SCRIPT_DIR / "task2_test.csv", index=False)
train_data.to_csv(SCRIPT_DIR / "task2_train.csv", index=False)