import pickle
import pandas as pd
from pathlib import Path
from sklearn.naive_bayes import CategoricalNB

SCRIPT_DIR = Path(__file__).resolve().parent.parent
train_data = pd.read_csv(SCRIPT_DIR / "task2_train.csv",
    dtype={"HOUSEID": str, "VEHICLE_CLASS": str},
)
PREDICTORS = [
    "HHSIZE",
    "DRVRCNT",
    "WRKCOUNT",
    "YOUNGCHILD",
    "HHFAMINC",
    "HOMEOWN",
    "URBAN",
]

def encode_categories(frame):
    X = frame[PREDICTORS].astype(int).copy()

    # Convert values into category IDs starting at zero.
    # HHZISE, FAMINC, URBAN ALL START AT 1, SO WE SUBTRACT 1 TO MAKE THEM START AT 0
    # HOMEOWN START AT 1, SO WE MAP 1 TO 0, 2 TO 1, AND 97 TO 2
    X["HHSIZE"] = X["HHSIZE"] - 1
    X["HHFAMINC"] = X["HHFAMINC"] - 1
    X["URBAN"] = X["URBAN"] - 1
    X["HOMEOWN"] = X["HOMEOWN"].map({1: 0, 2: 1, 97: 2})

    return X

# Prepare all training households.
X_train = encode_categories(train_data)
y_train = train_data["VEHICLE_CLASS"]

# Use the selected smoothing setting.
clf = CategoricalNB(
    alpha=0.1,
    min_categories=[13, 10, 8, 6, 11, 3, 4],
)

clf.fit(X_train, y_train)

# Save the final trained classifier.
model_path = SCRIPT_DIR / "task2_final_naive_bayes.pkl"

with open(model_path, "wb") as f:
    pickle.dump(clf, f)

print(f"Training households: {len(train_data)}")
print(f"Final model saved to: {model_path}")