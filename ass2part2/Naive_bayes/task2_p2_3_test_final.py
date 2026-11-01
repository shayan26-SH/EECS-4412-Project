from pathlib import Path
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent

PREDICTORS = [
    "HHSIZE", "DRVRCNT", "WRKCOUNT", "YOUNGCHILD",
    "HHFAMINC", "HOMEOWN", "URBAN",
]

# Apply exactly the same encoding used during training.
def encode_categories(frame):
    X = frame[PREDICTORS].astype(int).copy()

    X["HHSIZE"] = X["HHSIZE"] - 1
    X["HHFAMINC"] = X["HHFAMINC"] - 1
    X["URBAN"] = X["URBAN"] - 1
    X["HOMEOWN"] = X["HOMEOWN"].map({1: 0, 2: 1, 97: 2})

    return X


# Load the final trained classifier.
with open(SCRIPT_DIR / "task2_final_naive_bayes.pkl", "rb") as f:
    clf = pickle.load(f)

# Load the same test households used for the decision tree.
test_data = pd.read_csv(
    DATA_DIR / "task2_test.csv",
    dtype={"HOUSEID": str, "VEHICLE_CLASS": str},
)

X_test = encode_categories(test_data)
y_test = test_data["VEHICLE_CLASS"]

# Predict without retraining.
predictions = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, predictions)

print(f"Test households: {len(test_data)}")
print(f"Alpha: {clf.alpha}")
print(f"Final Naive Bayes test accuracy: {test_accuracy:.2%}")