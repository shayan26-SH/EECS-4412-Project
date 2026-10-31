from pathlib import Path
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score

SCRIPT_DIR = Path(__file__).resolve().parent

PREDICTORS = [
    "HHSIZE", "DRVRCNT", "WRKCOUNT", "YOUNGCHILD",
    "HHFAMINC", "HOMEOWN", "URBAN",
]

# Load the saved final decision tree.
with open(SCRIPT_DIR / "task2_final_tree.pkl", "rb") as f:
    tree = pickle.load(f)

# Load the reserved test households.
test_data = pd.read_csv(
    SCRIPT_DIR / "task2_test.csv",
    dtype={"HOUSEID": str, "VEHICLE_CLASS": str},
)

# Prepare the same predictors used during training.
X_test = pd.get_dummies(
    test_data[PREDICTORS],
    columns=["HOMEOWN", "URBAN"],
    dtype=int,
)

# Match the exact columns and order remembered by the model.
X_test = X_test.reindex(
    columns=tree.feature_names_in_,
    fill_value=0,
)

# Keep the correct labels for checking predictions.
y_test = test_data["VEHICLE_CLASS"]

# Predict using the saved model.
predictions = tree.predict(X_test)

# Compare predictions with the correct labels.
test_accuracy = accuracy_score(y_test, predictions)

print(f"Test households: {len(test_data)}")
print(f"Tree depth: {tree.get_depth()}")
print(f"Final test accuracy: {test_accuracy:.2%}")