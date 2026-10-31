from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
SCRIPT_DIR = Path(__file__).resolve().parent

train_data = pd.read_csv(SCRIPT_DIR / "task2_train.csv")

PREDICTORS = [
    "HHSIZE",
    "DRVRCNT",
    "WRKCOUNT",
    "YOUNGCHILD",
    "HHFAMINC",
    "HOMEOWN",
    "URBAN",
]

train_data = pd.read_csv(
    SCRIPT_DIR / "task2_train.csv",
    dtype={"HOUSEID": str, "VEHICLE_CLASS": str},
)

X_train = pd.get_dummies(
    train_data[PREDICTORS],
    columns=["HOMEOWN", "URBAN"],
    dtype=int,
)

y_train = train_data["VEHICLE_CLASS"]

# Train the decision tree model with our max_depth = 10 as found as the best
tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=10,
    random_state=42,
)

tree.fit(X_train, y_train)

model_path = SCRIPT_DIR / "task2_final_tree.pkl"

with open(model_path, "wb") as f:
    pickle.dump(tree, f)

print(f"Training households: {len(train_data)}")
print(f"Tree depth: {tree.get_depth()}")
print(f"Final model saved to: {model_path}")

# Now test the model on the validation/test set

# Load the reserved test households.
test_data = pd.read_csv(
    SCRIPT_DIR / "task2_test.csv",
    dtype={"HOUSEID": str, "VEHICLE_CLASS": str},
)

# Prepare the same predictors.
X_test = pd.get_dummies(
    test_data[PREDICTORS],
    columns=["HOMEOWN", "URBAN"],
    dtype=int,
)

# Match the training columns and their order.
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0,
)

y_test = test_data["VEHICLE_CLASS"]

# Predict using the trained tree, then check the answers.
test_predictions = tree.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"Final test accuracy: {test_accuracy:.2%}")


## Classification report
fig, ax = plt.subplots(figsize=(8, 7))

# Count actual-versus-predicted classes and draw the matrix.
ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_predictions,
    labels=tree.classes_,
    cmap="Blues",
    values_format="d",
    colorbar=False,
    ax=ax,
)

ax.set_xlabel("Predicted vehicle class")
ax.set_ylabel("Actual vehicle class")
ax.set_title(
    f"Decision tree confusion matrix\n"
    f"Test accuracy: {test_accuracy:.2%}"
)

fig.tight_layout()

# Save an image for the report.
fig.savefig(
    SCRIPT_DIR / "task2_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()