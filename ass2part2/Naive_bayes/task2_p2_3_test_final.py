from pathlib import Path
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

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


## Classification report
fig, ax = plt.subplots(figsize=(8, 7))

# Count actual-versus-predicted classes and draw the matrix.
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=clf.classes_,
    cmap="Blues",
    values_format="d",
    colorbar=False,
    ax=ax,
)

ax.set_xlabel("Predicted vehicle class")
ax.set_ylabel("Actual vehicle class")
ax.set_title(
    f"Naive Bayes confusion matrix\n"
    f"Test accuracy: {test_accuracy:.2%}"
)

fig.tight_layout()

# Save an image for the report.
fig.savefig(
    SCRIPT_DIR / "task2_naive_bayes_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

probabilities = clf.predict_proba(X_test)

print("\nClass probabilities for the first five households:")
print(
    pd.DataFrame(
        probabilities[:5],
        columns=clf.classes_,
    )
)


# ROC curve
# one vehicle vs rest


plt.figure(figsize=(8, 6))
for column, positive_class in enumerate(clf.classes_):
    # Current class versus all other classes.
    binary_y_test = (y_test == positive_class).astype(int)

    fpr, tpr, thresholds = roc_curve(
        binary_y_test,
        probabilities[:, column],
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"Class {positive_class} vs rest (AUC = {roc_auc:.3f})",
    )

plt.plot([0, 1], [0, 1], "k--", label="Random guess")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Naive Bayes: one-vs-rest ROC curves")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    SCRIPT_DIR / "task2_naive_bayes_roc.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()