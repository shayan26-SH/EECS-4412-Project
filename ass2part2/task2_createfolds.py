from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
# Divide the training portion into five stratified folds.
folds = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

validation_accuracies = []

for fold_number, (train_rows, validation_rows) in enumerate(
    folds.split(train_data, train_data["VEHICLE_CLASS"]),
    start=1,
):
    fold_training = train_data.iloc[train_rows]
    fold_validation = train_data.iloc[validation_rows]

    #convert categorical variables to dummy variables
    #HOMEOWN and URBAN are categorical variables
    X_train = pd.get_dummies(
        fold_training[PREDICTORS],
        columns=["HOMEOWN", "URBAN"],
        dtype=int,
    )

    # vehicle class is the target variable
    y_train = fold_training["VEHICLE_CLASS"]

    # Create a decision tree that uses Gini to select splits.
    tree = DecisionTreeClassifier(
        criterion="gini",
        random_state=42,
        max_depth=10,
    )

    # Train the tree.
    # this tries all possible questions abnout predictorsz
    # calculates the Gini impurity before and after each possible split
    # selects the split with the greatest reduction in impurity
    #repeats this process until the tree is fully grown
    tree.fit(X_train, y_train)

     # Prepare the validation household information.
    X_validation = pd.get_dummies(
        fold_validation[PREDICTORS],
        columns=["HOMEOWN", "URBAN"],
        dtype=int,
    )

    # Match the columns and their order to the training data.
    X_validation = X_validation.reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    # Keep the correct answers for checking afterwards.
    y_validation = fold_validation["VEHICLE_CLASS"]

    # Ask the trained tree to predict the vehicle classes.
    predictions = tree.predict(X_validation)

    # Compare predictions with the correct answers.
    accuracy = accuracy_score(y_validation, predictions)
    validation_accuracies.append(accuracy)
    print(f"Fold {fold_number} validation accuracy: {accuracy:.2%}")

    train_predictions = tree.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)

    print(
        f"Fold {fold_number}: "
        f"training={train_accuracy:.2%}, "
        f"validation={accuracy:.2%}"
    )


    print(f"Fold {fold_number}: tree trained")

    print(
        f"Fold {fold_number}: "
        f"{len(fold_training)} training households, "
        f"{len(fold_validation)} validation households"
    )
    print(f"tree depth: {tree.get_depth()}")

mean_accuracy = sum(validation_accuracies) / len(validation_accuracies)
print(f"Mean 5-fold validation accuracy: {mean_accuracy:.2%}")

#plot the tree
plt.figure(figsize=(20, 10))

plot_tree(
    tree,
    feature_names=X_train.columns.tolist(),
    class_names=tree.classes_.tolist(),
    impurity=True,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9,
)

plt.title("Decision tree from fold 5 — top levels")
plt.tight_layout()
plt.show()

#print the tree
tree_text = export_text(
    tree,
    feature_names=X_train.columns.tolist(),
    max_depth=tree.get_depth(),
)

output_path = SCRIPT_DIR / "task2_tree_fold5.txt"
output_path.write_text(tree_text, encoding="utf-8")

print(f"Tree saved to: {output_path}")

# save the tree to a file
tree_path = SCRIPT_DIR / f"task2_fold_{fold_number}_tree.pkl"
with open(tree_path, "wb") as f:
    pickle.dump(tree, f)

print(f"tree saved to {tree_path}")
