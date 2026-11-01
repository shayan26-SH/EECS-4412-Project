from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import  plot_tree
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
    X_train = encode_categories(fold_training)

    # vehicle class is the target variable
    y_train = fold_training["VEHICLE_CLASS"]

    # Create a Naive Bayes classifier.
    # the min categories are the number of unique values in each column
    # This is described on the doc
    clf = CategoricalNB(
    alpha=0.001,
    min_categories=[13, 10, 8, 6, 11, 3, 4],
    )
    clf.fit(X_train, y_train)
    
    # Print the classifier's parameters.
    print(f"Fold {fold_number} Naive Bayes classifier parameters:")
    print(clf.get_params())

     # Prepare the validation household information.
    X_validation = encode_categories(fold_validation)

    # Keep the correct answers for checking afterwards.
    y_validation = fold_validation["VEHICLE_CLASS"]

    # Ask the trained Naive Bayes classifier to predict the vehicle classes.
    predictions = clf.predict(X_validation)

    # Compare predictions with the correct answers.
    accuracy = accuracy_score(y_validation, predictions)
    validation_accuracies.append(accuracy)
    print(f"Fold {fold_number} validation accuracy: {accuracy:.2%}")

    train_predictions = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)


    print(
        f"Fold {fold_number}: "
        f"training={train_accuracy:.2%}, "
        f"validation={accuracy:.2%}"
    )


    print(f"Fold {fold_number}: Naive Bayes classifier trained")

    print(
        f"Fold {fold_number}: "
        f"{len(fold_training)} training households, "
        f"{len(fold_validation)} validation households"
    )

mean_accuracy = sum(validation_accuracies) / len(validation_accuracies)
print(f"Mean 5-fold validation accuracy: {mean_accuracy:.2%}")



output_path = SCRIPT_DIR / "Naive_bayes/task2_naive_bayes_fold5.txt"
output_path.write_text(str(clf), encoding="utf-8")

print(f"Naive Bayes classifier saved to: {output_path}")


