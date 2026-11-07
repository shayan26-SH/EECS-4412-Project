from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # pyright: ignore[reportMissingModuleSource]

from sklearn.cluster import KMeans # pyright: ignore[reportMissingModuleSource]
from sklearn.compose import ColumnTransformer # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import ( # type: ignore
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier



BASE = Path(__file__).resolve().parent
DATA_DIR = BASE.parent / "Task2"
OUTPUT = BASE / "results"
OUTPUT.mkdir(exist_ok=True)

NUMERIC = [
    "HHSIZE",
    "DRVRCNT",
    "WRKCOUNT",
    "YOUNGCHILD",
    "HHFAMINC",
]

CATEGORICAL = ["HOMEOWN", "URBAN"]
FEATURES = NUMERIC + CATEGORICAL


# ---------- Load the existing Task 2 data ----------

train = pd.read_csv(
    DATA_DIR / "task2_train.csv",
    dtype={"HOUSEID": str, "VEHICLE_CLASS": str},
)

test = pd.read_csv(
    DATA_DIR / "task2_test.csv",
    dtype={"HOUSEID": str, "VEHICLE_CLASS": str},
)

# HOUSEID and VEHICLE_CLASS are retained for reporting only.
# Neither is included in the clustering inputs.
# HHVEHCNT is also excluded because it defines VEHICLE_CLASS.

for name, frame in [("training", train), ("test", test)]:
    for column in FEATURES:
        frame[column] = pd.to_numeric(
            frame[column].astype(str).str.lstrip("'"),
            errors="raise",
        )

    if frame[FEATURES].isna().any().any():
        raise ValueError(f"Missing predictor values in {name} data.")

    if (frame[FEATURES] < 0).any().any():
        raise ValueError(
            f"Negative survey codes remain in {name} data."
        )

    if frame["VEHICLE_CLASS"].isna().any():
        raise ValueError(f"Missing class labels in {name} data.")


def make_preprocessor():
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            CATEGORICAL,
        ),
    ])


# ---------- Part 1: Clustering and SSE ----------

# Use all households for the descriptive clustering analysis.
all_data = pd.concat([train, test], ignore_index=True)

cluster_preprocessor = make_preprocessor()
X_all = cluster_preprocessor.fit_transform(all_data[FEATURES])

sse_results = []

for k in [3, 4, 5]:
    model = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=42,
    )

    clusters = model.fit_predict(X_all)

    sse_results.append({
        "k": k,
        "SSE": model.inertia_,
    })

    # Save each household's cluster membership.
    clustered = all_data.copy()
    clustered["CLUSTER"] = clusters
    clustered.to_csv(
        OUTPUT / f"households_k{k}.csv",
        index=False,
    )

    # Numeric means in original units.
    profiles = clustered.groupby("CLUSTER")[NUMERIC].mean()
    profiles.insert(
        0,
        "households",
        clustered.groupby("CLUSTER").size(),
    )
    profiles.to_csv(OUTPUT / f"numeric_profiles_k{k}.csv")

    # Category proportions within each cluster.
    for column in CATEGORICAL:
        proportions = pd.crosstab(
            clustered["CLUSTER"],
            clustered[column],
            normalize="index",
        )
        proportions.to_csv(
            OUTPUT / f"{column}_proportions_k{k}.csv"
        )

    # Compare cluster membership with vehicle ownership classes.
    pd.crosstab(
        clustered["CLUSTER"],
        clustered["VEHICLE_CLASS"],
    ).to_csv(OUTPUT / f"cluster_vs_class_k{k}.csv")


sse_table = pd.DataFrame(sse_results)
sse_table.to_csv(OUTPUT / "sse_comparison.csv", index=False)

best_row = sse_table.loc[sse_table["SSE"].idxmin()]

print("\nSSE RESULTS")
print(sse_table.to_string(index=False))
print(f"\nLowest observed SSE: k = {int(best_row['k'])}")

plt.figure(figsize=(7, 4))
plt.plot(
    sse_table["k"],
    sse_table["SSE"],
    marker="o",
)
plt.xticks([3, 4, 5])
plt.xlabel("Number of clusters (k)")
plt.ylabel("Within-cluster SSE")
plt.title("K-means: SSE for k = 3, 4, and 5")
plt.tight_layout()
plt.savefig(OUTPUT / "sse_comparison.png", dpi=200)
plt.close()


# ---------- Part 2: Held-out classification comparison ----------

# For this comparison, fit preprocessing and models on training
# data only. Test households remain unseen during fitting.
comparison_preprocessor = make_preprocessor()
X_train = comparison_preprocessor.fit_transform(train[FEATURES])
X_test = comparison_preprocessor.transform(test[FEATURES])

y_train = train["VEHICLE_CLASS"]
y_test = test["VEHICLE_CLASS"]

comparison_results = []

for k in [3, 4, 5]:
    model = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=42,
    )

    train_clusters = model.fit_predict(X_train)
    test_clusters = model.predict(X_test)

    # Cluster numbers have no inherent relationship to class labels.
    # Map each cluster to its most common TRAINING class.
    mapping_data = pd.DataFrame({
        "cluster": train_clusters,
        "vehicle_class": y_train.to_numpy(),
    })

    cluster_to_class = mapping_data.groupby(
        "cluster"
    )["vehicle_class"].agg(
        lambda values: values.value_counts().idxmax()
    )

    predictions = cluster_to_class.loc[test_clusters].to_numpy()

    comparison_results.append({
        "method": f"K-means k={k}, training-majority mapping",
        "accuracy": accuracy_score(y_test, predictions),
        "macro_F1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "ARI": adjusted_rand_score(y_test, test_clusters),
        "NMI": normalized_mutual_info_score(
            y_test,
            test_clusters,
        ),
    })

    pd.crosstab(
        pd.Series(test_clusters, name="cluster"),
        pd.Series(
            y_test.to_numpy(),
            name="actual_vehicle_class",
        ),
    ).to_csv(OUTPUT / f"test_cluster_vs_class_k{k}.csv")


# These are reproducible baselines, not your group's final
# Task 2 results unless the same settings are used there.
classifiers = [
    (
        "Decision tree baseline",
        DecisionTreeClassifier(
            criterion="gini",
            max_depth=10,
            random_state=42,
        ),
    ),
    ("Gaussian Naive Bayes baseline", GaussianNB()),
]

for name, classifier in classifiers:
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    comparison_results.append({
        "method": name,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_F1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "ARI": np.nan,
        "NMI": np.nan,
    })

comparison_table = pd.DataFrame(comparison_results)
comparison_table.to_csv(
    OUTPUT / "test_comparison.csv",
    index=False,
)

print("\nHELD-OUT TEST COMPARISON")
print(comparison_table.to_string(index=False))
print(f"\nOutput folder: {OUTPUT}")