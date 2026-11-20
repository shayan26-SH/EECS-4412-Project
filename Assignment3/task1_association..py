from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
OUTPUT = BASE / "task1_results"
OUTPUT.mkdir(exist_ok=True)

COLUMNS = [
    "HOUSEID", "HHSIZE", "DRVRCNT", "WRKCOUNT",
    "HHFAMINC", "HHVEHCNT", "URBAN",
]

raw = pd.read_csv(
    ROOT / "hhpub_202609201846.csv",
    usecols=COLUMNS,
    dtype=str,
)
raw.to_csv(OUTPUT / "raw_households.csv", index=False)

data = raw.copy()

# Convert survey codes, including apostrophe-prefixed values.
for column in COLUMNS[1:]:
    data[column] = pd.to_numeric(
        data[column].str.strip().str.lstrip("'"),
        errors="coerce",
    )

# Remove missing and invalid values while preserving valid zeros.
valid = (
    data["HOUSEID"].notna()
    & data["HHSIZE"].ge(1)
    & data["DRVRCNT"].ge(0)
    & data["WRKCOUNT"].ge(0)
    & data["HHVEHCNT"].ge(0)
    & data["HHFAMINC"].isin(range(1, 12))
    & data["URBAN"].isin([1, 2, 3, 4])
)
data = data.loc[valid].copy()

if data.empty:
    raise ValueError("No valid households remain.")

if data["HOUSEID"].duplicated().any():
    raise ValueError("Duplicate household identifiers need review.")

# Create categorical items for each household.
processed = pd.DataFrame({
    "HOUSEID": data["HOUSEID"],
    "size": pd.cut(
        data["HHSIZE"],
        [0, 1, 2, np.inf],
        labels=["size=1", "size=2", "size=3+"],
    ),
    "drivers": pd.cut(
        data["DRVRCNT"],
        [-1, 0, 1, np.inf],
        labels=["drivers=0", "drivers=1", "drivers=2+"],
    ),
    "workers": pd.cut(
        data["WRKCOUNT"],
        [-1, 0, 1, np.inf],
        labels=["workers=0", "workers=1", "workers=2+"],
    ),
    "income": pd.cut(
        data["HHFAMINC"],
        [0, 5, 7, 11],
        labels=[
            "income=below_50000",
            "income=50000_to_99999",
            "income=100000_plus",
        ],
    ),
    "vehicles": pd.cut(
        data["HHVEHCNT"],
        [-1, 0, 1, 2, np.inf],
        labels=[
            "vehicles=0", "vehicles=1",
            "vehicles=2", "vehicles=3+",
        ],
    ),
    "urban": data["URBAN"].astype(int).map({
        1: "urban=urban_area",
        2: "urban=urban_cluster",
        3: "urban=surrounded",
        4: "urban=outside",
    }),
})

processed.to_csv(OUTPUT / "processed_households.csv", index=False)

transactions = (
    processed.drop(columns="HOUSEID")
    .astype(str)
    .values.tolist()
)

encoder = TransactionEncoder()
basket = pd.DataFrame(
    encoder.fit(transactions).transform(transactions),
    columns=encoder.columns_,
)

# Enumerate every pair. Every larger itemset has support no higher
# than its pairs, so these pairs establish a safe mining threshold.
pair_supports = [
    basket[[a, b]].all(axis=1).mean()
    for a, b in combinations(basket.columns, 2)
]
positive_pairs = sorted(
    [value for value in pair_supports if value > 0],
    reverse=True,
)

if len(positive_pairs) < 5:
    raise ValueError("Fewer than five positive-support pairs exist.")

# Mine all itemsets that could enter the top five.
threshold = max(
    1 / len(basket),
    positive_pairs[4] - 1e-12,
)
frequent = fpgrowth(
    basket,
    min_support=threshold,
    use_colnames=True,
)

frequent["size"] = frequent["itemsets"].map(len)
frequent["items"] = frequent["itemsets"].map(
    lambda items: " | ".join(sorted(items))
)

top_five = (
    frequent.loc[frequent["size"] >= 2]
    .sort_values(
        ["support", "items"],
        ascending=[False, True],
    )
    .head(5)
)

itemsets = top_five[["items", "size", "support"]].copy()
itemsets["count"] = (
    itemsets["support"] * len(basket)
).round().astype(int)
itemsets.to_csv(OUTPUT / "top_five_itemsets.csv", index=False)


def support(items):
    return basket[sorted(items)].all(axis=1).mean()


# Generate every rule from the selected five itemsets.
rows = []

for itemset in top_five["itemsets"]:
    for size in range(1, len(itemset)):
        for subset in combinations(sorted(itemset), size):
            antecedent = frozenset(subset)
            consequent = itemset - antecedent

            p_a = support(antecedent)
            p_b = support(consequent)
            p_ab = support(itemset)
            confidence = p_ab / p_a

            rows.append({
                "antecedent": " & ".join(sorted(antecedent)),
                "consequent": " & ".join(sorted(consequent)),
                "support": p_ab,
                "confidence": confidence,
                "lift": p_ab / (p_a * p_b),
                "leverage": p_ab - p_a * p_b,
                "conviction": (
                    np.inf if confidence >= 1
                    else (1 - p_b) / (1 - confidence)
                ),
                "cosine": p_ab / np.sqrt(p_a * p_b),
                "jaccard": p_ab / (p_a + p_b - p_ab),
            })

rules = (
    pd.DataFrame(rows)
    .drop_duplicates(["antecedent", "consequent"])
    .sort_values(
        ["confidence", "support", "antecedent", "consequent"],
        ascending=[False, False, True, True],
    )
    .reset_index(drop=True)
)

highest = rules.head(2).copy()
highest.insert(0, "selection", "highest_confidence")

lowest = rules.iloc[2:].sort_values(
    ["confidence", "support", "antecedent", "consequent"],
    ascending=[True, False, True, True],
).head(2).copy()
lowest.insert(0, "selection", "lowest_confidence")

selected = pd.concat([highest, lowest], ignore_index=True)

rules.to_csv(OUTPUT / "all_candidate_rules.csv", index=False)
selected.to_csv(OUTPUT / "four_selected_rules.csv", index=False)
highest.to_csv(
    OUTPUT / "top_two_rules_interest_measures.csv",
    index=False,
)

print("\nFIVE MOST FREQUENT ITEMSETS")
print(itemsets.to_string(index=False))

print("\nFOUR SELECTED ASSOCIATION RULES")
print(selected.to_string(index=False))

print("\nHouseholds analysed:", len(data))
print("Results saved to:", OUTPUT)