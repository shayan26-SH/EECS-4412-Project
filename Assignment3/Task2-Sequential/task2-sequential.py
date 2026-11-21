from pathlib import Path

import pandas as pd
from prefixspan import PrefixSpan


# Find the folder containing this script.
BASE = Path(__file__).resolve().parent

# Move up two folders to the repository root, where the source CSV is stored.
ROOT = BASE.parent.parent

# Store generated results in a folder beside this script.
OUTPUT = BASE / "task2_results"

# Create the results folder if it does not already exist.
OUTPUT.mkdir(exist_ok=True)

# Keep only the trip fields needed for this analysis.
COLUMNS = ["HOUSEID", "PERSONID", "TDTRPNUM", "TRIPPURP"]

# Read those fields as text so the original values are preserved for cleaning.
raw = pd.read_csv(
    ROOT / "trippub_202609201847.csv",
    usecols=COLUMNS,
    dtype=str,
)

# Save a copy of the selected, unprocessed fields; omit the extra row-number column.
raw.to_csv(OUTPUT / "raw_trips.csv", index=False)

# Work on a copy so the raw data stays unchanged.
trips = raw.copy()

# Convert the trip position to a number; remove whitespace and leading apostrophes first.
trips["TDTRPNUM"] = pd.to_numeric(
    trips["TDTRPNUM"].str.strip().str.lstrip("'"),
    errors="coerce",
)

# Normalize the text purpose codes while preserving them as strings.
trips["TRIPPURP"] = (
    trips["TRIPPURP"]
    .str.strip()
    .str.lstrip("'")
    .str.upper()
)

# Give the trip-purpose codes readable names.
PURPOSES = {
    "HBW": "home_based_work",
    "HBSHOP": "home_based_shopping",
    "HBSOCREC": "home_based_social_recreation",
    "HBO": "home_based_other",
    "NHB": "non_home_based",
}

# Exclude rows without household or person identifiers.
valid_ids = trips["HOUSEID"].notna() & trips["PERSONID"].notna()
trips = trips.loc[valid_ids].copy()

# A valid trip has a positive whole-number trip position and a known purpose code.
trips["valid_event"] = (
    trips["TDTRPNUM"].gt(0)
    & trips["TDTRPNUM"].mod(1).eq(0)
    & trips["TRIPPURP"].isin(PURPOSES)
)

# Mark each person's sequence as complete only if every trip in it is valid.
complete_sequence = trips.groupby(
    ["HOUSEID", "PERSONID"]
)["valid_event"].transform("all")

# Drop entire sequences containing an invalid trip, rather than creating a gap.
trips = trips.loc[complete_sequence].copy()

# A person cannot have two trips in the same trip position.
if trips.duplicated(["HOUSEID", "PERSONID", "TDTRPNUM"]).any():
    raise ValueError("Duplicate trip positions need review.")

# Order each person's trips chronologically by trip number.
trips = trips.sort_values(["HOUSEID", "PERSONID", "TDTRPNUM"])

# Replace purpose codes with readable labels.
trips["purpose"] = trips["TRIPPURP"].map(PURPOSES)

# Save the cleaned trip events, including each readable purpose label.
trips[COLUMNS + ["purpose"]].to_csv(
    OUTPUT / "processed_trip_events.csv",
    index=False,
)

# Group each person's ordered trip purposes into a sequence.
grouped = trips.groupby(
    ["HOUSEID", "PERSONID"],
    sort=True,
)["purpose"].agg(list)

sequences = grouped.tolist()

# Stop if cleaning left no person sequences to analyze.
if not sequences:
    raise ValueError("No valid person sequences remain.")

# Save a readable version of each person's sequence.
sequence_table = grouped.reset_index(name="sequence")
sequence_table["sequence"] = sequence_table["sequence"].map(
    lambda sequence: " -> ".join(sequence)
)
sequence_table.to_csv(
    OUTPUT / "person_sequences.csv",
    index=False,
)

# Find the two most frequent ordered patterns, each at least two purposes long.
# Patterns can have gaps; for example, A -> C can occur in A -> B -> C.
miner = PrefixSpan(sequences)
miner.minlen = 2
patterns = miner.topk(2)

if len(patterns) < 2:
    raise ValueError("Fewer than two sequential patterns exist.")

# Convert the results into rows with counts and support proportions.
pattern_rows = [
    {
        "pattern": " -> ".join(pattern),
        "person_count": count,
        "support": count / len(sequences),
    }
    for count, pattern in patterns
]

pattern_table = pd.DataFrame(pattern_rows)
pattern_table.to_csv(
    OUTPUT / "top_two_sequential_patterns.csv",
    index=False,
)

# Return True when every item in pattern appears in sequence in the same order.
def contains_subsequence(sequence, pattern):
    position = 0

    for event in sequence:
        if event == pattern[position]:
            position += 1
            if position == len(pattern):
                return True

    return False


# Split each pattern into an earlier part (antecedent)
# and a later part (consequent).
rule_rows = []

for count, pattern in patterns:
    for split in range(1, len(pattern)):
        antecedent = pattern[:split]
        consequent = pattern[split:]

        # Count people whose sequence contains the antecedent in order.
        antecedent_count = sum(
            contains_subsequence(sequence, antecedent)
            for sequence in sequences
        )

        rule_rows.append({
            "antecedent": " -> ".join(antecedent),
            "consequent": " -> ".join(consequent),
            "person_count": count,
            "support": count / len(sequences),
            "antecedent_count": antecedent_count,
            "confidence": count / antecedent_count,
        })

# Remove duplicate rules and rank by confidence, then support.
rules = (
    pd.DataFrame(rule_rows)
    .drop_duplicates(["antecedent", "consequent"])
    .sort_values(
        ["confidence", "support", "antecedent", "consequent"],
        ascending=[False, False, True, True],
    )
)

top_rules = rules.head(2)

# Save all candidate rules and the two highest-confidence rules.
rules.to_csv(OUTPUT / "candidate_sequential_rules.csv", index=False)
top_rules.to_csv(
    OUTPUT / "top_two_sequential_rules.csv",
    index=False,
)

# Display the two most frequent patterns and the two top rules.
print("\nTWO MOST FREQUENT SEQUENTIAL PATTERNS")
print(pattern_table.to_string(index=False))

print("\nTWO HIGHEST-CONFIDENCE SEQUENTIAL RULES")
print(top_rules.to_string(index=False))

print("\nPersons analysed:", len(sequences))
print("Trips analysed:", len(trips))
print("Results saved to:", OUTPUT)
