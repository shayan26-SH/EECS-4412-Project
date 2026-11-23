<div align="center">

# EECS 4412 Data Mining Project

## Analysis of Household Characteristics, Vehicle Availability, and Travel Behaviour

York University · Fall 2026 
Team Members: Shayan Rabiee, Dylan Smith, Hanifah Lameed

</div>

<p align="center">
  <a href="#project-overview">Project Overview</a> &nbsp;·&nbsp;
  <a href="#dataset">Dataset</a> &nbsp;·&nbsp;
  <a href="#setup">Setup</a> &nbsp;·&nbsp;
  <a href="#phase-1">Phase 1</a> &nbsp;·&nbsp;
  <a href="#phase-2">Phase 2</a> &nbsp;·&nbsp;
  <a href="#phase-3--task-1-association-analysis">Association Analysis</a> &nbsp;·&nbsp;
  <a href="#phase-3--task-2-sequential-analysis">Sequential Analysis</a>
</p>

---

## Project Overview

This project uses the **2017 National Household Travel Survey (NHTS)** to investigate how household characteristics, vehicle ownership and usage, and daily travel behaviour relate to one another.

We work with three types of records: households, vehicles, and trips. Phase 1 covers data preparation, descriptive statistics, and visualization. Phase 2 covers principal component analysis (PCA), household vehicle ownership classification, and household clustering. The broader research objectives also include regression analyses of trip duration and annual vehicle mileage.



---

## Dataset

The **Federal Highway Administration (FHWA)** provides the 2017 NHTS. The survey records household demographics and income, vehicle ownership and characteristics, transportation modes, trip purpose, distance and duration, traveler characteristics, and day-of-week travel patterns.

- [NHTS website](https://nhts.ornl.gov/)
- [NHTS documentation](https://nhts.ornl.gov/documentation)
- [2017 NHTS codebook](https://nhts.ornl.gov/media/2017/doc/codebook_v1.2.pdf)

We used the codebook to interpret variable definitions, category codes, and missing-response codes.

### Dataset Files

The source CSVs are stored in the repository root.

| File | Records |
|---|---|
| `hhpub_202609201846.csv` | Households |
| `vehpub_202609201846.csv` | Vehicles |
| `trippub_202609201847.csv` | Trips |


---

## Setup

Open a terminal in the repository root and install the Python dependencies.

```bash
python -m pip install -r requirements.txt
```

The folder name `Assignment2` in the commands below matches the repository.


---

## Phase 1

Assignment 1 scripts, database notes, exported CSVs, visualizations and reference photos are grouped in `Assignment1/`. Original and student-ID-named versions are preserved. The source CSVs and `nhtsCleaned.rar` remain shared at the repository root.

From the repository root:

```bash
python Assignment1/Database.py
python Assignment1/Required.py
python Assignment1/Task6.py
```

- `Database.py` prints the dataset sizes and column names; its student-ID-named version performs the same check.
- `Required.py` generates the five Phase 1 figures in `Assignment1/217761263-217285287-219714815-Visualizations/`.
- `Task6.py` generates descriptive statistics and its named CSV within `Assignment1/`.
- `217761263-217285287-219714815-Task6.py` additionally exports the seven-sheet Phase 1 workbook. Run it with `python Assignment1/217761263-217285287-219714815-Task6.py`; its CSV and workbook stay inside `Assignment1/`.

All Phase 1 scripts resolve source and output paths relative to their file location. The Phase 1 workbook is separate from the combined Phase 2 submission workbook.


---

## Phase 2

### Task 1 — Principal Component Analysis

From the repository root, run:

```bash
python Assignment2/Task1/task1_pca.py
```

The script joins trip and household records using `HOUSEID`, cleans and standardizes six numeric variables, and retains three principal components. Together, the components explain **76.66%** of the variance.

The script and its figures are grouped in `Assignment2/Task1/`. Input paths are relative to the script, so it runs from any working directory. It saves the explained-variance chart, PC1-versus-PC2 scatter plot and a PCA-only `task1_pca_scores.xlsx` workbook in `Assignment2/Task1/results/`. Charts are saved without opening interactive windows.

The combined submission workbook remains at `Assignment2/Submission/217761263-217285287-219714815.xlsx` because it contains both Task 1 and Task 2 sheets. Running PCA does not modify that shared workbook. If the analysis changes, update only its `PCA_3PCs` sheet after verification, preserving `Classification_Data`.

### Task 2 — Classification

We used seven household predictors to classify households into four vehicle ownership categories: **0, 1, 2, or 3+ vehicles**. Both final models were evaluated on the same **25,060 held-out households**.

The labelled, training, and test CSVs are already committed. To regenerate them, run these commands from the repository root. All Task 2 paths are resolved relative to the scripts, including the cleaned CSV output.

```bash
python Assignment2/Task2/task2_01_clean_data.py
python Assignment2/Task2/task2_02_label_classes.py
python Assignment2/Task2/task2_03_split_train_test.py
```

Task 2 scripts, datasets, models and figures are grouped in `Assignment2/Task2/`, including the `Naive_bayes/` subfolder. The scripts are numbered in execution order:

- `00`: optionally inspect the selected input columns.
- `01`–`03`: clean records, assign vehicle classes, and create the training/test split.
- `04`: cross-validate the decision tree.
- `05`: train and evaluate the final decision tree.
- `06`: cross-validate Categorical Naive Bayes.
- `07`–`08`: train and evaluate the final Naive Bayes model.

Scripts `06`–`08` are in `Assignment2/Task2/Naive_bayes/`. Files named `task2_unused_*_placeholder.py` are unused stubs, not pipeline steps; confusion matrices are generated by the evaluation scripts.

To run the cross-validation scripts from the repository root:

```bash
python Assignment2/Task2/task2_04_cross_validate_tree.py
python Assignment2/Task2/Naive_bayes/task2_06_cross_validate_naive_bayes.py
```

Each cross-validation script currently evaluates one setting: tree depth 10 or Naive Bayes alpha 0.001. The final Naive Bayes training script uses the selected alpha 0.1. Reproducing all reported tuning trials requires changing the setting between runs.

To train and evaluate the final classifiers, run:

```bash
python Assignment2/Task2/task2_05_train_evaluate_tree.py
python Assignment2/Task2/Naive_bayes/task2_07_train_naive_bayes.py
python Assignment2/Task2/Naive_bayes/task2_08_evaluate_naive_bayes.py
```

| Final classifier | Held-out test accuracy |
|---|---:|
| Decision tree (`max_depth=10`) | 68.28% |
| Categorical Naïve Bayes (`alpha=0.1`) | 67.28% |

The evaluation scripts generate confusion matrices and one-vs-rest ROC curves. Decision tree outputs are saved in `Assignment2/Task2/`, and Naïve Bayes outputs are saved in `Assignment2/Task2/Naive_bayes/`.

The submission workbook, `Assignment2/Submission/217761263-217285287-219714815.xlsx`, contains both `PCA_3PCs` and `Classification_Data`. The classification sheet combines all 100,238 training and 25,060 test households, preserving their source row order within each split. It includes the seven predictors, `VEHICLE_CLASS`, `HOUSEID`, and a `Split` column (`Train` or `Test`), with a frozen header row and filters. Its values were checked against `Assignment2/Task2/task2_train.csv` and `task2_test.csv`; the original PCA sheet was preserved unchanged. The model scripts do not refresh this worksheet automatically. The Task 1 script writes its PCA-only workbook to `Assignment2/Task1/results/task1_pca_scores.xlsx`, leaving this combined submission workbook untouched.

### Task 3 — Clustering Analysis

From the repository root, run:

```bash
python Assignment2/Task3/task3_clustering.py
```

Task 3 code and generated outputs are grouped in `Assignment2/Task3/`. Input files remain shared with Task 2 at `Assignment2/Task2/task2_train.csv` and `Assignment2/Task2/task2_test.csv`; paths are resolved relative to the script, so it can be run from any working directory.

The script uses the same seven household predictors as Task 2. It standardizes numeric predictors, one-hot encodes `HOMEOWN` and `URBAN`, and runs k-means with `k = 3, 4, 5`. Vehicle count and the derived vehicle ownership class are excluded from the clustering inputs.

| Number of clusters | Within-cluster SSE |
|---:|---:|
| 3 | 413,388.26 |
| 4 | 360,377.71 |
| 5 | 323,705.25 |

Five clusters produced the lowest observed SSE. In a separate held-out evaluation, the five-cluster model achieved **61.46% accuracy** after each cluster was mapped to its most frequent vehicle class in the training data. Both final Task 2 classifiers achieved higher test accuracy.

The script saves SSE results, cluster assignments, cluster profiles, and a test comparison in `Assignment2/Task3/results/`. The generated `test_comparison.csv` uses **Gaussian Naïve Bayes** as a baseline. The final **Categorical Naïve Bayes** result comes from the Task 2 evaluation.

### Task 4 — Progress on Research Objectives

The Phase 2 report reviews progress on the five original questions about transportation mode, trip duration, annual vehicle mileage, household vehicle ownership, and household travel-day profiles. Classification directly addresses household vehicle ownership. We revised the fifth question to ask whether the selected household characteristics form interpretable groups and how closely those groups align with vehicle ownership classes.


---

## Phase 3 — Task 1: Association Analysis

### Objective

Identify frequently occurring combinations of household characteristics and examine the association rules derived from them.

**Research question:** Which combinations of household size, driver count, worker count, income category, urban classification, and vehicle ownership occur most frequently, and which association rules have the highest and lowest confidence?

### Dataset and Preprocessing

The analysis uses `hhpub_202609201846.csv`. Each household forms one transaction.

The script removes records with missing or invalid values, preserves legitimate zeros, and converts numeric attributes into categorical items.

| Attribute | Transaction categories |
|---|---|
| Household size | 1, 2, 3+ members |
| Driver count | 0, 1, 2+ drivers |
| Worker count | 0, 1, 2+ workers |
| Household income | Below $50,000; $50,000–$99,999; $100,000+ |
| Vehicle count | 0, 1, 2, 3+ vehicles |
| Urban classification | Four original NHTS categories |

`HOUSEID` is retained for identifying records and excluded from the mining inputs.

### Method

The script uses **FP-Growth** to find the five most frequent itemsets containing at least two items. Support is the proportion of retained households containing every item in an itemset.

It generates all association rules from these five itemsets and selects four distinct rules: two with the highest confidence and two with the lowest confidence. Ties are resolved using support and alphabetical ordering.

For the two highest-confidence rules, the script calculates five interest measures:

- Lift
- Leverage
- Conviction
- Cosine
- Jaccard

The selected measures should be checked against the measures covered in Module 005.

### Installation and Execution

Install the required packages:

```bash
python -m pip install pandas numpy mlxtend
```

From the repository root, run:

```bash
python Assingment3/Task1-Association/task1_association.py
```

The expected folder structure is:

```text
EECS-4412-Project/
├── hhpub_202609201846.csv
└── Assingment3/
    └── Task1-Association/
        └── task1_association.py
```

### Output Files

Results are saved in `Assingment3/Task1-Association/task1_results/`.

| File | Contents |
|---|---|
| `raw_households.csv` | Selected household attributes before cleaning |
| `processed_households.csv` | Cleaned and categorized household transactions |
| `top_five_itemsets.csv` | Five most frequent itemsets, their sizes, support, and counts |
| `all_candidate_rules.csv` | All rules generated from the selected itemsets |
| `four_selected_rules.csv` | Two highest- and two lowest-confidence rules |
| `top_two_rules_interest_measures.csv` | Five interest measures for the two highest-confidence rules |

Support and confidence are saved as proportions. Multiply by 100 to express them as percentages.

The analysis uses unweighted household records, so the results describe the retained sample. Association rules describe co-occurrence and do not establish causation.


---

## Phase 3 — Task 2: Sequential Analysis

### Objective

Identify frequent ordered patterns of trip purposes within a person’s travel day and derive sequential rules from those patterns.

**Research question:** What are the two most frequent ordered trip-purpose patterns, and which rules derived from them have the highest confidence?

### Dataset

The analysis uses `trippub_202609201847.csv`.

| Field | Purpose |
|---|---|
| `HOUSEID` | Identifies the household |
| `PERSONID` | Identifies a person within the household |
| `TDTRPNUM` | Determines trip order |
| `TRIPPURP` | Describes the generalized trip purpose |

The Trip table provides individual travel events and their order. Trips are grouped by `HOUSEID` and `PERSONID` to construct one sequence per person.

### Preprocessing

The script performs the following steps:

1. Converts trip numbers to numeric values.
2. Removes whitespace and leading apostrophes from trip-purpose codes and converts them to uppercase.
3. Excludes records with missing household or person identifiers.
4. Removes entire person sequences containing invalid trip numbers or unrecognized purpose codes.
5. Checks for duplicate trip positions within each person’s sequence.
6. Sorts trips by trip number and constructs ordered purpose sequences.

The following purpose codes are retained:

| Code | Description |
|---|---|
| `HBW` | Home-based work |
| `HBSHOP` | Home-based shopping |
| `HBSOCREC` | Home-based social or recreational |
| `HBO` | Home-based other |
| `NHB` | Non-home-based |

### Method

The script uses **PrefixSpan** to find the two most frequent sequential patterns containing at least two events.

Patterns preserve order and allow intervening trips. For example, `A → C` occurs within `A → B → C`.

Support is calculated as:

```text
Support = People containing the pattern / Total people analysed
```

Each person contributes at most one count to a pattern’s support, regardless of how many times the pattern appears in their sequence. People with a single valid trip remain in the support denominator.

The script splits each selected pattern into an antecedent prefix and a consequent suffix, then selects the two highest-confidence candidate rules.

```text
Confidence = People containing the complete ordered pattern
             / People containing the antecedent
```

### Installation

```bash
python -m pip install pandas prefixspan
```

### Folder Structure and Execution

With `ROOT = BASE.parent.parent`, use this structure:

```text
EECS-4412-Project/
├── trippub_202609201847.csv
└── Assingment3/
    └── Task2/
        └── task2_sequential.py
```

From the repository root, run:

```bash
python Assingment3/Task2/task2_sequential.py
```

### Output Files

Results are saved in `Assingment3/Task2/task2_results/`.

| File | Contents |
|---|---|
| `raw_trips.csv` | Selected trip fields before preprocessing |
| `processed_trip_events.csv` | Retained trip events and readable purpose labels |
| `person_sequences.csv` | Ordered trip-purpose sequence for each person |
| `top_two_sequential_patterns.csv` | Two most frequent patterns, person counts, and support |
| `candidate_sequential_rules.csv` | Rules derived from the two selected patterns |
| `top_two_sequential_rules.csv` | Two highest-confidence candidate rules |

Support and confidence are stored as proportions. Multiply by 100 to express them as percentages.

The results describe unweighted person sequences from the retained sample and patterns within a recorded travel day.


To visit the WikiPage => [View Project Wiki](../../wiki)
