# EECS 4412 Data Mining Project

## Analysis of Household Characteristics, Vehicle Availability, and Travel Behaviour

York University · Fall 2026

## Project Overview

This project uses the **2017 National Household Travel Survey (NHTS)** to investigate how household characteristics, vehicle ownership and usage, and daily travel behaviour relate to one another.

We work with three types of records: households, vehicles, and trips. Phase 1 covers data preparation, descriptive statistics, and visualization. Phase 2 covers principal component analysis (PCA), household vehicle ownership classification, and household clustering. The broader research objectives also include regression analyses of trip duration and annual vehicle mileage.

## Team Members

- Shayan Rabiee
- Dylan Smith
- Hanifah Lameed

## Dataset

The **Federal Highway Administration (FHWA)** provides the 2017 NHTS. The survey records household demographics and income, vehicle ownership and characteristics, transportation modes, trip purpose, distance and duration, traveller characteristics, and day-of-week travel patterns.

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

## Setup

Open a terminal in the repository root and install the Python dependencies.

```bash
python -m pip install -r requirements.txt
```

The folder name `Assingment2` in the commands below matches the repository.

## Phase 2

### Task 1 — Principal Component Analysis

From the repository root, run:

```bash
python 217761263-217285287-219714815-Task1-PCA.py
```

The script joins trip and household records using `HOUSEID`, cleans and standardizes six numeric variables, and retains three principal components. Together, the components explain **76.66%** of the variance.

The script writes the component scores to the `PCA_3PCs` sheet of `217761263-217285287-219714815.xlsx` and generates an explained-variance chart and a PC1-versus-PC2 scatter plot. Committed copies of the workbook and PCA figures are in `Part 2 figures + excel/`.

### Task 2 — Classification

We used seven household predictors to classify households into four vehicle ownership categories: **0, 1, 2, or 3+ vehicles**. Both final models were evaluated on the same **25,060 held-out households**.

The labelled, training, and test CSVs are already committed. To regenerate them, run these commands from the repository root. The cleaning script writes its output to the current directory, so run it inside `Assingment2`.

```bash
cd Assingment2
python task2_split.py
python task2_labelling.py
python task2_train_test_split.py
cd ..
```

To train and evaluate the final classifiers, run:

```bash
python Assingment2/task2_full_training_set_test.py
python Assingment2/Naive_bayes/task2_p2_2_train_final.py
python Assingment2/Naive_bayes/task2_p2_3_test_final.py
```

| Final classifier | Held-out test accuracy |
|---|---:|
| Decision tree (`max_depth=10`) | 68.28% |
| Categorical Naïve Bayes (`alpha=0.1`) | 67.28% |

The evaluation scripts generate confusion matrices and one-vs-rest ROC curves. Decision tree outputs are saved in `Assingment2/`, and Naïve Bayes outputs are saved in `Assingment2/Naive_bayes/`.

### Task 3 — Clustering Analysis

From the repository root, run:

```bash
python Assingment2/task3_clustering.py
```

The script uses the same seven household predictors as Task 2. It standardizes numeric predictors, one-hot encodes `HOMEOWN` and `URBAN`, and runs k-means with `k = 3, 4, 5`. Vehicle count and the derived vehicle ownership class are excluded from the clustering inputs.

| Number of clusters | Within-cluster SSE |
|---:|---:|
| 3 | 413,388.26 |
| 4 | 360,377.71 |
| 5 | 323,705.25 |

Five clusters produced the lowest observed SSE. In a separate held-out evaluation, the five-cluster model achieved **61.46% accuracy** after each cluster was mapped to its most frequent vehicle class in the training data. Both final Task 2 classifiers achieved higher test accuracy.

The script saves SSE results, cluster assignments, cluster profiles, and a test comparison in `Assingment2/task3_results/`. The generated `test_comparison.csv` uses **Gaussian Naïve Bayes** as a baseline. The final **Categorical Naïve Bayes** result comes from the Task 2 evaluation.

### Task 4 — Progress on Research Objectives

The Phase 2 report reviews progress on the five original questions about transportation mode, trip duration, annual vehicle mileage, household vehicle ownership, and household travel-day profiles. Classification directly addresses household vehicle ownership. We revised the fifth question to ask whether the selected household characteristics form interpretable groups and how closely those groups align with vehicle ownership classes.
