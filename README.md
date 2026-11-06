# EECS 4412 Data Mining Project

York University · Fall 2026  
Team members: Shayan Rabiee, Dylan Smith, and Hanifah Lameed

This project uses the 2017 National Household Travel Survey (NHTS) to study how household characteristics and vehicle availability relate to travel behaviour. Phase 2 covers principal component analysis (PCA), household vehicle ownership classification, and household clustering.

## Data

The source files are stored in the repository root.

| File | Records |
|---|---|
| `hhpub_202609201846.csv` | Households |
| `vehpub_202609201846.csv` | Vehicles |
| `trippub_202609201847.csv` | Trips |

[NHTS documentation](https://nhts.ornl.gov/documentation) · [2017 NHTS codebook](https://nhts.ornl.gov/media/2017/doc/codebook_v1.2.pdf)

## Setup

Open a terminal in the repository root and install the dependencies.

```bash
python -m pip install -r requirements.txt
```

The folder name `Assingment2` in the commands below matches the repository.

## Task 1 — Principal Component Analysis

From the repository root, run:

```bash
python 217761263-217285287-219714815-Task1-PCA.py
```

The script joins trip and household data using `HOUSEID`, cleans and standardizes six numeric variables, and retains three principal components. Together, the components explain 76.66% of the variance. The script writes the component scores to the `PCA_3PCs` sheet of `217761263-217285287-219714815.xlsx` and generates an explained-variance chart and a PC1-versus-PC2 scatter plot.

## Task 2 — Classification

We used seven household characteristics to predict whether a household had **0, 1, 2, or 3+ vehicles**. Both classifiers were evaluated on the same held-out set of 25,060 households.

The labelled, training, and test CSVs are already committed. To regenerate them, run the following from the repository root. Run `task2_split.py` inside `Assingment2` because it writes its output to the current directory.

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

The evaluation scripts generate confusion matrices and one-vs-rest ROC plots. Decision tree outputs are saved in `Assingment2/`, and Naïve Bayes outputs are saved in `Assingment2/Naive_bayes/`.

## Task 3 — Clustering Analysis

From the repository root, run:

```bash
python Assingment2/task3_clustering.py
```

The script uses the same seven household predictors as Task 2. It standardizes numeric predictors, one-hot encodes `HOMEOWN` and `URBAN`, and runs k-means for `k = 3, 4, 5`. Vehicle count and vehicle class are excluded from the clustering inputs.

| Number of clusters | Within-cluster SSE |
|---:|---:|
| 3 | 413,388.26 |
| 4 | 360,377.71 |
| 5 | 323,705.25 |

Five clusters produced the lowest observed SSE. In a separate held-out evaluation, the `k = 5` model reached 61.46% accuracy after each cluster was mapped to its most frequent vehicle class in the training data. Both final Task 2 classifiers achieved higher test accuracy.

Outputs are saved in `Assingment2/task3_results/`. They include the SSE table and plot, cluster profiles, cluster assignments, and the held-out comparison. The generated `test_comparison.csv` includes a **Gaussian Naïve Bayes baseline**. The final **Categorical Naïve Bayes** result is produced by the Task 2 evaluation script.

## Task 4 — Progress on Research Objectives

The Phase 2 report reviews progress on all five original research questions. Classification directly addresses household vehicle ownership. The clustering question was revised to focus on household profiles and their relationship to vehicle ownership classes, reflecting the variables used in Phase 2.
