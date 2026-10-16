import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# ============================================================
# 1. LOAD TRIP DATASET
# ============================================================

trip = pd.read_csv(
    "trippub_202609201847.csv",
    usecols=["HOUSEID", "TRPMILES", "TRVLCMIN", "R_AGE"],
    low_memory=False
)

print("Trip dataset loaded successfully.")
print("Shape:", trip.shape)

household = pd.read_csv(
    "hhpub_202609201846.csv",
    usecols=["HOUSEID", "HHSIZE", "HHVEHCNT", "WRKCOUNT"],
    low_memory=False
)

# Household-level features are stored separately from trip records.
trip = trip.merge(
    household,
    on="HOUSEID",
    how="left",
    validate="many_to_one"
)


# ============================================================
# 2. SELECT PCA FEATURES
# ============================================================

pca_columns = [
    "TRPMILES",
    "TRVLCMIN",
    "R_AGE",
    "HHSIZE",
    "HHVEHCNT",
    "WRKCOUNT"
]

pca_data = trip[pca_columns].copy()


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

for column in pca_columns:

    pca_data[column] = pd.to_numeric(
        pca_data[column],
        errors="coerce"
    )


# Remove missing observations
pca_data = pca_data.dropna()


# Remove negative NHTS special-response codes
for column in pca_columns:

    pca_data = pca_data[
        pca_data[column] >= 0
    ]


print("\nClean PCA dataset:")
print(pca_data.head())

print(
    "\nNumber of valid observations:",
    len(pca_data)
)


# ============================================================
# 4. STANDARDIZE DATA
# ============================================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    pca_data
)


# ============================================================
# 5. PERFORM PCA
# ============================================================

pca = PCA(
    n_components=3
)

principal_components = pca.fit_transform(
    scaled_data
)

# ============================================================
# PCA LOADINGS
# ============================================================

loadings = pd.DataFrame(
    pca.components_.T,
    index=pca_columns,
    columns=["PC1", "PC2", "PC3"]
)

print("\n======================================")
print("PCA LOADINGS")
print("======================================")
print(loadings)


# ============================================================
# 6. EXPLAINED VARIANCE
# ============================================================

explained_variance = (
    pca.explained_variance_ratio_
)

print("\n======================================")
print("PCA EXPLAINED VARIANCE")
print("======================================")

print(
    "PC1:",
    round(explained_variance[0] * 100, 2),
    "%"
)

print(
    "PC2:",
    round(explained_variance[1] * 100, 2),
    "%"
)

print(
    "PC3:",
    round(explained_variance[2] * 100, 2),
    "%"
)

print(
    "Total:",
    round(
        explained_variance.sum() * 100,
        2
    ),
    "%"
)


# ============================================================
# 7. CREATE PCA DATAFRAME
# ============================================================

pca_result = pd.DataFrame(
    principal_components,
    columns=[
        "PC1",
        "PC2",
        "PC3"
    ]
)

print("\nFirst PCA observations:")
print(pca_result.head())


# ============================================================
# 8. GRAPH 1 - EXPLAINED VARIANCE
# ============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    ["PC1", "PC2", "PC3"],
    explained_variance * 100
)

plt.xlabel(
    "Principal Component"
)

plt.ylabel(
    "Explained Variance (%)"
)

plt.title(
    "Explained Variance of Principal Components"
)

plt.tight_layout()

plt.savefig(
    "Task1_PCA_Explained_Variance.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. GRAPH 2 - PC1 VS PC2
# ============================================================

plot_data = pca_result.sample(
    n=min(
        10000,
        len(pca_result)
    ),
    random_state=42
)

plt.figure(figsize=(8, 6))

plt.scatter(
    plot_data["PC1"],
    plot_data["PC2"],
    alpha=0.3,
    s=10
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    "PCA Projection of Trip Records: PC1 vs PC2"
)

plt.tight_layout()

plt.savefig(
    "Task1_PCA_PC1_vs_PC2.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. CREATE EXCEL WORKBOOK
# ============================================================

workbook_name = (
    "217761263-217285287-219714815.xlsx"
)

with pd.ExcelWriter(
    workbook_name,
    engine="xlsxwriter"
) as writer:

    pca_result.to_excel(
        writer,
        sheet_name="PCA_3PCs",
        index=False
    )


print(
    "\nExcel workbook successfully created:"
)

print(workbook_name)
