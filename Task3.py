import pandas as pd
# ============================================================
# EECS 4412 - Phase 1
# Task 3: Excel Export of the Three Datasets
# 2017 National Household Travel Survey (NHTS)
# ============================================================
# 1. LOAD DATASETS

household = pd.read_csv(
    "hhpub_202609201846.csv",
    low_memory=False
)

vehicle = pd.read_csv(
    "vehpub_202609201846.csv",
    low_memory=False
)

trip = pd.read_csv(
    "trippub_202609201847.csv",
    low_memory=False
)

# 2. EXPORT TO EXCEL

excel_filename = "217761263-217285287-219714815.xlsx"

print("Writing Excel file (this can take a few minutes)...")

with pd.ExcelWriter(excel_filename) as writer:
    household.to_excel(writer, sheet_name="Household", index=False)
    vehicle.to_excel(writer, sheet_name="Vehicle", index=False)
    trip.to_excel(writer, sheet_name="Trip", index=False)

print("\nExcel file successfully created:")
print(excel_filename)
