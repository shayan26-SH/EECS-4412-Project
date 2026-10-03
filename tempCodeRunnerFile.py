import os
import pandas as pd


# ============================================================
# EECS 4412 - Phase 1
# Task 6: Basic Statistical Analysis
# 2017 National Household Travel Survey
# ============================================================


# ============================================================
# 1. LOAD DATASETS
# ============================================================

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


print("======================================")
print("TASK 6 - BASIC STATISTICAL ANALYSIS")
print("======================================")


# ============================================================
# NOMINAL ATTRIBUTE
# TRPTRANS - Transportation Mode
# Operations:
# 1. Mode
# 2. Frequency / Percentage
# ============================================================

print("\n======================================")
print("1. NOMINAL ATTRIBUTE: TRPTRANS")
print("======================================")


# Convert to numeric
trip["TRPTRANS_NUM"] = pd.to_numeric(
    trip["TRPTRANS"],
    errors="coerce"
)


# Official transportation mode labels
transport_labels = {
    1: "Walk",
    2: "Bicycle",
    3: "Car",
    4: "SUV",
    5: "Van",
    6: "Pickup Truck",
    7: "Golf Cart / Segway",
    8: "Motorcycle / Moped",
    9: "RV",
    10: "School Bus",
    11: "Public / Commuter Bus",
    12: "Paratransit",
    13: "Private / Charter Bus",
    14: "City-to-City Bus",
    15: "Amtrak / Commuter Rail",
    16: "Subway / Rail / Streetcar",
    17: "Taxi / Uber / Lyft",
    18: "Rental Car",
    19: "Airplane",
    20: "Boat / Ferry",
    24: "Special Transit",
    97: "Other"
}


# Remove missing/invalid values
trip_transport = trip[
    trip["TRPTRANS_NUM"].isin(transport_labels.keys())
].copy()


trip_transport["Transportation_Mode"] = (
    trip_transport["TRPTRANS_NUM"]
    .map(transport_labels)
)


# ----------------------------
# STATISTIC 1: MODE
# ----------------------------

transport_mode = (
    trip_transport["Transportation_Mode"]
    .mode()
    .iloc[0]
)

print("\nMode:")
print(transport_mode)


# ----------------------------
# STATISTIC 2:
# FREQUENCY AND PERCENTAGE
# ----------------------------

transport_frequency = (
    trip_transport["Transportation_Mode"]
    .value_counts()
)

transport_percentage = (
    trip_transport["Transportation_Mode"]
    .value_counts(normalize=True) * 100
)


transport_summary = pd.DataFrame({
    "Frequency": transport_frequency,
    "Percentage": transport_percentage
})


print("\nTransportation Mode Frequencies:")
print(transport_summary)


# ============================================================
# ORDINAL ATTRIBUTE
# HHFAMINC - Household Income Category
# Operations:
# 1. Median
# 2. Mode
# ============================================================

print("\n======================================")
print("2. ORDINAL ATTRIBUTE: HHFAMINC")
print("======================================")


household["HHFAMINC_NUM"] = pd.to_numeric(
    household["HHFAMINC"],
    errors="coerce"
)


# Valid income categories are 1 through 11
income = household.loc[
    household["HHFAMINC_NUM"].between(1, 11),
    "HHFAMINC_NUM"
].dropna()


income_labels = {
    1: "Less than $10,000",
    2: "$10,000-$14,999",
    3: "$15,000-$24,999",
    4: "$25,000-$34,999",
    5: "$35,000-$49,999",
    6: "$50,000-$74,999",
    7: "$75,000-$99,999",
    8: "$100,000-$124,999",
    9: "$125,000-$149,999",
    10: "$150,000-$199,999",
    11: "$200,000 or more"
}


# ----------------------------
# STATISTIC 1: MEDIAN
# ----------------------------

income_median = income.median()

print("\nMedian Income Category Code:")
print(income_median)

print("Median Income Category:")
print(
    income_labels.get(
        int(income_median),
        "Unknown"
    )
)


# ----------------------------
# STATISTIC 2: MODE
# ----------------------------

income_mode = income.mode().iloc[0]

print("\nMode Income Category Code:")
print(income_mode)

print("Mode Income Category:")
print(
    income_labels.get(
        int(income_mode),
        "Unknown"
    )
)


# ============================================================
# INTERVAL ATTRIBUTE
# TDAYDATE - Travel Year/Month
# Operations:
# 1. Mean travel month
# 2. Standard deviation
# ============================================================

print("\n======================================")
print("3. INTERVAL ATTRIBUTE: TDAYDATE")
print("======================================")


household["TDAYDATE_NUM"] = pd.to_numeric(
    household["TDAYDATE"],
    errors="coerce"
)


# Remove missing values
travel_date = (
    household["TDAYDATE_NUM"]
    .dropna()
    .astype(int)
)


# Convert YYYYMM into year and month
travel_year = travel_date // 100
travel_month = travel_date % 100


# Keep only legitimate calendar months
valid_dates = (
    (travel_month >= 1) &
    (travel_month <= 12)
)


travel_year = travel_year[valid_dates]
travel_month = travel_month[valid_dates]


# Convert calendar date to a continuous month index.
# Example:
# Jan 2016 -> 2016 * 12 + 0
# Feb 2016 -> 2016 * 12 + 1
month_index = (
    travel_year * 12 +
    travel_month - 1
)


# ----------------------------
# STATISTIC 1: MEAN DATE
# ----------------------------

mean_month_index = month_index.mean()

mean_year = int(mean_month_index // 12)

mean_month = int(
    round(
        mean_month_index -
        mean_year * 12
    ) + 1
)


# Correct possible rounding to month 13
if mean_month == 13:
    mean_month = 1
    mean_year += 1


print("\nMean Travel Month:")
print(
    f"{mean_year}-{mean_month:02d}"
)


# ----------------------------
# STATISTIC 2:
# STANDARD DEVIATION
# ----------------------------

date_std = month_index.std()

print("\nStandard Deviation:")
print(
    f"{date_std:.2f} months"
)


# ============================================================
# RATIO ATTRIBUTE
# TRPMILES - Trip Distance
# Operations:
# 1. Mean
# 2. Median
# ============================================================

print("\n======================================")
print("4. RATIO ATTRIBUTE: TRPMILES")
print("======================================")


trip["TRPMILES_NUM"] = pd.to_numeric(
    trip["TRPMILES"],
    errors="coerce"
)


# Remove negative special/missing codes
trip_miles = trip.loc[
    trip["TRPMILES_NUM"] >= 0,
    "TRPMILES_NUM"
].dropna()


# ----------------------------
# STATISTIC 1: MEAN
# ----------------------------

trip_mean = trip_miles.mean()

print("\nMean Trip Distance:")
print(
    f"{trip_mean:.2f} miles"
)


# ----------------------------
# STATISTIC 2: MEDIAN
# ----------------------------

trip_median = trip_miles.median()

print("\nMedian Trip Distance:")
print(
    f"{trip_median:.2f} miles"
)


# ============================================================
# CREATE TASK 6 CSV FILE
# ============================================================

print("\n======================================")
print("CREATING TASK 6 CSV FILE")
print("======================================")


# Prepare cleaned values for CSV

nominal_csv = (
    trip_transport["Transportation_Mode"]
    .reset_index(drop=True)
)

ordinal_csv = (
    income
    .map(income_labels)
    .reset_index(drop=True)
)


# Rebuild clean YYYYMM values
interval_csv = (
    travel_year.astype(str) +
    travel_month.astype(str).str.zfill(2)
).reset_index(drop=True)


ratio_csv = (
    trip_miles
    .reset_index(drop=True)
)


# Each dimension contains a different number of records.
# pandas will automatically fill shorter columns with blanks.
task6_data = pd.concat(
    [
        nominal_csv,
        ordinal_csv,
        interval_csv,
        ratio_csv
    ],
    axis=1
)


task6_data.columns = [
    "TRPTRANS_Nominal",
    "HHFAMINC_Ordinal",
    "TDAYDATE_Interval",
    "TRPMILES_Ratio"
]


# ============================================================
# CREATE TASK 6 CSV FILE
# ============================================================

csv_filename = (
    "Shayan-Rabiee-Dylan-Smith-Hanifah-Lameed--T1.csv"
)

task6_data.to_csv(
    csv_filename,
    index=False
)

print("\nCSV file successfully created:")
print(csv_filename)


# ============================================================
# CREATE TASK 7 EXCEL WORKBOOK
# ============================================================

print("\n======================================")
print("CREATING TASK 7 EXCEL WORKBOOK")
print("======================================")

workbook_name = "workbook217761263-217285287-219714815.xlsx"

with pd.ExcelWriter(workbook_name, engine="xlsxwriter") as writer:

    # --------------------------------------------------------
    # SHEET 1: HOUSEHOLD
    # Complete household dataset
    # --------------------------------------------------------
    household.to_excel(
        writer,
        sheet_name="Household",
        index=False
    )

    # --------------------------------------------------------
    # SHEET 2: VEHICLE
    # Complete vehicle dataset
    # --------------------------------------------------------
    vehicle.to_excel(
        writer,
        sheet_name="Vehicle",
        index=False
    )

    # --------------------------------------------------------
    # SHEET 3: TRIP
    # Complete trip dataset
    # --------------------------------------------------------
    trip.to_excel(
        writer,
        sheet_name="Trip",
        index=False
    )

    # --------------------------------------------------------
    # SHEET 4: TRPTRANS
    # Nominal dimension
    # --------------------------------------------------------
    trptrans_sheet = pd.DataFrame({
        "TRPTRANS": trip_transport["TRPTRANS_NUM"],
        "Transportation_Mode":
            trip_transport["Transportation_Mode"]
    })

    trptrans_sheet.to_excel(
        writer,
        sheet_name="TRPTRANS",
        index=False
    )

    # --------------------------------------------------------
    # SHEET 5: HHFAMINC
    # Ordinal dimension
    # --------------------------------------------------------
    hhfaminc_sheet = pd.DataFrame({
        "HHFAMINC": income,
        "Income_Category": income.map(income_labels)
    })

    hhfaminc_sheet.to_excel(
        writer,
        sheet_name="HHFAMINC",
        index=False
    )

    # --------------------------------------------------------
    # SHEET 6: TDAYDATE
    # Interval dimension
    # --------------------------------------------------------
    tdaydate_sheet = pd.DataFrame({
        "TDAYDATE":
            travel_year.astype(str) +
            travel_month.astype(str).str.zfill(2)
    })

    tdaydate_sheet.to_excel(
        writer,
        sheet_name="TDAYDATE",
        index=False
    )

    # --------------------------------------------------------
    # SHEET 7: TRPMILES
    # Ratio dimension
    # --------------------------------------------------------
    trpmiles_sheet = pd.DataFrame({
        "TRPMILES": trip_miles
    })

    trpmiles_sheet.to_excel(
        writer,
        sheet_name="TRPMILES",
        index=False
    )


print("\nExcel workbook successfully created:")
print(workbook_name)

print("\n======================================")
print("TASK 6 COMPLETE")
print("======================================")

