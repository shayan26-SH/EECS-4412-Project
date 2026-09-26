import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# EECS 4412 - Phase 1
# Task 4.6: Five Required Visualizations
# 2017 National Household Travel Survey (NHTS)
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


print("===== DATASET SUMMARY =====")

print(
    f"Household: {household.shape[0]:,} instances, "
    f"{household.shape[1]} attributes"
)

print(
    f"Vehicle: {vehicle.shape[0]:,} instances, "
    f"{vehicle.shape[1]} attributes"
)

print(
    f"Trip: {trip.shape[0]:,} instances, "
    f"{trip.shape[1]} attributes"
)


# ============================================================
# 2. CREATE FOLDER FOR SAVED GRAPHS
# ============================================================

output_folder = "Task4_Visualizations"

os.makedirs(output_folder, exist_ok=True)

print("\nGraphs will be saved in:", output_folder)


# ============================================================
# VISUALIZATION 1
# TRPTRANS - BAR CHART
# Transportation Mode Distribution
# ============================================================

print("\nCreating Visualization 1...")

# Convert TRPTRANS to numeric
trip["TRPTRANS_NUM"] = pd.to_numeric(
    trip["TRPTRANS"],
    errors="coerce"
)

# Official 2017 NHTS transportation-mode codes
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

# Keep only valid transportation codes
transport = trip[
    trip["TRPTRANS_NUM"].isin(transport_labels.keys())
].copy()

# Convert code to understandable name
transport["Transportation Mode"] = (
    transport["TRPTRANS_NUM"].map(transport_labels)
)

# Count trips by mode
mode_counts = (
    transport["Transportation Mode"]
    .value_counts()
    .sort_values(ascending=False)
)

# Create graph
plt.figure(figsize=(13, 7))

mode_counts.plot(kind="bar")

plt.title("Figure 1: Distribution of Transportation Modes")
plt.xlabel("Transportation Mode")
plt.ylabel("Number of Trips")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "Figure1_Transportation_Mode_BarChart.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# VISUALIZATION 2
# HHFAMINC - PIE CHART
# Household Income Distribution
# ============================================================

print("Creating Visualization 2...")

household["HHFAMINC_NUM"] = pd.to_numeric(
    household["HHFAMINC"],
    errors="coerce"
)

# Official HHFAMINC income categories
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

# Remove -7, -8, -9 and other invalid values
income = household[
    household["HHFAMINC_NUM"].isin(income_labels.keys())
].copy()

income["Income Category"] = (
    income["HHFAMINC_NUM"].map(income_labels)
)

# Keep categories in correct income order
income_counts = (
    income["HHFAMINC_NUM"]
    .value_counts()
    .sort_index()
)

pie_labels = [
    income_labels[code]
    for code in income_counts.index
]

plt.figure(figsize=(12, 8))

wedges, texts, autotexts = plt.pie(
    income_counts.values,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(
    "Figure 2: Distribution of Household Income Categories"
)

# Put category names in legend instead of directly on pie
plt.legend(
    wedges,
    pie_labels,
    title="Household Income",
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "Figure2_Household_Income_PieChart.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# VISUALIZATION 3
# TRPMILES - HISTOGRAM
# Trip Distance Distribution
# ============================================================

print("Creating Visualization 3...")

trip["TRPMILES_NUM"] = pd.to_numeric(
    trip["TRPMILES"],
    errors="coerce"
)

# Remove missing/special negative values
trip_distance = trip.loc[
    trip["TRPMILES_NUM"] >= 0,
    "TRPMILES_NUM"
].dropna()

print("\nTRPMILES statistics:")
print(trip_distance.describe())

# Calculate 99th percentile
# This is used only to make the graph easier to read.
trip_distance_99 = trip_distance.quantile(0.99)

print(
    f"\n99th percentile of TRPMILES: "
    f"{trip_distance_99:.2f} miles"
)

trip_distance_graph = trip_distance[
    trip_distance <= trip_distance_99
]

plt.figure(figsize=(10, 6))

plt.hist(
    trip_distance_graph,
    bins=40,
    edgecolor="black"
)

plt.title(
    "Figure 3: Distribution of Trip Distance"
)

plt.xlabel("Trip Distance (Miles)")
plt.ylabel("Number of Trips")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "Figure3_Trip_Distance_Histogram.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# VISUALIZATION 4
# ANNMILES - BOX PLOT
# Annual Vehicle Mileage
# ============================================================

print("Creating Visualization 4...")

vehicle["ANNMILES_NUM"] = pd.to_numeric(
    vehicle["ANNMILES"],
    errors="coerce"
)

# Remove negative NHTS special codes
annual_miles = vehicle.loc[
    vehicle["ANNMILES_NUM"] >= 0,
    "ANNMILES_NUM"
].dropna()

print("\nANNMILES statistics:")
print(annual_miles.describe())

plt.figure(figsize=(11, 5))

plt.boxplot(
    annual_miles,
    vert=False
)

plt.title(
    "Figure 4: Distribution of Annual Vehicle Mileage"
)

plt.xlabel("Annual Vehicle Mileage (Miles)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "Figure4_Annual_Mileage_BoxPlot.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# VISUALIZATION 5
# VEHAGE VS ANNMILES - SCATTER PLOT
# ============================================================

print("Creating Visualization 5...")

vehicle["VEHAGE_NUM"] = pd.to_numeric(
    vehicle["VEHAGE"],
    errors="coerce"
)

# Select only valid values
vehicle_scatter = vehicle.loc[
    (vehicle["VEHAGE_NUM"] >= 0) &
    (vehicle["ANNMILES_NUM"] >= 0),
    ["VEHAGE_NUM", "ANNMILES_NUM"]
].dropna()

# Remove only the top 1% of annual mileage values
# for visualization clarity.
mileage_99 = vehicle_scatter[
    "ANNMILES_NUM"
].quantile(0.99)

vehicle_scatter = vehicle_scatter[
    vehicle_scatter["ANNMILES_NUM"] <= mileage_99
]

# There are over 250,000 vehicle records.
# Use a reproducible random sample to prevent overplotting.
if len(vehicle_scatter) > 10000:

    vehicle_scatter_graph = vehicle_scatter.sample(
        n=10000,
        random_state=42
    )

else:
    vehicle_scatter_graph = vehicle_scatter


plt.figure(figsize=(10, 6))

plt.scatter(
    vehicle_scatter_graph["VEHAGE_NUM"],
    vehicle_scatter_graph["ANNMILES_NUM"],
    alpha=0.30,
    s=10
)

plt.title(
    "Figure 5: Vehicle Age vs. Annual Vehicle Mileage"
)

plt.xlabel("Vehicle Age (Years)")
plt.ylabel("Annual Vehicle Mileage (Miles)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "Figure5_VehicleAge_vs_Mileage_ScatterPlot.png"
    ),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n======================================")
print("ALL 5 VISUALIZATIONS CREATED")
print("======================================")

print("\nFiles saved in:", output_folder)

print("""
1. Figure1_Transportation_Mode_BarChart.png
2. Figure2_Household_Income_PieChart.png
3. Figure3_Trip_Distance_Histogram.png
4. Figure4_Annual_Mileage_BoxPlot.png
5. Figure5_VehicleAge_vs_Mileage_ScatterPlot.png
""")