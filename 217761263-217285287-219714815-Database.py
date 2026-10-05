import pandas as pd  # type: ignore[import-not-found]

# Load the three NHTS files you are using
household = pd.read_csv("hhpub_202609201846.csv", low_memory=False)
vehicle = pd.read_csv("vehpub_202609201846.csv", low_memory=False)
trip = pd.read_csv("trippub_202609201847.csv", low_memory=False)

# Print dataset sizes
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

# Print all column names
print("\n===== HOUSEHOLD COLUMNS =====")
print(household.columns.tolist())

print("\n===== VEHICLE COLUMNS =====")
print(vehicle.columns.tolist())

print("\n===== TRIP COLUMNS =====")
print(trip.columns.tolist())