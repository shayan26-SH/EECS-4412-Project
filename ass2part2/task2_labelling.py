from pathlib import Path
import pandas as pd
#households in each class:
#VEHICLE_CLASS
#0      5911
#1     40027
#2     48372
#3+    30988
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "task2_split_data.csv"

data = pd.read_csv(DATA_PATH)

vehicle_count = data["HHVEHCNT"].astype(int)

data["VEHICLE_CLASS"] = vehicle_count.clip(upper=3).map({
    0: "0",
    1: "1",
    2: "2",
    3: "3+",
})
print("households in each class:")
print(data["VEHICLE_CLASS"].value_counts().sort_index())
