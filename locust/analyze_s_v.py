# sign and verifytime min avg std max
import pandas as pd
import glob
import numpy as np

files = glob.glob("logs/results_*_stats.csv")

all_data = []

for file in files:
    df = pd.read_csv(file)

    if "sign_time" in df.columns and "verify_time" in df.columns:
        all_data.append(df[["sign_time", "verify_time"]])

data = pd.concat(all_data)

def stats(x):
    return {
        "min": np.min(x),
        "avg": np.mean(x),
        "max": np.max(x),
        "std": np.std(x)
    }

sign_stats = stats(data["sign_time"])
verify_stats = stats(data["verify_time"])

print("\n📊 SIGN TIME STATS")
print(sign_stats)

print("\n📊 VERIFY TIME STATS")
print(verify_stats)