import pandas as pd
import requests
import os

KEY_SIZES = [512, 1024, 2048, 3072, 4096]
CONCURRENCY = [1]

BASE_DIR = "logs"
OUTPUT_FILE = os.path.join(BASE_DIR, "final_results.csv")

results = []

for key in KEY_SIZES:
    for users in CONCURRENCY:

        print(f"\n📊 Processing KEY={key}, USERS={users}")

        # -------------------------
        # LOCUST STATS
        # -------------------------
        stats_file = f"{BASE_DIR}/results_k{key}_u{users}_stats.csv"

        if not os.path.exists(stats_file):
            print(f"❌ Missing file: {stats_file}")
            continue

        df = pd.read_csv(stats_file)

        agg = df[df["Name"] == "Aggregated"]
        if agg.empty:
            print("❌ No Aggregated row found")
            continue

        agg = agg.iloc[0]

        system_metrics = {
            "throughput": agg.get("Requests/s", 0),
            "avg_latency": agg.get("Average Response Time", 0),
            "p50": agg.get("50%", 0),
            "p95": agg.get("95%", 0),
            "p99": agg.get("99%", 0),
            "error_count": agg.get("Failure Count", 0)
        }

        # -------------------------
        # CRYPTO METRICS
        # -------------------------
        try:
            crypto = requests.get("http://localhost:8001/metrics/crypto", timeout=5).json()
        except Exception:
            crypto = {}

        try:
            verify = requests.get("http://localhost:8002/metrics/verify", timeout=5).json()
        except Exception:
            verify = {}

        # -------------------------
        # RESOURCE METRICS (optional aggregation)
        # -------------------------
        resource_file = f"{BASE_DIR}/resource_k{key}_u{users}.csv"

        cpu_avg = mem_avg = 0

        if os.path.exists(resource_file):
            rdf = pd.read_csv(resource_file)

            def extract_cpu(x):
                try:
                    return float(str(x).replace("%", ""))
                except:
                    return 0

            def extract_mem(x):
                try:
                    return float(str(x).split("/")[0].replace("MiB", "").replace("GiB", ""))
                except:
                    return 0

            if "CPU%" in rdf.columns:
                cpu_avg = rdf["CPU%"].apply(extract_cpu).mean()

            if "Memory" in rdf.columns:
                mem_avg = rdf["Memory"].apply(extract_mem).mean()

        # -------------------------
        # FINAL MERGE
        # -------------------------
        final = {
            "key_size": key,
            "users": users,

            # system metrics
            **system_metrics,

            # crypto metrics
            "sign_min": crypto.get("sign_min"),
            "sign_avg": crypto.get("sign_avg"),
            "sign_max": crypto.get("sign_max"),
            "sign_std": crypto.get("sign_std"),
            "token_size": crypto.get("token_size"),
            "signature_size": crypto.get("signature_size"),

            # verify metrics
            "verify_min": verify.get("verify_min"),
            "verify_avg": verify.get("verify_avg"),
            "verify_max": verify.get("verify_max"),
            "verify_std": verify.get("verify_std"),

            # resource metrics
            "cpu_avg": cpu_avg,
            "mem_avg": mem_avg,

            # metadata
            "algorithm": "RS256"
        }

        results.append(final)

        print(f"✅ Done KEY={key}, USERS={users}")

# -------------------------
# SAVE FINAL DATASET
# -------------------------
df_final = pd.DataFrame(results)
df_final.to_csv(OUTPUT_FILE, index=False)

print("\n🎉 FINAL BENCHMARK COMPLETE")
print(f"📁 Saved to: {OUTPUT_FILE}")