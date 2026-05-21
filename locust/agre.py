import pandas as pd
import glob
import os
import re

BASE_DIR = "logs/reuse"   # or logs/reuse

results_crypto = []
results_system = []
results_resource = []

# =========================
# HELPERS
# =========================

def extract_users(file):
    return int(re.search(r'u(\d+)', file).group(1))

def parse_size(value):
    if pd.isna(value):
        return 0.0

    if not isinstance(value, str):
        try:
            return float(value)
        except:
            return 0.0

    value = value.strip()

    try:
        if "MB" in value:
            return float(value.replace("MB", ""))
        elif "kB" in value or "KB" in value:
            return float(value.replace("kB", "").replace("KB", "")) / 1024
        elif "GB" in value:
            return float(value.replace("GB", "")) * 1024
        elif "B" in value:
            return float(value.replace("B", "")) / (1024 * 1024)
        else:
            return float(value)
    except:
        return 0.0

# =========================
# LOOP ALL MLDSA KEYS
# =========================

for key_path in glob.glob(f"{BASE_DIR}/*"):
    key = key_path.split("/")[-1]

    # =========================
    # CRYPTO METRICS
    # =========================
    auth_files = glob.glob(f"{key_path}/auth_*.csv")

    for a_file in auth_files:
        users = extract_users(a_file)
        v_file = a_file.replace("auth", "verify")

        if not os.path.exists(v_file):
            continue

        auth_df = pd.read_csv(a_file)
        verify_df = pd.read_csv(v_file)

        results_crypto.append({
            "algorithm": key,
            "users": users,
            "sign_min": auth_df["sign_min"][0],
            "sign_avg": auth_df["sign_avg"][0],
            "sign_max": auth_df["sign_max"][0],
            "verify_min": verify_df["verify_min"][0],
            "verify_avg": verify_df["verify_avg"][0],
            "verify_max": verify_df["verify_max"][0],
            "token_size": auth_df["token_size"][0],
            "signature_size": auth_df["signature_size"][0]
        })

    # =========================
    # SYSTEM METRICS
    # =========================
    stat_files = glob.glob(f"{key_path}/locust_*_stats.csv")

    for s_file in stat_files:
        users = extract_users(s_file)

        df = pd.read_csv(s_file)

        if "Name" not in df.columns:
            continue

        agg = df[df["Name"] == "Aggregated"]

        if agg.empty:
            continue

        row = agg.iloc[0]

        results_system.append({
            "algorithm": key,
            "users": users,
            "throughput": row.get("Requests/s", 0),
            "avg_latency": row.get("Average Response Time", 0),
            "p50": row.get("50%", 0),
            "p95": row.get("95%", 0),
            "p99": row.get("99%", 0),
            "error_rate": row.get("Failure Count", 0) / row.get("Request Count", 1)
        })

    # =========================
    # RESOURCE METRICS
    # =========================
    res_files = glob.glob(f"{key_path}/resource_*.csv")

    for r_file in res_files:
        users = extract_users(r_file)

        df = pd.read_csv(r_file)

        df["CPU%"] = pd.to_numeric(df["CPU%"].astype(str).str.replace("%", ""), errors="coerce").fillna(0)

        df["Mem"] = pd.to_numeric(
            df["Memory"].astype(str).str.extract(r'(\d+\.\d+)')[0],
            errors="coerce"
        ).fillna(0)

        df["NetIO"] = df["NetIO"].astype(str)
        net_split = df["NetIO"].str.split("/", expand=True)

        df["Net_In"] = net_split[0].apply(parse_size)
        df["Net_Out"] = net_split[1].apply(parse_size)

        df["BlockIO"] = df["BlockIO"].astype(str)
        blk_split = df["BlockIO"].str.split("/", expand=True)

        df["Blk_In"] = blk_split[0].apply(parse_size)
        df["Blk_Out"] = blk_split[1].apply(parse_size)

        cpu_avg = df["CPU%"].mean()
        cpu_peak = df["CPU%"].max()

        mem_avg = df["Mem"].mean()
        mem_peak = df["Mem"].max()

        net_in = df["Net_In"].max()
        net_out = df["Net_Out"].max()

        blk_in = df["Blk_In"].max()
        blk_out = df["Blk_Out"].max()

        system_match = next(
            (s for s in results_system if s["algorithm"] == key and s["users"] == users),
            None
        )

        efficiency = 0
        if system_match and cpu_avg > 0:
            efficiency = system_match["throughput"] / cpu_avg

        results_resource.append({
            "algorithm": key,
            "users": users,
            "cpu_avg": cpu_avg,
            "cpu_peak": cpu_peak,
            "mem_avg": mem_avg,
            "mem_peak": mem_peak,
            "net_in_MB": net_in,
            "net_out_MB": net_out,
            "blk_in_MB": blk_in,
            "blk_out_MB": blk_out,
            "efficiency": efficiency
        })

# =========================
# SAVE OUTPUT
# =========================

os.makedirs("final_results", exist_ok=True)

pd.DataFrame(results_crypto).to_csv("final_results/table_crypto_mldsa.csv", index=False)
pd.DataFrame(results_system).to_csv("final_results/table_system_mldsa.csv", index=False)
pd.DataFrame(results_resource).to_csv("final_results/table_resource_mldsa.csv", index=False)

print("✅ MLDSA FINAL TABLES GENERATED → final_results/")