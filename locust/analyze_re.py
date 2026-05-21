import glob
import re
import pandas as pd

# =========================
# HELPERS
# =========================

def avg(arr):
    return sum(arr) / len(arr) if arr else 0


def parse_size(value):
    """
    Convert Docker stats units → MB
    Supports: B, KB, MB, GB
    """
    value = value.strip().lower()

    try:
        num = float(re.findall(r"[0-9.]+", value)[0])

        if "gb" in value:
            return num * 1024
        elif "mb" in value:
            return num
        elif "kb" in value:
            return num / 1024
        elif "b" in value:
            return num / (1024 * 1024)
        else:
            return num
    except:
        return 0


# =========================
# LOAD FILES
# =========================

files = sorted(glob.glob("logs/resource_*.csv"))

results = []

# =========================
# PROCESS EACH FILE
# =========================

for file in files:

    cpu = []
    mem = []
    net_in = []
    net_out = []
    blk_in = []
    blk_out = []

    with open(file) as f:
        for line in f:
            line = line.strip()

            # skip header or empty or ANSI garbage
            if not line or "container" in line.lower():
                continue

            # remove ANSI escape codes
            line = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', line)

            parts = line.split(",")

            if len(parts) < 5:
                continue

            try:
                # CPU
                cpu.append(float(parts[1].replace("%", "").strip()))

                # MEMORY (MiB part only)
                mem_used = parts[2].split("/")[0].strip().lower()
                mem.append(parse_size(mem_used))

                # NETWORK
                net = parts[3].split("/")
                if len(net) == 2:
                    net_in.append(parse_size(net[0]))
                    net_out.append(parse_size(net[1]))

                # BLOCK IO
                blk = parts[4].split("/")
                if len(blk) == 2:
                    blk_in.append(parse_size(blk[0]))
                    blk_out.append(parse_size(blk[1]))

            except:
                continue

    # =========================
    # EXTRACT CONCURRENCY
    # =========================
    match = re.search(r"resource_(\d+)", file)
    concurrency = int(match.group(1)) if match else 0

    # =========================
    # STORE RESULT
    # =========================
    results.append({
        "concurrency": concurrency,

        # CPU
        "cpu_avg": avg(cpu),
        "cpu_max": max(cpu) if cpu else 0,

        # Memory
        "mem_avg_mb": avg(mem),
        "mem_max_mb": max(mem) if mem else 0,

        # Network (MB)
        "net_in_mb": sum(net_in),
        "net_out_mb": sum(net_out),

        # Block IO (MB)
        "blk_in_mb": sum(blk_in),
        "blk_out_mb": sum(blk_out),
    })


# =========================
# FINAL DATAFRAME
# =========================

df = pd.DataFrame(results)

if df.empty:
    print("❌ No resource data found in logs/")
    exit()

df = df.sort_values("concurrency")

# =========================
# SAVE OUTPUT
# =========================

df.to_csv("logs/resource_summary.csv", index=False)

print("\n✅ RESOURCE SUMMARY GENERATED\n")
print(df)