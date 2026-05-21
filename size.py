import os
import csv

BASE_DIR = "certs"

results = []

print("\n=== PQC Key Sizes ===\n")
print(f"{'Algorithm':<20} {'Public (bytes)':<15} {'Private (bytes)':<15}")

for folder in sorted(os.listdir(BASE_DIR)):
    path = os.path.join(BASE_DIR, folder)

    if os.path.isdir(path):
        pub_path = os.path.join(path, "public_key.bin")
        priv_path = os.path.join(path, "private_key.bin")

        if os.path.exists(pub_path) and os.path.exists(priv_path):
            pub_size = os.path.getsize(pub_path)
            priv_size = os.path.getsize(priv_path)

            print(f"{folder:<20} {pub_size:<15} {priv_size:<15}")

            results.append([folder, pub_size, priv_size])

# Save to CSV (for paper/analysis)
with open("key_sizes.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Algorithm", "Public Key (bytes)", "Private Key (bytes)"])
    writer.writerows(results)

print("\n[✓] Saved results to key_sizes.csv")