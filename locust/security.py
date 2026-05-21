import requests
import base64
import json
import time
import csv
import os
import subprocess

from pqcrypto.sign import ml_dsa_44, ml_dsa_65, ml_dsa_87

AUTH = "http://localhost:8001"
USER = "http://localhost:8002"
GATEWAY = "http://localhost:8080"

TOTAL = 50

KEY_VARIANTS = ["mldsa44", "mldsa65", "mldsa87"]

# -------------------------
# CSV SETUP
# -------------------------
os.makedirs("logs/security", exist_ok=True)
file_path = "logs/security/security_results.csv"
write_header = not os.path.exists(file_path)

csv_file = open(file_path, "a", newline="")
writer = csv.writer(csv_file)

if write_header:
    writer.writerow(["key_size", "TDR", "FDR", "RSR", "ASR"])

# =========================
# MAIN LOOP
# =========================
for KEY_SIZE in KEY_VARIANTS:

    print(f"\n🚀 TESTING {KEY_SIZE}\n")

    # -------------------------
    # SELECT ALGO
    # -------------------------
    if KEY_SIZE == "mldsa44":
        ALGO = ml_dsa_44
    elif KEY_SIZE == "mldsa65":
        ALGO = ml_dsa_65
    elif KEY_SIZE == "mldsa87":
        ALGO = ml_dsa_87

    # -------------------------
    # RESTART DOCKER WITH KEY
    # -------------------------
    subprocess.run("docker compose down", shell=True)
    subprocess.run(f"KEY_SIZE={KEY_SIZE} docker compose up -d --build", shell=True)

    time.sleep(5)

    # -------------------------
    # RESET SYSTEM
    # -------------------------
    requests.post(f"{AUTH}/reset")
    requests.post(f"{USER}/reset")

    tdr_detected = 0
    fdr_detected = 0
    rsr_success = 0

    # -------------------------
    # LOAD FAKE KEY
    # -------------------------
    FAKE_KEY = os.path.join(
        "../certs",
        "mldsa87" if KEY_SIZE != "mldsa87" else "mldsa65",
        "private_key.bin"
    )

    with open(FAKE_KEY, "rb") as f:
        fake_sk = f.read()

    # -------------------------
    # HELPER
    # -------------------------
    def tamper(token):
        msg_b64, sig_b64 = token.split(".")
        message = json.loads(base64.urlsafe_b64decode(msg_b64 + "=="))

        message["admin"] = True

        new_msg = base64.urlsafe_b64encode(
            json.dumps(message).encode()
        ).decode().rstrip("=")

        return f"{new_msg}.{sig_b64}"

    # -------------------------
    # TEST LOOP
    # -------------------------
    for _ in range(TOTAL):

        res = requests.post(f"{AUTH}/login")
        token = res.json()["token"]

        # 1️⃣ TAMPER
        tampered = tamper(token)
        r = requests.get(
            f"{GATEWAY}/protected",
            headers={"Authorization": f"Bearer {tampered}"}
        )

        if r.status_code != 200:
            tdr_detected += 1

        # 2️⃣ FORGERY
        try:
            fake_payload = {"user": "attacker", "iat": int(time.time())}
            message = json.dumps(fake_payload).encode()

            fake_sig = ALGO.sign(fake_sk, message)

            fake_token = (
                base64.urlsafe_b64encode(message).decode().rstrip("=")
                + "."
                + base64.urlsafe_b64encode(fake_sig).decode().rstrip("=")
            )

            r = requests.get(
                f"{GATEWAY}/protected",
                headers={"Authorization": f"Bearer {fake_token}"}
            )

            if r.status_code != 200:
                fdr_detected += 1

        except Exception:
            fdr_detected += 1

        # 3️⃣ REPLAY
        r1 = requests.get(
            f"{GATEWAY}/protected",
            headers={"Authorization": f"Bearer {token}"}
        )

        r2 = requests.get(
            f"{GATEWAY}/protected",
            headers={"Authorization": f"Bearer {token}"}
        )

        if r2.status_code == 200:
            rsr_success += 1

    # -------------------------
    # METRICS
    # -------------------------
    TDR = tdr_detected / TOTAL
    FDR = fdr_detected / TOTAL
    RSR = rsr_success / TOTAL
    ASR = rsr_success / (TOTAL * 3)

    print(f"KEY: {KEY_SIZE}")
    print(f"TDR: {TDR:.4f}")
    print(f"FDR: {FDR:.4f}")
    print(f"RSR: {RSR:.4f}")
    print(f"ASR: {ASR:.4f}")

    writer.writerow([KEY_SIZE, TDR, FDR, RSR, ASR])

csv_file.close()

print("\n✅ ALL TESTS COMPLETED")