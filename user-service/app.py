from fastapi import FastAPI, Request, HTTPException
import time, statistics, os, json, base64

from pqcrypto.sign import (
    ml_dsa_44, ml_dsa_65, ml_dsa_87,
    falcon_512, falcon_1024,
    sphincs_sha2_128f_simple, sphincs_sha2_128s_simple,
    sphincs_sha2_192f_simple, sphincs_sha2_192s_simple,
    sphincs_sha2_256f_simple, sphincs_sha2_256s_simple
)

app = FastAPI()

KEY_SIZE = os.environ["KEY_SIZE"]
REPLAY_PROTECTION = os.getenv("REPLAY_PROTECTION", "OFF")

ALG_MAP = {
    "mldsa44": ml_dsa_44,
    "mldsa65": ml_dsa_65,
    "mldsa87": ml_dsa_87,
    "falcon512": falcon_512,
    "falcon1024": falcon_1024,
    "sphincs128f": sphincs_sha2_128f_simple,
    "sphincs128s": sphincs_sha2_128s_simple,
    "sphincs192f": sphincs_sha2_192f_simple,
    "sphincs192s": sphincs_sha2_192s_simple,
    "sphincs256f": sphincs_sha2_256f_simple,
    "sphincs256s": sphincs_sha2_256s_simple,
}

if KEY_SIZE not in ALG_MAP:
    raise Exception("Invalid KEY_SIZE")

ALGO = ALG_MAP[KEY_SIZE]

# Load public key
with open(f"/app/certs/{KEY_SIZE}/public_key.bin", "rb") as f:
    PUBLIC_KEY = f.read()

verify_times = []
used_jti = set()

# ✅ Correct base64 decode
def b64decode_fixed(data: str) -> bytes:
    padding = '=' * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)

@app.post("/reset")
def reset():
    global verify_times, used_jti
    verify_times, used_jti = [], set()
    return {"status": "user reset done"}

# 🔥 Accept BOTH GET (header) and POST (body)
@app.api_route("/protected", methods=["GET", "POST"])
async def protected(request: Request):
    token = None

    # 1) Try header (ML-DSA / Falcon)
    auth_header = request.headers.get("Authorization")
    if auth_header and " " in auth_header:
        token = auth_header.split()[1]

    # 2) Try body (SPHINCS)
    if not token:
        try:
            body = await request.json()
            token = body.get("token")
        except:
            pass

    if not token:
        raise HTTPException(401, "Missing token")

    try:
        start = time.perf_counter_ns()

        parts = token.split(".")
        if len(parts) != 3:
            raise HTTPException(401, "Invalid token format")

        header_b64, payload_b64, sig_b64 = parts

        message = f"{header_b64}.{payload_b64}".encode()
        signature = b64decode_fixed(sig_b64)

        ALGO.verify(PUBLIC_KEY, message, signature)

        payload = json.loads(b64decode_fixed(payload_b64))

        verify_time = (time.perf_counter_ns() - start) / 1e6
        verify_times.append(verify_time)

        # Replay protection
        if REPLAY_PROTECTION == "ON":
            jti = payload.get("jti")
            if jti in used_jti:
                raise HTTPException(401, "Replay attack")
            used_jti.add(jti)

        return {"verify_time": verify_time}

    except Exception as e:
        print("VERIFY ERROR:", repr(e))
        raise HTTPException(401, "Invalid token")

@app.get("/metrics/verify")
def verify_metrics():
    return {
        "verify_min": min(verify_times) if verify_times else 0,
        "verify_avg": statistics.mean(verify_times) if verify_times else 0,
        "verify_max": max(verify_times) if verify_times else 0,
        "verify_std": statistics.stdev(verify_times) if len(verify_times) > 1 else 0
    }