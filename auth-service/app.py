from fastapi import FastAPI
import time
import statistics
import uuid
import json
import base64

from pqcrypto.sign import (
    ml_dsa_44,
    ml_dsa_65,
    ml_dsa_87,
    falcon_512,
    falcon_1024,
    sphincs_sha2_128f_simple,
    sphincs_sha2_128s_simple,
    sphincs_sha2_192f_simple,
    sphincs_sha2_192s_simple,
    sphincs_sha2_256f_simple,
    sphincs_sha2_256s_simple
)

from rl.inference import choose_algorithm


app = FastAPI()


ALG_MAP = {

    "mldsa44": (
        ml_dsa_44,
        "ML-DSA-44"
    ),

    "mldsa65": (
        ml_dsa_65,
        "ML-DSA-65"
    ),

    "mldsa87": (
        ml_dsa_87,
        "ML-DSA-87"
    ),

    "falcon512": (
        falcon_512,
        "Falcon-512"
    ),

    "falcon1024": (
        falcon_1024,
        "Falcon-1024"
    ),

    "sphincs128f": (
        sphincs_sha2_128f_simple,
        "SPHINCS+-SHA2-128f"
    ),

    "sphincs128s": (
        sphincs_sha2_128s_simple,
        "SPHINCS+-SHA2-128s"
    ),

    "sphincs192f": (
        sphincs_sha2_192f_simple,
        "SPHINCS+-SHA2-192f"
    ),

    "sphincs192s": (
        sphincs_sha2_192s_simple,
        "SPHINCS+-SHA2-192s"
    ),

    "sphincs256f": (
        sphincs_sha2_256f_simple,
        "SPHINCS+-SHA2-256f"
    ),

    "sphincs256s": (
        sphincs_sha2_256s_simple,
        "SPHINCS+-SHA2-256s"
    )
}


def load_crypto(algorithm):

    if algorithm not in ALG_MAP:
        raise Exception(
            f"Unknown algorithm: {algorithm}"
        )

    ALGO, ALG_NAME = ALG_MAP[algorithm]

    with open(
        f"/app/certs/{algorithm}/private_key.bin",
        "rb"
    ) as f:

        PRIVATE_KEY = f.read()

    return (
        ALGO,
        ALG_NAME,
        PRIVATE_KEY
    )


sign_times = []
token_sizes = []
signature_sizes = []


@app.post("/reset")
def reset():

    global sign_times
    global token_sizes
    global signature_sizes

    sign_times = []
    token_sizes = []
    signature_sizes = []

    return {
        "status": "auth reset done"
    }


@app.post("/login")
def login():

    # Temporary values
    # Later replace with real metrics

    current_users = 250
    security_bits = 192

    cpu_usage = 150
    memory_usage = 70

    latency = 20
    throughput = 70


    selected_algorithm = choose_algorithm(

        current_users,
        security_bits,
        cpu_usage,
        memory_usage,
        latency,
        throughput

    )

    print(
        f"Selected: {selected_algorithm}"
    )

    ALGO, ALG_NAME, PRIVATE_KEY = load_crypto(
        selected_algorithm
    )


    header = {

        "alg": ALG_NAME,
        "typ": "JWT"

    }


    payload = {

        "sub": "user",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
        "jti": str(uuid.uuid4())

    }


    header_b64 = base64.urlsafe_b64encode(
        json.dumps(header).encode()
    ).decode().rstrip("=")


    payload_b64 = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).decode().rstrip("=")


    message = f"{header_b64}.{payload_b64}".encode()


    start = time.perf_counter()

    signature = ALGO.sign(
        PRIVATE_KEY,
        message
    )

    sign_time = (
        time.perf_counter() - start
    ) * 1000


    sig_b64 = base64.urlsafe_b64encode(
        signature
    ).decode().rstrip("=")


    token = (
        f"{header_b64}."
        f"{payload_b64}."
        f"{sig_b64}"
    )


    sign_times.append(
        sign_time
    )

    token_sizes.append(
        len(token)
    )

    signature_sizes.append(
        len(signature)
    )


    return {

        "selected_algorithm":
            selected_algorithm,

        "token":
            token,

        "sign_time":
            sign_time,

        "token_size":
            len(token),

        "signature_size":
            len(signature)
    }


@app.get("/metrics")
def metrics():

    return {
        "sign_avg":
            statistics.mean(sign_times)
            if sign_times else 0,

        "token_size":
            token_sizes[-1]
            if token_sizes else 0,

        "signature_size":
            signature_sizes[-1]
            if signature_sizes else 0
    }