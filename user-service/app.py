from fastapi import FastAPI
import time
import statistics
import base64
import os

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

app = FastAPI()

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
    "sphincs256s": sphincs_sha2_256s_simple
}

verify_times=[]


@app.get("/health")
def health():

    return{
        "status":"healthy"
    }


@app.post("/verify")
def verify(data:dict):

    try:

        algorithm=data["algorithm"]

        if algorithm not in ALG_MAP:

            raise Exception(
                f"{algorithm} not available"
            )

        algo=ALG_MAP[algorithm]

        message=data["message"].encode()

        sig=data["signature"]

        padding='=' * (
            -len(sig)%4
        )

        signature=base64.urlsafe_b64decode(
            sig+padding
        )

        cert_path=(
            f"/app/certs/"
            f"{algorithm}/"
            f"public_key.bin"
        )

        if not os.path.exists(
            cert_path
        ):
            raise Exception(
                f"Public key missing: {cert_path}"
            )

        with open(
            cert_path,
            "rb"
        ) as f:

            public_key=f.read()

        start=time.perf_counter_ns()

        algo.verify(
            public_key,
            message,
            signature
        )

        verify_time=(
            time.perf_counter_ns()
            -start
        )/1e6

        verify_times.append(
            verify_time
        )

        return{

            "verified":True,
            "algorithm":algorithm,
            "verify_time":verify_time
        }

    except Exception as e:

        return{

            "verified":False,
            "error":str(e)
        }


@app.get("/metrics/verify")
def metrics():

    return{

        "verify_min":
        min(verify_times)
        if verify_times else 0,

        "verify_avg":
        statistics.mean(
            verify_times
        )
        if verify_times else 0,

        "verify_max":
        max(verify_times)
        if verify_times else 0,

        "requests":
        len(
            verify_times
        )
    }