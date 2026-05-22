from fastapi import FastAPI, Request
import time
import uuid
import json
import base64
import os
import threading
import requests

from rl.inference import choose_algorithm
from rl.reward import calculate_reward
from rl.runtime_buffer import RuntimeBuffer
from rl.online_update import run_online_update
from metrics_collector import MetricsCollector


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

MODEL = os.getenv("RL_MODEL", "Q")

ALG_MAP = {
    "mldsa44": (ml_dsa_44, "ML-DSA-44"),
    "mldsa65": (ml_dsa_65, "ML-DSA-65"),
    "mldsa87": (ml_dsa_87, "ML-DSA-87"),
    "falcon512": (falcon_512, "Falcon-512"),
    "falcon1024": (falcon_1024, "Falcon-1024"),
    "sphincs128f": (sphincs_sha2_128f_simple, "SPHINCS+-SHA2-128f"),
    "sphincs128s": (sphincs_sha2_128s_simple, "SPHINCS+-SHA2-128s"),
    "sphincs192f": (sphincs_sha2_192f_simple, "SPHINCS+-SHA2-192f"),
    "sphincs192s": (sphincs_sha2_192s_simple, "SPHINCS+-SHA2-192s"),
    "sphincs256f": (sphincs_sha2_256f_simple, "SPHINCS+-SHA2-256f"),
    "sphincs256s": (sphincs_sha2_256s_simple, "SPHINCS+-SHA2-256s"),
}

# Global instances
collector = MetricsCollector()
buffer = RuntimeBuffer()
key_cache = {}

# Online learning control variables
update_lock = threading.Lock()
request_counter = 0
last_update_time = time.time()

def check_and_trigger_update():
    global request_counter, last_update_time
    should_update = False
    with update_lock:
        request_counter += 1
        now = time.time()
        # Trigger update every 100 requests or 5 minutes (300 seconds)
        if request_counter >= 100 or (now - last_update_time) >= 300:
            request_counter = 0
            last_update_time = now
            should_update = True
            
    if should_update:
        # Run update asynchronously in a background thread to prevent blocking requests
        threading.Thread(target=run_online_update, daemon=True).start()

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL
    }

@app.get("/metrics/crypto")
def get_metrics():
    return collector.get_compiled_metrics()

@app.post("/reset")
def reset_metrics():
    global collector, request_counter
    # Clear sliding windows and reset counters
    with collector._lock:
        collector.latency_window.clear()
        collector.request_timestamps.clear()
        collector.total_requests = 0
        collector.successful_requests = 0
        collector.algorithm_switches = 0
        collector.adaptation_attempts = 0
        collector.successful_adaptations = 0
        collector.security_satisfactions.clear()
        collector.switching_times.clear()
        collector.service_interruptions.clear()
        collector.previous_algorithm = None
        collector.previous_reward = None
    with update_lock:
        request_counter = 0
    return {"status": "metrics reset done"}

@app.post("/login")
def login(request: Request):
    start_request = time.perf_counter()
    
    # 1. Collect current metrics to formulate the current state
    metrics = collector.get_compiled_metrics()
    cpu_m = metrics["cpu_avg_m"]
    mem_mi = metrics["memory_avg_mi"]
    latency_ms = metrics["latency_ms"]
    throughput_rps = metrics["throughput_rps"]
    
    # Read security requirement from request headers, fallback to 128
    security_req_str = request.headers.get("X-Security-Requirement", "128")
    try:
        security_req = float(security_req_str)
    except ValueError:
        security_req = 128.0
        
    # Classify phase based on throughput
    if throughput_rps < 50.0:
        phase = 2.0  # low load
    elif throughput_rps > 200.0:
        phase = 1.0  # high load
    else:
        phase = 0.0  # normal load

    # 2. RL inference engine selects algorithm
    # Security bits mapping for model input
    sec_bits_map = {128.0: 128.0, 192.0: 192.0, 256.0: 256.0}
    model_sec_bits = sec_bits_map.get(security_req, 128.0)
    
    # Get dynamic decision from RL model
    KEY, heuristic_reward = choose_algorithm(
        cpu_m,
        mem_mi,
        latency_ms,
        throughput_rps,
        model_sec_bits,
        phase,
        MODEL
    )
    
    # Ensure selected algorithm is supported
    if KEY not in ALG_MAP:
        KEY = "mldsa44"
    ALGO, ALG_NAME = ALG_MAP[KEY]
    
    # 3. Load private key with switching cost tracking
    start_switch = time.perf_counter()
    switched = False
    
    if KEY not in key_cache:
        # Load key from disk
        with open(f"/app/certs/{KEY}/private_key.bin", "rb") as f:
            key_cache[KEY] = f.read()
        switched = True
        
    PRIVATE_KEY = key_cache[KEY]
    
    if switched:
        switch_time_ms = (time.perf_counter() - start_switch) * 1000.0
        service_interruption_ms = switch_time_ms * 1.5
    else:
        switch_time_ms = 0.0
        service_interruption_ms = 0.0
        
    # 4. Generate JWT Signature
    payload = {
        "sub": "user",
        "iat": int(time.time()),
        "jti": str(uuid.uuid4())
    }
    msg = json.dumps(payload).encode()
    
    start_sign = time.perf_counter()
    signature = ALGO.sign(PRIVATE_KEY, msg)
    sign_time_ms = (time.perf_counter() - start_sign) * 1000.0
    
    # Construct base64 encoded token parts
    header = {"alg": ALG_NAME, "typ": "JWT"}
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(msg).decode().rstrip("=")
    sig_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")
    token = f"{header_b64}.{payload_b64}.{sig_b64}"
    
    # 5. Measure request latency and record metrics
    request_latency_ms = (time.perf_counter() - start_request) * 1000.0
    
    # Map selected security level
    sec_bits_selected = 128.0
    if "192" in KEY or "65" in KEY:
        sec_bits_selected = 192.0
    elif "256" in KEY or "87" in KEY or "1024" in KEY:
        sec_bits_selected = 256.0
        
    # 6. Compute reward using standard formula
    reward = calculate_reward(
        throughput=throughput_rps,
        latency=request_latency_ms,
        cpu=cpu_m,
        memory=mem_mi,
        security_satisfaction=(100.0 if sec_bits_selected >= security_req else (sec_bits_selected / security_req * 100.0)),
        jwt_continuity=metrics["jwt_continuity"],
        runtime_adaptability=metrics["runtime_adaptability"],
        switching_time=switch_time_ms,
        service_interruption=service_interruption_ms,
        pqc_adoption_score=metrics["pqc_adoption"]
    )
    
    # Record current experience in the MetricsCollector
    collector.record_request(
        latency_ms=request_latency_ms,
        is_success=True,
        algorithm=KEY,
        security_required=security_req,
        security_selected=sec_bits_selected,
        reward=reward,
        switch_time_ms=switch_time_ms,
        service_interruption_ms=service_interruption_ms
    )
    
    # 7. Write transition to experience buffer
    # Re-evaluate state after request to capture transition next_state
    next_metrics = collector.get_compiled_metrics()
    
    state_dict = {
        "cpu": cpu_m,
        "memory": mem_mi,
        "latency": request_latency_ms,
        "throughput": throughput_rps,
        "request_rate": throughput_rps,
        "security_requirement": security_req,
        "network_load": throughput_rps * (len(token) / 1024.0), # Network load based on request size
        "runtime_adaptability": metrics["runtime_adaptability"],
        "jwt_continuity": metrics["jwt_continuity"],
        "switching_time": switch_time_ms,
        "service_interruption": service_interruption_ms,
        "pqc_adoption": metrics["pqc_adoption"]
    }
    
    next_state_dict = {
        "cpu": next_metrics["cpu_avg_m"],
        "memory": next_metrics["memory_avg_mi"],
        "latency": next_metrics["latency_ms"],
        "throughput": next_metrics["throughput_rps"],
        "request_rate": next_metrics["throughput_rps"],
        "security_requirement": security_req,
        "network_load": next_metrics["throughput_rps"] * (len(token) / 1024.0),
        "runtime_adaptability": next_metrics["runtime_adaptability"],
        "jwt_continuity": next_metrics["jwt_continuity"],
        "switching_time": next_metrics["switching_time"],
        "service_interruption": next_metrics["service_interruption"],
        "pqc_adoption": next_metrics["pqc_adoption"]
    }
    
    buffer.add(state_dict, KEY, reward, next_state_dict)
    
    # 7.5 Log to runtime_metrics.csv
    metrics_path = "/app/rl/results/runtime_metrics.csv"
    if not os.path.exists(metrics_path):
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            f.write("timestamp,cpu,memory,latency,throughput,jwt_continuity,security_satisfaction,switching_time,service_interruption,pqc_adoption\n")
    with open(metrics_path, "a") as f:
        f.write(f"{time.time()},{cpu_m},{mem_mi},{request_latency_ms},{throughput_rps},{metrics['jwt_continuity']},{100.0 if sec_bits_selected >= security_req else (sec_bits_selected / security_req * 100.0)},{switch_time_ms},{service_interruption_ms},{metrics['pqc_adoption']}\n")
    
    # 8. Check and trigger online learning incremental update
    check_and_trigger_update()
    
    return {
        "token": token,
        "model": MODEL,
        "algorithm": KEY,
        "reward": reward,
        "sign_time": sign_time_ms
    }

@app.post("/verify")
def verify(request: Request):

    try:

        auth_header = request.headers.get(
            "Authorization"
        )

        if not auth_header:
            return {
                "verified": False,
                "error": "Missing token"
            }

        token = auth_header.split()[1]

        parts = token.split(".")

        if len(parts) != 3:
            return {
                "verified": False,
                "error": "Invalid token"
            }

        header_b64, payload_b64, sig_b64 = parts

        # Decode JWT header
        header = json.loads(
            base64.urlsafe_b64decode(
                header_b64 + "=" * (-len(header_b64) % 4)
            )
        )

        algorithm = header["alg"]

        # convert display names → user-service names
        algo_map = {
            "ML-DSA-44":"mldsa44",
            "ML-DSA-65":"mldsa65",
            "ML-DSA-87":"mldsa87",
            "Falcon-512":"falcon512",
            "Falcon-1024":"falcon1024",
            "SPHINCS+-SHA2-128f":"sphincs128f",
            "SPHINCS+-SHA2-128s":"sphincs128s",
            "SPHINCS+-SHA2-192f":"sphincs192f",
            "SPHINCS+-SHA2-192s":"sphincs192s",
            "SPHINCS+-SHA2-256f":"sphincs256f",
            "SPHINCS+-SHA2-256s":"sphincs256s"
        }

        algorithm = algo_map[algorithm]

        message = (
            payload_b64
        )

        signature = sig_b64

        response = requests.post(
            "http://user-service:8000/verify",
            json={
                "algorithm": algorithm,
                "message": message,
                "signature": signature
            }
        )

        return response.json()

    except Exception as e:

        return {
            "verified": False,
            "error": str(e)
        }