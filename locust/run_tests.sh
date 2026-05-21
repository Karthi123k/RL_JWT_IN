#!/bin/bash

set -euo pipefail

# ==========================================================
# CONFIG
# ==========================================================

MODE="reuse"
DURATION="60s"

CONCURRENCY=(
1
50
100
250
500
)

KEY_SIZES=(
mldsa44
mldsa65
mldsa87
falcon512
falcon1024
sphincs128f
sphincs128s
sphincs192f
sphincs192s
sphincs256f
sphincs256s
)

declare -A ALG_MAP=(

[mldsa44]="ML-DSA-44"
[mldsa65]="ML-DSA-65"
[mldsa87]="ML-DSA-87"

[falcon512]="Falcon-512"
[falcon1024]="Falcon-1024"

[sphincs128f]="SPHINCS+-SHA2-128f-simple"
[sphincs128s]="SPHINCS+-SHA2-128s-simple"

[sphincs192f]="SPHINCS+-SHA2-192f-simple"
[sphincs192s]="SPHINCS+-SHA2-192s-simple"

[sphincs256f]="SPHINCS+-SHA2-256f-simple"
[sphincs256s]="SPHINCS+-SHA2-256s-simple"
)

# ==========================================================
# SECURITY INFO
# ==========================================================

declare -A SECURITY_LEVEL=(

[mldsa44]=2
[mldsa65]=3
[mldsa87]=5

[falcon512]=1
[falcon1024]=5

[sphincs128f]=1
[sphincs128s]=1

[sphincs192f]=3
[sphincs192s]=3

[sphincs256f]=5
[sphincs256s]=5
)

declare -A SECURITY_BITS=(

[mldsa44]=128
[mldsa65]=192
[mldsa87]=256

[falcon512]=128
[falcon1024]=256

[sphincs128f]=128
[sphincs128s]=128

[sphincs192f]=192
[sphincs192s]=192

[sphincs256f]=256
[sphincs256s]=256
)

# ==========================================================
# FILES
# ==========================================================

LOCUST_FILE="locust_reuse.py"

BASE_DIR="logs/$MODE"

mkdir -p "$BASE_DIR"

OVERALL_FILE="$BASE_DIR/overall_benchmark.csv"

echo "algorithm,key_size,nist_level,security_bits,users,sign_avg_ms,verify_avg_ms,token_size,signature_size,latency_ms,throughput_rps,failures,cpu_avg_m,cpu_peak_m,memory_avg_mi,memory_peak_mi" \
> "$OVERALL_FILE"

MONITOR_PID=""

cleanup() {

    if [[ -n "$MONITOR_PID" ]]
    then
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

echo ""
echo "🚀 PQC BENCHMARK STARTED"
echo ""

# ==========================================================
# MAIN LOOP
# ==========================================================

for KEY_SIZE in "${KEY_SIZES[@]}"
do

    OQS_ALG="${ALG_MAP[$KEY_SIZE]}"
    NIST="${SECURITY_LEVEL[$KEY_SIZE]}"
    SEC_BITS="${SECURITY_BITS[$KEY_SIZE]}"

    echo ""
    echo "=================================="
    echo "Algorithm : $OQS_ALG"
    echo "Key       : $KEY_SIZE"
    echo "Security  : $NIST"
    echo "=================================="

    KEY_DIR="$BASE_DIR/$KEY_SIZE"

    mkdir -p "$KEY_DIR"

    BENCH_FILE="$KEY_DIR/benchmark_summary.csv"

    cp "$OVERALL_FILE" "$BENCH_FILE"

    # ======================================================
    # UPDATE DEPLOYMENTS
    # ======================================================

    kubectl set env deployment/auth-service \
    KEY_SIZE="$KEY_SIZE" \
    OQS_ALG="$OQS_ALG" \
    -n pqc-jwt

    kubectl set env deployment/user-service \
    KEY_SIZE="$KEY_SIZE" \
    OQS_ALG="$OQS_ALG" \
    -n pqc-jwt

    kubectl rollout status deployment/auth-service \
    -n pqc-jwt \
    --timeout=180s

    kubectl rollout status deployment/user-service \
    -n pqc-jwt \
    --timeout=180s

    echo "Waiting gateway..."

    sleep 10

    until curl -s \
    -X POST \
    http://localhost:8080/login \
    >/dev/null
    do
        sleep 2
    done

    # ======================================================
    # USERS LOOP
    # ======================================================

    for USERS in "${CONCURRENCY[@]}"
    do

        echo ""
        echo "Running USERS=$USERS"

        RESULT_PREFIX="$KEY_DIR/u${USERS}"

        curl -s -X POST http://localhost:8001/reset >/dev/null || true
        curl -s -X POST http://localhost:8002/reset >/dev/null || true

        RESOURCE_FILE="$KEY_DIR/resource_u${USERS}.csv"

        echo "timestamp,cpu_m,memory_mi" \
        > "$RESOURCE_FILE"

        (
        while true
        do

            DATA=$(kubectl top pod \
            -n pqc-jwt \
            --no-headers | \
            grep -E "auth|user" || true)

            CPU=$(echo "$DATA" | \
            awk '{
            gsub("m","",$2)
            sum+=$2
            }
            END{
            print sum+0
            }')

            MEM=$(echo "$DATA" | \
            awk '{
            gsub("Mi","",$3)
            sum+=$3
            }
            END{
            print sum+0
            }')

            echo "$(date '+%F %T'),$CPU,$MEM"

            sleep 2

        done
        ) >> "$RESOURCE_FILE" &

        MONITOR_PID=$!

        # ======================================================
        # RUN LOCUST
        # ======================================================

        locust \
        -f "$LOCUST_FILE" \
        -H http://localhost:8080 \
        --headless \
        -u "$USERS" \
        -r "$USERS" \
        --run-time "$DURATION" \
        --stop-timeout 5 \
        --csv="$RESULT_PREFIX"

        kill "$MONITOR_PID" 2>/dev/null || true

        MONITOR_PID=""

        # ======================================================
        # SECURITY METRICS
        # ======================================================

        AUTH=$(curl -s http://localhost:8080/metrics/crypto)

        VERIFY=$(curl -s http://localhost:8080/metrics/verify)

        SIGN_AVG=$(echo "$AUTH" | jq -r '.sign_avg // 0')
        VERIFY_AVG=$(echo "$VERIFY" | jq -r '.verify_avg // 0')

        TOKEN_SIZE=$(echo "$AUTH" | jq -r '.token_size // 0')
        SIGNATURE_SIZE=$(echo "$AUTH" | jq -r '.signature_size // 0')

        # ======================================================
        # LOCUST METRICS
        # ======================================================

        STATS="${RESULT_PREFIX}_stats.csv"

        AGG=$(grep "Aggregated" "$STATS" | tail -1)

        FAILS=$(echo "$AGG" | cut -d',' -f4)
        LATENCY=$(echo "$AGG" | cut -d',' -f6)
        RPS=$(echo "$AGG" | cut -d',' -f9)

        FAILS=${FAILS:-0}
        LATENCY=${LATENCY:-0}
        RPS=${RPS:-0}

        # ======================================================
        # RESOURCE STATS
        # ======================================================

        CPU_AVG=$(awk -F',' '
        NR>1{sum+=$2;c++}
        END{
        if(c>0) print sum/c
        else print 0
        }' "$RESOURCE_FILE")

        CPU_PEAK=$(awk -F',' '
        NR>1{
        if($2>max) max=$2
        }
        END{
        print max+0
        }' "$RESOURCE_FILE")

        MEM_AVG=$(awk -F',' '
        NR>1{sum+=$3;c++}
        END{
        if(c>0) print sum/c
        else print 0
        }' "$RESOURCE_FILE")

        MEM_PEAK=$(awk -F',' '
        NR>1{
        if($3>max) max=$3
        }
        END{
        print max+0
        }' "$RESOURCE_FILE")

        ROW="$OQS_ALG,$KEY_SIZE,$NIST,$SEC_BITS,$USERS,$SIGN_AVG,$VERIFY_AVG,$TOKEN_SIZE,$SIGNATURE_SIZE,$LATENCY,$RPS,$FAILS,$CPU_AVG,$CPU_PEAK,$MEM_AVG,$MEM_PEAK"

        echo "$ROW" >> "$BENCH_FILE"
        echo "$ROW" >> "$OVERALL_FILE"

        echo "✅ USERS=$USERS completed"

    done

done

echo ""
echo "🚀 ALL BENCHMARKS COMPLETED"
echo "📊 Overall benchmark:"
echo "$OVERALL_FILE"
echo ""