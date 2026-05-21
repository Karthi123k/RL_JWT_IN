#!/bin/bash

set -euo pipefail

# ==========================================================
# CONFIG
# ==========================================================

MODE="reuse"
DURATION="60s"
REPEATS=3
SPAWN_RATE=50

CONCURRENCY=(
50
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
[mldsa44]=128
[sphincs128s]=128

[sphincs192f]=192
[sphincs192s]=192

[sphincs256f]=256
[sphincs256s]=256
)

LOCUST_FILE="locust_reuse.py"

BASE_DIR="logs/$MODE"

mkdir -p "$BASE_DIR"

OVERALL="$BASE_DIR/overall_benchmark.csv"

# ==========================================================
# CSV HEADER
# ==========================================================

echo "algorithm,key_size,nist_level,security_bits,users,sign_mean_ms,sign_std_ms,verify_mean_ms,verify_std_ms,token_size,signature_size,latency_mean_ms,latency_std_ms,throughput_mean_rps,throughput_std_rps,failures,cpu_mean_m,cpu_std_m,memory_mean_mi,memory_std_mi" > "$OVERALL"

# ==========================================================
# FUNCTIONS
# ==========================================================

mean(){

awk '
{
for(i=1;i<=NF;i++){
sum+=$i
n++
}
}
END{
if(n>0)
printf "%.4f",sum/n
else
print 0
}'
}

std(){

awk '
{
for(i=1;i<=NF;i++){
a[n]=$i
sum+=$i
n++
}
}
END{

if(n==0){
print 0
exit
}

mean=sum/n

for(i=0;i<n;i++)
var+=(a[i]-mean)^2

printf "%.4f",sqrt(var/n)

}'
}

cleanup(){

if [[ -n "${MONITOR_PID:-}" ]]
then
kill "$MONITOR_PID" 2>/dev/null || true
fi

}

trap cleanup EXIT INT TERM

# ==========================================================
# START
# ==========================================================

echo ""
echo "======================================="
echo "PQC Benchmark Started"
echo "======================================="
echo ""

for KEY_SIZE in "${KEY_SIZES[@]}"
do

OQS_ALG=${ALG_MAP[$KEY_SIZE]}
NIST=${SECURITY_LEVEL[$KEY_SIZE]}
SEC_BITS=${SECURITY_BITS[$KEY_SIZE]}

KEY_DIR="$BASE_DIR/$KEY_SIZE"

mkdir -p "$KEY_DIR"

BENCH="$KEY_DIR/benchmark_summary.csv"

cp "$OVERALL" "$BENCH"

echo ""
echo "Algorithm : $OQS_ALG"

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

sleep 10

until curl -s -X POST http://localhost:8080/login >/dev/null
do
sleep 2
done

for USERS in "${CONCURRENCY[@]}"
do

echo ""
echo "Users=$USERS"

SIGNS=""
VERIFYS=""
LATS=""
RPSS=""
CPUS=""
MEMS=""
FAILS_ALL=""

TOKEN_SIZE=0
SIGNATURE_SIZE=0

for RUN in $(seq 1 $REPEATS)
do

echo "Run=$RUN"

RESULT="$KEY_DIR/u${USERS}_r${RUN}"

RESOURCE="$KEY_DIR/resource_u${USERS}_r${RUN}.csv"

echo "time,cpu,memory" > "$RESOURCE"

(

while true
do

DATA=$(kubectl top pod \
-n pqc-jwt \
--no-headers | \
grep -E "auth|user" || true)

CPU=$(echo "$DATA" | awk '{
gsub("m","",$2)
sum+=$2
}
END{
print sum+0
}')

MEM=$(echo "$DATA" | awk '{
gsub("Mi","",$3)
sum+=$3
}
END{
print sum+0
}')

echo "$(date '+%F %T'),$CPU,$MEM"

sleep 2

done

) >> "$RESOURCE" &

MONITOR_PID=$!

curl -s -X POST http://localhost:8001/reset >/dev/null || true
curl -s -X POST http://localhost:8002/reset >/dev/null || true

locust \
-f "$LOCUST_FILE" \
-H http://localhost:8080 \
--headless \
-u "$USERS" \
-r "$SPAWN_RATE" \
--run-time "$DURATION" \
--stop-timeout 5 \
--csv="$RESULT"

kill "$MONITOR_PID" 2>/dev/null || true

AUTH=$(curl -s http://localhost:8080/metrics/crypto)

VERIFY=$(curl -s http://localhost:8080/metrics/verify)

SIGN=$(echo "$AUTH" | jq -r '.sign_avg //0')
VERIFYAVG=$(echo "$VERIFY" | jq -r '.verify_avg //0')

TOKEN_SIZE=$(echo "$AUTH" | jq -r '.token_size //0')
SIGNATURE_SIZE=$(echo "$AUTH" | jq -r '.signature_size //0')

AGG=$(grep "Aggregated" "${RESULT}_stats.csv" | tail -1)

LAT=$(echo "$AGG" | cut -d',' -f6)
RPS=$(echo "$AGG" | cut -d',' -f9)
FAIL=$(echo "$AGG" | cut -d',' -f4)

CPUAVG=$(awk -F',' '
NR>1{
sum+=$2
c++
}
END{
if(c>0)
print sum/c
else
print 0
}' "$RESOURCE")

MEMAVG=$(awk -F',' '
NR>1{
sum+=$3
c++
}
END{
if(c>0)
print sum/c
else
print 0
}' "$RESOURCE")

SIGNS="$SIGNS $SIGN"
VERIFYS="$VERIFYS $VERIFYAVG"
LATS="$LATS $LAT"
RPSS="$RPSS $RPS"
CPUS="$CPUS $CPUAVG"
MEMS="$MEMS $MEMAVG"
FAILS_ALL="$FAILS_ALL $FAIL"

done

SIGN_MEAN=$(echo "$SIGNS" | mean)
SIGN_STD=$(echo "$SIGNS" | std)

VERIFY_MEAN=$(echo "$VERIFYS" | mean)
VERIFY_STD=$(echo "$VERIFYS" | std)

LAT_MEAN=$(echo "$LATS" | mean)
LAT_STD=$(echo "$LATS" | std)

RPS_MEAN=$(echo "$RPSS" | mean)
RPS_STD=$(echo "$RPSS" | std)

CPU_MEAN=$(echo "$CPUS" | mean)
CPU_STD=$(echo "$CPUS" | std)

MEM_MEAN=$(echo "$MEMS" | mean)
MEM_STD=$(echo "$MEMS" | std)

FAIL_TOTAL=$(echo "$FAILS_ALL" | mean)

ROW="$OQS_ALG,$KEY_SIZE,$NIST,$SEC_BITS,$USERS,$SIGN_MEAN,$SIGN_STD,$VERIFY_MEAN,$VERIFY_STD,$TOKEN_SIZE,$SIGNATURE_SIZE,$LAT_MEAN,$LAT_STD,$RPS_MEAN,$RPS_STD,$FAIL_TOTAL,$CPU_MEAN,$CPU_STD,$MEM_MEAN,$MEM_STD"

echo "$ROW" >> "$BENCH"

echo "$ROW" >> "$OVERALL"

echo "Completed USERS=$USERS"

done

done

echo ""
echo "======================================="
echo "Benchmark Completed"
echo "======================================="
echo "$OVERALL"
echo ""