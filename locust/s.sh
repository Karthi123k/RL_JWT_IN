KEYS=(512 1024 2048 3072 4096)

rm -f logs/security/security_results.csv

for K in "${KEYS[@]}"
do
    echo "Running for KEY_SIZE=$K"

    export KEY_SIZE=$K

    docker compose down
    docker compose up --build -d

    sleep 5

    python3 security.py
done