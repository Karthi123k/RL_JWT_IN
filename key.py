import os
import json
import oqs
from oqs import Signature

BASE_DIR = "certs"


def classify_algorithm(name):
    """Classify algorithm type + metadata"""
    if "ML-DSA" in name or "Dilithium" in name:
        return {"type": "lattice", "standard": "FIPS 204"}
    elif "Falcon" in name:
        return {"type": "lattice", "standard": "FIPS 206 (draft)"}
    elif "SPHINCS+" in name:
        variant = "fast" if "f" in name else "small"
        return {"type": "hash-based", "variant": variant, "standard": "FIPS 205"}
    else:
        return {"type": "experimental", "standard": "non-standard"}


def clean_folder_name(name):
    """Convert algorithm name → folder name"""
    return (
        name.lower()
        .replace("+", "")
        .replace("-", "")
        .replace("simple", "")
        .replace("sha2", "")
        .replace("shake", "")
    )


def generate_keys(alg):
    try:
        folder = clean_folder_name(alg)
        print(f"[+] Generating {alg}")

        with Signature(alg) as signer:
            public_key = signer.generate_keypair()
            private_key = signer.export_secret_key()

        path = os.path.join(BASE_DIR, folder)
        os.makedirs(path, exist_ok=True)

        # Save keys
        with open(f"{path}/public_key.bin", "wb") as f:
            f.write(public_key)

        with open(f"{path}/private_key.bin", "wb") as f:
            f.write(private_key)

        # Metadata
        meta = classify_algorithm(alg)
        meta["algorithm"] = alg
        meta["folder"] = folder

        with open(f"{path}/meta.json", "w") as f:
            json.dump(meta, f, indent=4)

        print(f"[✓] Done: {alg}\n")

    except Exception as e:
        print(f"[!] Failed: {alg} → {e}\n")


def main():
    print("=== Generating ALL PQC Signature Keys ===\n")

    algorithms = oqs.get_enabled_sig_mechanisms()

    print(f"[INFO] Total algorithms found: {len(algorithms)}\n")

    for alg in algorithms:
        generate_keys(alg)

    print("=== All keys generated ===")


if __name__ == "__main__":
    main()