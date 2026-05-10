from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("config.json")


def check_package(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def main() -> None:
    print("Checking project setup...\n")

    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        "src/train_model.py",
        "src/predict.py",
        "src/config.json",
        "data/email.csv",
        "models/README.md",
    ]

    all_ok = True

    for relative_path in required_files:
        path = PROJECT_ROOT / relative_path
        if path.exists():
            print(f"OK: {relative_path}")
        else:
            print(f"MISSING: {relative_path}")
            all_ok = False

    print("\nChecking Python packages...")
    packages = ["tensorflow", "streamlit", "pandas", "sklearn", "numpy"]

    for package in packages:
        if check_package(package):
            print(f"OK: {package}")
        else:
            print(f"NOT INSTALLED: {package}")
            all_ok = False

    print("\nChecking config...")
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        cfg = json.load(file)

    for key in ["data_path", "model_path", "tokenizer_path", "max_len", "max_words"]:
        if key in cfg:
            print(f"OK: config contains {key}")
        else:
            print(f"MISSING CONFIG KEY: {key}")
            all_ok = False

    print("\nResult:")
    if all_ok:
        print("Everything looks ready.")
    else:
        print("Some checks failed. Install requirements or restore missing files.")


if __name__ == "__main__":
    main()