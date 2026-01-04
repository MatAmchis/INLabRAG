from __future__ import annotations
import importlib.metadata
import sys
from typing import Dict, Tuple
import torch


def print_env_versions(
    packages: Tuple[str, ...] = ("torch", "sentence-transformers", "transformers", "faiss-cpu")
) -> None:
    print("Python:", sys.version.split()[0])
    for pkg in packages:
        try:
            print(f"{pkg}: {importlib.metadata.version(pkg)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{pkg}: not found")


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        try:
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        except Exception:
            pass

    return device


def env_summary_dict(
    packages: Tuple[str, ...] = ("torch", "sentence-transformers", "transformers", "faiss-cpu")
) -> Dict[str, str]:
    info: Dict[str, str] = {"python": sys.version.split()[0]}

    for pkg in packages:
        try:
            info[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            info[pkg] = "not found"

    info["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return info


if __name__ == "__main__":
    print_env_versions()
    get_device()
