import subprocess
import sys

packages = {
    "torch": "torch",
    "sentence_transformers": "sentence-transformers",
    "numpy": "numpy",
    "pandas": "pandas",
    "faiss": "faiss-cpu",
    "sklearn": "scikit-learn",
    "tqdm": "tqdm",
}

print("=" * 50)
print("Checking required packages...")
print("=" * 50)

missing = []

for import_name, pip_name in packages.items():
    try:
        __import__(import_name)
        # special check for torch + CUDA
        if import_name == "torch":
            import torch
            cuda = torch.cuda.is_available()
            ver = torch.__version__
            print(f"  [OK] torch {ver} | CUDA: {cuda}")
        else:
            import importlib
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            print(f"  [OK] {import_name} {ver}")
    except ImportError:
        print(f"  [MISSING] {import_name}")
        missing.append(pip_name)

print("=" * 50)

if missing:
    print(f"Missing: {missing}")
    print("Installing...")
    for pkg in missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    print("Done. Re-run this script to verify.")
else:
    print("All packages are installed!")

# Extra: check BAAI model cache
from pathlib import Path
cache_dirs = [
    Path.home() / ".cache" / "huggingface" / "hub",
    Path.home() / ".cache" / "torch" / "sentence_transformers",
]
model_name = "BAAI/bge-large-en-v1.5"
found = False
for d in cache_dirs:
    if d.exists():
        for p in d.iterdir():
            if "bge-large" in p.name.lower():
                print(f"\n[OK] BAAI model cache found: {p}")
                found = True
if not found:
    print(f"\n[INFO] BAAI model not cached yet — will download on first run.")

