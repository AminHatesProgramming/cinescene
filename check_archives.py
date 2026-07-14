# check_archives.py
import zipfile
import os

# Check archive (1)
if os.path.exists('data/raw/archive (1).zip'):
    with zipfile.ZipFile('data/raw/archive (1).zip', 'r') as z:
        print("=== archive (1).zip ===")
        for name in z.namelist():
            print(f"  {name}")
else:
    print("archive (1).zip not found")

print()

# Check archive (2)
if os.path.exists('data/raw/archive (2).zip'):
    with zipfile.ZipFile('data/raw/archive (2).zip', 'r') as z:
        print("=== archive (2).zip ===")
        for name in z.namelist():
            print(f"  {name}")
else:
    print("archive (2).zip not found")

print()

# Check MovieSummaries.tar.gz
if os.path.exists('data/raw/MovieSummaries.tar.gz'):
    import tarfile
    with tarfile.open('data/raw/MovieSummaries.tar.gz', 'r:gz') as tar:
        print("=== MovieSummaries.tar.gz ===")
        for member in tar.getmembers()[:10]:  # First 10 files
            print(f"  {member.name}")
else:
    print("MovieSummaries.tar.gz not found")
