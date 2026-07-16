"""Version compliance check for all project dependencies.

Actually inspects the installed environment via importlib.metadata instead of
a hardcoded table, so the result reflects what's really installed.
"""
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version

# (distribution name on PyPI, minimum required version)
REQUIREMENTS = [
    ("python-dotenv", "1.0.0"),
    ("pyyaml", "6.0.1"),
    ("pydantic", "2.7.0"),
    ("pydantic-settings", "2.3.0"),
    ("qdrant-client", "1.9.0"),
    ("torch", "2.3.0"),
    ("torchvision", "0.18.0"),
    ("transformers", "4.43.0"),
    ("accelerate", "0.31.0"),
    ("sentencepiece", "0.2.0"),
    ("qwen-vl-utils", "0.0.8"),
    ("sentence-transformers", "3.0.0"),
    ("open-clip-torch", "2.24.0"),
    ("Pillow", "10.3.0"),
    ("tqdm", "4.66.0"),
    ("numpy", "1.26.0"),
    ("rich", "13.7.0"),
    ("fastapi", "0.111.0"),
    ("uvicorn", "0.30.0"),
    ("python-multipart", "0.0.9"),
]

header = "{:<25} {:<14} {:<14} {}".format("Package", "Installed", "Required", "Status")
print(header)
print("-" * 65)

all_ok = True
for pkg, required in REQUIREMENTS:
    try:
        installed = version(pkg)
    except PackageNotFoundError:
        print("{:<25} {:<14} {:<14} {}".format(pkg, "NOT FOUND", ">=" + required, "MISSING"))
        all_ok = False
        continue

    clean = installed.split("+")[0]  # Strip +cpu / +cu121 local version suffix
    ok = Version(clean) >= Version(required)
    status = "OK" if ok else "OUTDATED"
    if not ok:
        all_ok = False
    print("{:<25} {:<14} {:<14} {}".format(pkg, installed, ">=" + required, status))

print()
if all_ok:
    print("All {} packages meet minimum version requirements.".format(len(REQUIREMENTS)))
else:
    print("Some packages are missing or outdated. Run: uv sync")
