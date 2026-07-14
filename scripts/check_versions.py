"""Version compliance check for all project dependencies."""
from packaging.version import Version

# (package, installed_ver, min_required)
checks = [
    ("python-dotenv",        "1.2.2",   "1.0.0"),
    ("pyyaml",               "6.0.3",   "6.0.1"),
    ("pydantic",             "2.13.4",  "2.7.0"),
    ("pydantic-settings",    "2.14.2",  "2.3.0"),
    ("qdrant-client",        "1.18.0",  "1.9.0"),
    ("torch",                "2.13.0",  "2.3.0"),
    ("torchvision",          "0.28.0",  "0.18.0"),
    ("transformers",         "5.13.1",  "4.43.0"),
    ("accelerate",           "1.14.0",  "0.31.0"),
    ("sentencepiece",        "0.2.2",   "0.2.0"),
    ("qwen-vl-utils",        "0.0.14",  "0.0.8"),
    ("sentence-transformers","5.6.0",   "3.0.0"),
    ("open-clip-torch",      "3.3.0",   "2.24.0"),
    ("Pillow",               "12.3.0",  "10.3.0"),
    ("tqdm",                 "4.68.4",  "4.66.0"),
    ("numpy",                "2.4.6",   "1.26.0"),
    ("rich",                 "15.0.0",  "13.7.0"),
    ("fastapi",              "0.139.0", "0.111.0"),
    ("uvicorn",              "0.51.0",  "0.30.0"),
    ("python-multipart",     "0.0.32",  "0.0.9"),
]

header = "{:<25} {:<14} {:<14} {}".format("Package", "Installed", "Required", "Status")
print(header)
print("-" * 65)

all_ok = True
for pkg, installed, required in checks:
    clean = installed.split("+")[0]  # Strip +cpu suffix
    ok = Version(clean) >= Version(required)
    status = "OK" if ok else "OUTDATED"
    if not ok:
        all_ok = False
    row = "{:<25} {:<14} {:<14} {}".format(pkg, installed, ">=" + required, status)
    print(row)

print()
if all_ok:
    print("All {} packages meet minimum version requirements.".format(len(checks)))
else:
    print("Some packages are outdated. Run: uv sync")
