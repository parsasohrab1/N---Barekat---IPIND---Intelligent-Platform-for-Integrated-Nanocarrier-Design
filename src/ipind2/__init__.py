"""IPIND² - Intelligent Platform for Integrated Nanocarrier Design."""

import os as _os

# joblib/loky on Windows without ``wmic`` prints an error when counting cores; we set the value
# explicitly (it must come before importing any library that pulls in joblib).
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_os.cpu_count() or 1))

__version__ = "0.5.0"
