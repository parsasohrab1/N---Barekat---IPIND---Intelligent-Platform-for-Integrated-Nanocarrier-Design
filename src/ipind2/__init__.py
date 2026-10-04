"""IPIND² - Intelligent Platform for Integrated Nanocarrier Design."""

import os as _os

# joblib/loky روی ویندوز بدون ``wmic`` هنگام شمارش هسته‌ها خطا چاپ می‌کند؛ مقدار را
# صریح تعیین می‌کنیم (باید پیش از import هر کتابخانه‌ای که joblib را می‌کشد باشد).
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_os.cpu_count() or 1))

__version__ = "0.5.0"
