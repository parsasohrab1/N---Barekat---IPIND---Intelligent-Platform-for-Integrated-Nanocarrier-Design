"""
آداپتورهای اتصال به آزمایشگاه خودکار (Lab-in-the-loop Adapters)

نسخه اول طبق SRS (FR-11): «آداپتور generic با فرمت CSV/REST؛ اتصال مستقیم به تجهیزات
خاص در فازهای بعدی». این ماژول دقیقاً همین دو آداپتور generic را پیاده‌سازی می‌کند و
یک رابط پایه (``LabAdapter``) تعریف می‌کند تا آداپتورهای اختصاصی تجهیزات (liquid
handler، رباتیک سنتز و ...) در آینده بدون تغییر کد فراخوان اضافه شوند.

See docs/SRS.md §4.9 (FR-11).
"""

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from .schema import ExperimentalResult


class LabAdapter(ABC):
    """رابط پایه هر آداپتور آزمایشگاهی."""

    @abstractmethod
    def fetch_new_results(self) -> List[ExperimentalResult]:
        """نتایج آزمایشگاهی جدید را از منبع بیرونی می‌خواند و برمی‌گرداند.

        هر فراخوانی باید فقط رکوردهایی را برگرداند که از فراخوانی قبلی همین
        نمونه (instance) جدید هستند (idempotent per-instance)."""

    def close(self) -> None:  # pragma: no cover - پیش‌فرض no-op
        pass


class CSVLabAdapter(LabAdapter):
    """
    آداپتور خواندن نتایج آزمایشگاهی از یک فایل CSV با ستون‌های منطبق بر
    ``ExperimentalResult`` (نگاه کنید به sql/schema.sql -> experimental_results).

    یک نمونه، ردیف‌هایی را که قبلاً برگردانده حفظ نمی‌کند بین اجراهای مجزای پردازش
    (process)؛ برای پایداری cursor بین اجراها، شماره آخرین ردیف پردازش‌شده را در
    لایه فراخوان (مثلاً پایگاه داده) ذخیره و به‌عنوان ``skip_rows`` در سازنده بدهید.
    """

    def __init__(self, path: str, skip_rows: int = 0):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"فایل CSV آزمایشگاهی یافت نشد: {self.path}")
        self._rows_returned = skip_rows

    def fetch_new_results(self) -> List[ExperimentalResult]:
        with self.path.open(newline="", encoding="utf-8") as fh:
            reader = list(csv.DictReader(fh))

        new_rows = reader[self._rows_returned :]
        self._rows_returned = len(reader)
        return [ExperimentalResult.from_dict(row) for row in new_rows]


class RESTLabAdapter(LabAdapter):
    """
    آداپتور generic برای اتصال به یک endpoint REST که نتایج آزمایشگاهی را به شکل
    یک آرایه JSON (لیست از آبجکت‌هایی با همان فیلدهای ExperimentalResult) برمی‌گرداند.

    برای اتصال به تجهیزات خاص (liquid handler، دستگاه غربالگری اختصاصی)، این کلاس را
    زیرکلاسی کنید و ``_parse_response`` را بازنویسی (override) کنید.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        results_path: str = "/results",
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.results_path = results_path
        self.timeout = timeout

    def _headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def fetch_new_results(self) -> List[ExperimentalResult]:
        import requests  # local import: این آداپتور تنها مصرف‌کننده requests است

        response = requests.get(
            f"{self.base_url}{self.results_path}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return self._parse_response(payload)

    def _parse_response(self, payload) -> List[ExperimentalResult]:
        if not isinstance(payload, list):
            raise ValueError("پاسخ REST باید یک لیست JSON از رکوردها باشد")
        return [ExperimentalResult.from_dict(record) for record in payload]
