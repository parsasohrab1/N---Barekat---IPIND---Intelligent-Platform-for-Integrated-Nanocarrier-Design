"""
Automated Lab Adapters (Lab-in-the-loop Adapters)

First version per the SRS (FR-11): "generic adapter with CSV/REST format; direct connection to specific
equipment in later phases". This module implements exactly these two generic adapters and
defines a base interface (``LabAdapter``) so that equipment-specific adapters (liquid
handler, synthesis robotics, etc.) can be added in the future without changing the calling code.

See docs/SRS.md §4.9 (FR-11).
"""

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from .schema import ExperimentalResult


class LabAdapter(ABC):
    """Base interface of every lab adapter."""

    @abstractmethod
    def fetch_new_results(self) -> List[ExperimentalResult]:
        """Reads new lab results from the external source and returns them.

        Each call must return only records that are new since the previous call of this
        instance (idempotent per-instance)."""

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


class CSVLabAdapter(LabAdapter):
    """
    Adapter for reading lab results from a CSV file with columns matching
    ``ExperimentalResult`` (see sql/schema.sql -> experimental_results).

    An instance does not retain rows it has already returned between separate process
    runs; for cursor persistence across runs, store the number of the last processed row in the
    calling layer (e.g., the database) and pass it to the constructor as ``skip_rows``.
    """

    def __init__(self, path: str, skip_rows: int = 0):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Lab CSV file not found: {self.path}")
        self._rows_returned = skip_rows

    def fetch_new_results(self) -> List[ExperimentalResult]:
        with self.path.open(newline="", encoding="utf-8") as fh:
            reader = list(csv.DictReader(fh))

        new_rows = reader[self._rows_returned :]
        self._rows_returned = len(reader)
        return [ExperimentalResult.from_dict(row) for row in new_rows]


class RESTLabAdapter(LabAdapter):
    """
    Generic adapter for connecting to a REST endpoint that returns lab results as
    a JSON array (a list of objects with the same fields as ExperimentalResult).

    To connect to specific equipment (liquid handler, proprietary screening device), subclass this
    class and override ``_parse_response``.
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
        import requests  # local import: this adapter is the only consumer of requests

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
            raise ValueError("The REST response must be a JSON list of records")
        return [ExperimentalResult.from_dict(record) for record in payload]
