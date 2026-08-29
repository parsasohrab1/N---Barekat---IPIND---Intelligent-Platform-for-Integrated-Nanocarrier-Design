import csv

import pandas as pd
import pytest

from ipind2.lab_automation import (
    CSVLabAdapter,
    ExperimentalResult,
    RESTLabAdapter,
    ingest_results,
    should_trigger_retrain,
)

CSV_FIELDS = [
    "molecule_id",
    "experimental_size_nm",
    "experimental_zeta_potential",
    "experimental_loading_efficiency",
    "experimental_cytotoxicity",
    "experimental_date",
    "lab_technician",
]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class TestExperimentalResult:
    def test_from_dict_parses_all_fields(self):
        result = ExperimentalResult.from_dict(
            {
                "molecule_id": "42",
                "experimental_size_nm": "105.3",
                "experimental_zeta_potential": "-12.5",
                "experimental_loading_efficiency": "88",
                "experimental_cytotoxicity": "15.2",
                "experimental_date": "2026-08-20",
                "lab_technician": "Sara",
            }
        )
        assert result.molecule_id == 42
        assert result.experimental_size_nm == pytest.approx(105.3)
        assert result.experimental_date.isoformat() == "2026-08-20"
        assert result.lab_technician == "Sara"

    def test_from_dict_missing_molecule_id_raises(self):
        with pytest.raises(ValueError):
            ExperimentalResult.from_dict({"experimental_size_nm": "10"})

    def test_from_dict_tolerates_missing_optional_fields(self):
        result = ExperimentalResult.from_dict({"molecule_id": "1"})
        assert result.experimental_size_nm is None
        assert result.experimental_date is None

    def test_to_dict_roundtrip(self):
        result = ExperimentalResult.from_dict({"molecule_id": "1", "experimental_size_nm": "10.0"})
        d = result.to_dict()
        assert d["molecule_id"] == 1
        assert d["experimental_size_nm"] == pytest.approx(10.0)


class TestCSVLabAdapter:
    def test_fetches_all_rows_on_first_call(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        _write_csv(
            csv_path,
            [
                {"molecule_id": "1", "experimental_size_nm": "100"},
                {"molecule_id": "2", "experimental_size_nm": "110"},
            ],
        )
        adapter = CSVLabAdapter(str(csv_path))
        results = adapter.fetch_new_results()
        assert len(results) == 2
        assert [r.molecule_id for r in results] == [1, 2]

    def test_second_fetch_only_returns_new_rows(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        _write_csv(csv_path, [{"molecule_id": "1", "experimental_size_nm": "100"}])
        adapter = CSVLabAdapter(str(csv_path))
        first = adapter.fetch_new_results()
        assert len(first) == 1

        # شبیه‌سازی افزوده‌شدن یک نتیجه جدید به فایل CSV آزمایشگاهی
        _write_csv(
            csv_path,
            [
                {"molecule_id": "1", "experimental_size_nm": "100"},
                {"molecule_id": "2", "experimental_size_nm": "120"},
            ],
        )
        second = adapter.fetch_new_results()
        assert len(second) == 1
        assert second[0].molecule_id == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CSVLabAdapter(str(tmp_path / "does_not_exist.csv"))


class _FakeResponse:
    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._payload


class TestRESTLabAdapter:
    def test_fetch_new_results_parses_json_list(self, monkeypatch):
        payload = [
            {"molecule_id": 7, "experimental_size_nm": 95.0},
            {"molecule_id": 8, "experimental_loading_efficiency": 80.0},
        ]

        def fake_get(url, headers=None, timeout=None):
            assert url.endswith("/results")
            return _FakeResponse(payload)

        monkeypatch.setattr("requests.get", fake_get)

        adapter = RESTLabAdapter(base_url="https://lab.example.com", api_key="secret")
        results = adapter.fetch_new_results()
        assert len(results) == 2
        assert results[0].molecule_id == 7

    def test_non_list_payload_raises(self, monkeypatch):
        monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse({"not": "a list"}))
        adapter = RESTLabAdapter(base_url="https://lab.example.com")
        with pytest.raises(ValueError):
            adapter.fetch_new_results()


class TestIngestion:
    def test_ingest_results_builds_dataframe(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        _write_csv(
            csv_path,
            [{"molecule_id": "1", "experimental_size_nm": "100", "lab_technician": "Ali"}],
        )
        adapter = CSVLabAdapter(str(csv_path))
        df = ingest_results(adapter)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["molecule_id"] == 1

    def test_ingest_results_empty_returns_empty_dataframe_with_columns(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        _write_csv(csv_path, [])
        adapter = CSVLabAdapter(str(csv_path))
        df = ingest_results(adapter)
        assert df.empty
        assert "molecule_id" in df.columns


class TestShouldTriggerRetrain:
    def test_below_threshold(self):
        assert not should_trigger_retrain(5, min_batch=10)

    def test_at_threshold(self):
        assert should_trigger_retrain(10, min_batch=10)

    def test_invalid_min_batch(self):
        with pytest.raises(ValueError):
            should_trigger_retrain(10, min_batch=0)
