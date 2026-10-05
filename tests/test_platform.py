"""Tests of the data layer (FR-08), FAIR/MIRIBEL (FR-13), cache, benchmark gate (FR-12) and pipeline."""

import json
import threading

import numpy as np
import pandas as pd
import pytest

from ipind2.benchmarking import BenchmarkHistory, benchmark_release, dataset_fingerprint, frozen_reference_dataset
from ipind2.benchmarking.release import load_published_results
from ipind2.cache import MemoryCache, make_cache
from ipind2.database import (
    AuditRepository,
    ExperimentalResult,
    JobRepository,
    Molecule,
    MoleculeRepository,
    init_db,
    make_engine,
    make_session_factory,
    session_scope,
)
from ipind2.fair import (
    NanoparticleRecord,
    Observation,
    attach_measurements,
    export_records,
    load_jsonld,
    record_from_candidate,
    to_jsonld,
    validate_record,
)

QUAT = "CCCCCCCCCCCCCCCC[N+](C)(C)C"


@pytest.fixture()
def factory():
    engine = make_engine("sqlite://")
    init_db(engine)
    return make_session_factory(engine)


class TestDatabase:
    def test_molecule_dedup_by_canonical_smiles_and_descriptors(self, factory):
        with session_scope(factory) as s:
            repo = MoleculeRepository(s)
            first = repo.add_molecule("OCC", "lipid")
            again = repo.add_molecule("C(O)C", "lipid")  # the same molecule, a different notation
            assert first.id == again.id and repo.count() == 1
            assert first.molecular_weight == pytest.approx(46.07, abs=0.05) and first.inchikey

    def test_invalid_smiles_not_stored(self, factory):
        with session_scope(factory) as s:
            repo = MoleculeRepository(s)
            assert repo.add_molecule("bad(((") is None and repo.count() == 0
            assert repo.add_molecules(["OCC", "bad(((", "CCN"]) == [1, 2]

    def test_predictions_and_experiments_round_trip(self, factory):
        with session_scope(factory) as s:
            repo = MoleculeRepository(s)
            mid = repo.add_molecule(QUAT).id
            repo.store_predictions(mid, {"phys_size_nm": 98.0}, {"bio_cytotoxicity_ic50_ug_ml": 41.0}, 0.9, "v1", "HEK293")
            added = repo.add_experimental_results(pd.DataFrame([
                {"molecule_id": mid, "experimental_size_nm": 101.0, "experimental_date": "2026-05-01", "lab_technician": "A"},
                {"molecule_id": 424242, "experimental_size_nm": 1.0},  # nonexistent ⇒ rejected
            ]))
            assert added == 1
            pending = repo.unconsumed_results()
            assert len(pending) == 1 and pending.iloc[0]["experimental_size_nm"] == 101.0
            repo.mark_consumed(pending["_row_id"].tolist())
            assert repo.unconsumed_results().empty

    def test_foreign_keys_enforced(self, factory):
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            with session_scope(factory) as s:
                s.add(ExperimentalResult(molecule_id=999, experimental_size_nm=1.0))

    def test_transaction_rolls_back_on_error(self, factory):
        with pytest.raises(RuntimeError):
            with session_scope(factory) as s:
                MoleculeRepository(s).add_molecule("OCC")
                raise RuntimeError("boom")
        with session_scope(factory) as s:
            assert MoleculeRepository(s).count() == 0

    def test_lookup_feeds_active_learning(self, factory):
        with session_scope(factory) as s:
            repo = MoleculeRepository(s)
            ids = repo.add_molecules(["OCC", "CCN"])
            assert repo.smiles_by_id() == {ids[0]: "OCC", ids[1]: "CCN"}

    def test_audit_and_jobs(self, factory):
        with session_scope(factory) as s:
            AuditRepository(s).record("login", "alice", detail={"k": 1})
            AuditRepository(s).record("login", "bob")
            assert [r.username for r in AuditRepository(s).recent(10, "alice")] == ["alice"]
            JobRepository(s).create("j1", "design", "alice", {"n": 1})
            assert JobRepository(s).get("j1").status == "queued" and JobRepository(s).get("nope") is None

    def test_schema_sql_and_orm_agree_on_core_columns(self):
        """sql/schema.sql (the PostgreSQL reference) and the ORM must not drift apart."""
        import re
        from pathlib import Path

        ddl = (Path(__file__).parent.parent / "sql" / "schema.sql").read_text(encoding="utf-8")
        for table in ("molecules", "physicochemical_properties", "biological_properties", "experimental_results"):
            match = re.search(rf"CREATE TABLE {table} \((.*?)\n\);", ddl, re.S)
            assert match, f"{table} is not in schema.sql"
            sql_columns = {line.split()[0] for line in match.group(1).splitlines() if line.strip() and not line.strip().startswith("--")}
            from ipind2.database import Base

            orm_columns = {c.name for c in Base.metadata.tables[table].columns}
            assert sql_columns <= orm_columns, (table, sql_columns - orm_columns)


class TestFAIR:
    CANDIDATE = {
        "smiles": QUAT,
        "predictions": {"phys_size_nm": 99.0, "phys_zeta_potential_mV": 25.0, "bio_cytotoxicity_ic50_ug_ml": 40.0, "phys_pdi": 0.15},
        "confidence": {"phys_size_nm": 0.9},
    }

    def test_predicted_values_require_model_version(self):
        with pytest.raises(ValueError):
            Observation(1.0, "predicted")
        with pytest.raises(ValueError):
            Observation(1.0, "guessed")
        assert Observation(1.0, "measured").provenance == "measured"

    def test_candidate_record_marks_everything_predicted(self):
        record = record_from_candidate(self.CANDIDATE, "v7", "lipid")
        observations = [*record.material.values(), *record.biological.values()]
        assert observations and all(o.provenance == "predicted" and o.model_version == "v7" for o in observations)
        assert record.identifier.startswith("urn:uuid:")

    def test_measurements_attach_with_measured_provenance(self):
        record = attach_measurements(
            record_from_candidate(self.CANDIDATE, "v7"),
            {"experimental_size_nm": 104.0, "lab_technician": "Dr. A", "experimental_date": "2026-05-01"},
        )
        assert record.material["hydrodynamic_size_nm"].provenance == "measured"
        assert record.material["hydrodynamic_size_nm"].value == 104.0
        assert record.protocol["operator"].value == "Dr. A"

    def test_completeness_scoring(self):
        record = record_from_candidate(self.CANDIDATE, "v7", "lipid")
        report = validate_record(record)
        assert 0 < report.score < 1 and not report.complete
        assert "characterization_method" in report.missing["protocol"] and "cell_line" in report.missing["biological"]
        assert "hydrodynamic_size_nm" in report.predicted_only
        assert validate_record(NanoparticleRecord("urn:uuid:x")).score == 0.0

    def test_jsonld_structure_and_units(self):
        node = to_jsonld(record_from_candidate(self.CANDIDATE, "v7", "lipid"))
        assert node["@id"].startswith("urn:uuid:") and "ipind:NanoparticleRecord" in node["@type"] and node["license"]
        size = next(o for o in node["ipind:material"] if o["name"] == "hydrodynamic_size_nm")
        assert size["unit"].endswith("NanoM") and size["prov:wasGeneratedBy"]["ipind:provenance"] == "predicted"

    def test_jsonld_round_trip_preserves_provenance(self, tmp_path):
        records = [
            attach_measurements(record_from_candidate(self.CANDIDATE, "v7", "lipid"), {"experimental_size_nm": 104.0}),
            record_from_candidate(self.CANDIDATE, "v7"),
        ]
        path = export_records(records, str(tmp_path / "out.jsonld"), "jsonld")
        loaded = load_jsonld(str(path))
        assert [r.identifier for r in loaded] == [r.identifier for r in records]
        assert loaded[0].material["hydrodynamic_size_nm"].provenance == "measured"
        assert loaded[1].material["hydrodynamic_size_nm"].model_version == "v7"
        assert json.loads(path.read_text(encoding="utf-8"))["@context"]

    def test_csv_export_is_flat_with_provenance(self, tmp_path):
        path = export_records([record_from_candidate(self.CANDIDATE, "v7")], str(tmp_path / "o.csv"), "csv")
        frame = pd.read_csv(path)
        assert {"identifier", "field", "value", "unit", "provenance", "model_version", "license"} <= set(frame.columns)
        assert set(frame["provenance"]) == {"predicted"}

    def test_unknown_format_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            export_records([], str(tmp_path / "x"), "xml")


class TestCache:
    def test_ttl_expiry(self, monkeypatch):
        import time

        cache = MemoryCache(default_ttl=10)
        clock = [1000.0]
        monkeypatch.setattr(time, "monotonic", lambda: clock[0])
        cache.set("k", {"a": 1}, ttl_seconds=5)
        assert cache.get("k") == {"a": 1}
        clock[0] += 6
        assert cache.get("k") is None

    def test_lru_eviction_and_delete(self):
        cache = MemoryCache(max_items=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")
        cache.set("c", 3)  # b is the least recently used
        assert cache.get("b") is None and cache.get("a") == 1 and cache.get("c") == 3
        cache.delete("a")
        assert cache.get("a") is None

    def test_thread_safety(self):
        cache = MemoryCache(max_items=50)
        errors = []

        def worker(offset):
            try:
                for i in range(300):
                    cache.set(f"k{(i + offset) % 70}", i)
                    cache.get(f"k{i % 70}")
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert not errors

    def test_make_cache_falls_back_when_redis_unreachable(self, monkeypatch):
        monkeypatch.setenv("IPIND_REDIS_URL", "redis://127.0.0.1:1/0")
        assert isinstance(make_cache(), MemoryCache)


class TestReleaseBenchmark:
    def test_reference_set_is_deterministic_and_fingerprinted(self):
        a, b = frozen_reference_dataset(80), frozen_reference_dataset(80)
        assert a.equals(b) and dataset_fingerprint(a) == dataset_fingerprint(b)
        assert dataset_fingerprint(a) != dataset_fingerprint(frozen_reference_dataset(80, seed=1))

    def test_gate_records_history_and_detects_regression(self, smoke_bundle, tmp_path):
        reference = frozen_reference_dataset(120)
        history = BenchmarkHistory(str(tmp_path / "h.json"))
        first = benchmark_release(smoke_bundle, history, reference)
        assert len(first.results) == 14 and all(not r.regressed for r in first.regressions)
        assert len(history.load()) == 14

        same = benchmark_release(smoke_bundle, history, reference, record=False)
        assert all(not r.regressed for r in same.regressions)  # the same model ⇒ no degradation

        import copy

        degraded = copy.deepcopy(smoke_bundle)
        degraded.version = "degraded"
        for model in degraded.physico.models:
            with __import__("torch").no_grad():
                for parameter in model.parameters():
                    parameter.add_(0.5 * __import__("torch").randn_like(parameter))
        worse = benchmark_release(degraded, history, reference, record=False)
        assert any(r.regressed for r in worse.regressions) and not worse.passed

    def test_smoke_models_violate_nfr_gate(self, smoke_bundle, tmp_path):
        report = benchmark_release(smoke_bundle, BenchmarkHistory(str(tmp_path / "h.json")), frozen_reference_dataset(100), record=False)
        assert report.nfr_violations and "NFR" not in report.to_markdown().splitlines()[0]
        assert not report.passed

    def test_changed_reference_starts_new_history(self, smoke_bundle, tmp_path):
        history = BenchmarkHistory(str(tmp_path / "h.json"))
        benchmark_release(smoke_bundle, history, frozen_reference_dataset(60))
        other = benchmark_release(smoke_bundle, history, frozen_reference_dataset(60, seed=3), record=False)
        assert all("No previous run" in r.message for r in other.regressions)

    def test_published_results_require_citation(self, tmp_path):
        good = tmp_path / "p.json"
        good.write_text(json.dumps({"phys_size_nm": {"metric": "rmse", "value": 4.2, "citation": "doi:10.x/y"}}), encoding="utf-8")
        assert load_published_results(str(good))["phys_size_nm"]["value"] == 4.2
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"phys_size_nm": {"value": 4.2}}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_published_results(str(bad))
        assert load_published_results(None) == {} and load_published_results(str(tmp_path / "missing.json")) == {}
