"""ارزیاب TRL نباید با داده سنتتیک یا ادعا به سطح ۵ برسد."""

import json

from ipind2.trl import assess

GOOD_REPORT = {
    "NFR-04": {"passed": True}, "NFR-05": {"passed": True}, "NFR-06": {"passed": True}, "NFR-08": {"passed": True},
    "accuracy_holdout": {k: {"passed": True} for k in ("NFR-01", "NFR-02", "NFR-03", "FR-03")},
    "benchmark_gate": {"passed": True},
}
SYNTHETIC_HISTORY = [{"dataset": "phys_size_nm@15dd7e38"}]


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_no_evidence_means_low_level(tmp_path):
    result = assess(str(tmp_path / "none.json"), str(tmp_path / "none2.json"))
    assert result.achieved_level == 3 and not result.target_met


def test_synthetic_only_evidence_caps_at_trl4(tmp_path):
    result = assess(_write(tmp_path, "r.json", GOOD_REPORT), _write(tmp_path, "h.json", SYNTHETIC_HISTORY), test_suite_passed=True)
    assert result.achieved_level == 4 and not result.target_met
    missing = {c.id for c in result.missing_for(5)}
    assert {"R2", "R3", "R4", "R5", "R6"} <= missing and "R1" not in missing


def test_failing_tests_block_level_4(tmp_path):
    result = assess(_write(tmp_path, "r.json", GOOD_REPORT), None, test_suite_passed=False)
    assert result.achieved_level == 3


def test_failed_nfr_blocks_level_4(tmp_path):
    bad = json.loads(json.dumps(GOOD_REPORT))
    bad["NFR-04"]["passed"] = False
    assert assess(_write(tmp_path, "r.json", bad), None, test_suite_passed=True).achieved_level == 3


def test_attestation_requires_signature_date_and_evidence(tmp_path):
    unsigned = {"staging_deployment": {"evidence": "url"}, "security_review": {"signed_by": "X"}}
    result = assess(_write(tmp_path, "r.json", GOOD_REPORT), None,
                    attestations_path=_write(tmp_path, "a.json", unsigned), test_suite_passed=True)
    assert {"R5", "R6"} <= {c.id for c in result.missing_for(5)}


def _bench(tmp_path, gnn, forest):
    return _write(tmp_path, "pb.json", {"dataset": "lantern-hela", "summary": {
        "ipind2-gnn": {"r2_mean": gnn}, "random-forest (Morgan)": {"r2_mean": forest}}})


def test_weak_real_data_performance_does_not_satisfy_r2(tmp_path):
    """GNN پلتفرم روی داده واقعی از خط پایه ساده ضعیف‌تر است ⇒ R2 برقرار نیست."""
    result = assess(_write(tmp_path, "r.json", GOOD_REPORT), None, public_benchmark_path=_bench(tmp_path, 0.30, 0.48), test_suite_passed=True)
    r2 = next(c for c in result.criteria if c.id == "R2")
    assert not r2.met and "0.300" in r2.evidence and "0.480" in r2.evidence


def test_strong_real_data_performance_satisfies_r2(tmp_path):
    result = assess(_write(tmp_path, "r.json", GOOD_REPORT), None, public_benchmark_path=_bench(tmp_path, 0.55, 0.48), test_suite_passed=True)
    assert next(c for c in result.criteria if c.id == "R2").met


def test_all_real_evidence_reaches_trl5(tmp_path):
    report = dict(GOOD_REPORT, md_real_validation={"complete": True})
    history = SYNTHETIC_HISTORY + [{"dataset": "lnp-622"}]
    signed = {"signed_by": "Dr. A", "date": "2026-10-01", "evidence": "report-123"}
    result = assess(_write(tmp_path, "r.json", report), _write(tmp_path, "h.json", history),
                    _write(tmp_path, "a.json", {"staging_deployment": signed, "security_review": signed}),
                    public_benchmark_path=_bench(tmp_path, 0.6, 0.48),
                    real_lab_results=12, real_lab_improved_holdout=True, test_suite_passed=True)
    assert result.achieved_level == 5 and result.target_met and not result.missing_for(5)


def test_too_few_real_lab_results_do_not_count(tmp_path):
    result = assess(_write(tmp_path, "r.json", GOOD_REPORT), None, real_lab_results=3, real_lab_improved_holdout=True, test_suite_passed=True)
    assert "R4" in {c.id for c in result.missing_for(5)}


def test_markdown_lists_every_criterion(tmp_path):
    text = assess(_write(tmp_path, "r.json", GOOD_REPORT), None, test_suite_passed=True).to_markdown()
    assert all(f"| {i}:" in text or f"{i}:" in text for i in ("T1", "T2", "T3", "R1", "R2", "R3", "R4", "R5", "R6"))
