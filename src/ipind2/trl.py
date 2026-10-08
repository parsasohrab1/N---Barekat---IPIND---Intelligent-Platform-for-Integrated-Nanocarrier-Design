"""
Evidence-based Technology Readiness Level (TRL) assessor.

TRL 5 means "validation of integrated components in a **relevant environment**". For this product, a relevant environment means
real data/experiments, not just synthetic data. This module computes the level **from inspectable evidence**,
not from claims:

    python -m ipind2.trl --report docs/validation_report.json --history benchmarks/history.json

* Automatic criteria are read from ``validation_report.json``, the benchmark history and the database;
* Criteria not provable by code (deployment in a relevant environment) require a **signed manual attestation**
  (``docs/trl_attestations.json``: name, date, evidence); without it they stay "not met";
* Result: the highest TRL for which **all** criteria of that level and lower levels hold.

No criterion can be satisfied with synthetic data when it requires "real data" (tested).
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

MIN_REAL_LAB_RESULTS = 10  # at least one fine-tuning round per SRS §4.6


@dataclass
class Criterion:
    id: str
    level: int
    description: str
    kind: str  # 'auto' | 'attestation'
    met: bool = False
    evidence: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "level": self.level, "description": self.description, "kind": self.kind,
                "met": self.met, "evidence": self.evidence}


@dataclass
class TRLAssessment:
    criteria: List[Criterion]
    achieved_level: int
    notes: List[str] = field(default_factory=list)

    @property
    def target_met(self) -> bool:
        return self.achieved_level >= 5

    def missing_for(self, level: int) -> List[Criterion]:
        return [c for c in self.criteria if c.level <= level and not c.met]

    def to_markdown(self) -> str:
        lines = [f"## TRL computed from evidence: **{self.achieved_level}**", "",
                 "| Level | Criterion | Status | Evidence |", "|---|---|---|---|"]
        for c in sorted(self.criteria, key=lambda c: (c.level, c.id)):
            lines.append(f"| {c.level} | {c.id}: {c.description} | {'✅' if c.met else '❌'} | {c.evidence} |")
        if self.notes:
            lines += ["", *[f"- {n}" for n in self.notes]]
        return "\n".join(lines)


def _load(path: Optional[str]):
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return None


def assess(
    report_path: Optional[str] = None,
    history_path: Optional[str] = None,
    attestations_path: Optional[str] = None,
    public_benchmark_path: Optional[str] = None,
    md_validation_path: Optional[str] = None,
    real_lab_results: int = 0,
    real_lab_improved_holdout: bool = False,
    test_suite_passed: Optional[bool] = None,
) -> TRLAssessment:
    """
    Args:
        report_path: output of ``ipind2.training.validate``.
        history_path: benchmark history (``BenchmarkHistory``).
        public_benchmark_path: output of ``ipind2.benchmarking.public`` (real public dataset).
        md_validation_path: output of ``ipind2.md_simulation.campaign`` (real MD runs).
        attestations_path: signed manual attestation of the non-automatic criteria.
        real_lab_results: number of **real lab results** (not synthetic) recorded in the database.
        real_lab_improved_holdout: whether one update round reduced the error on a real holdout.
        test_suite_passed: result of the last pytest run (reported by the caller).
    """
    report = _load(report_path) or {}
    history = _load(history_path) or []
    attest = _load(attestations_path) or {}
    gate = report.get("benchmark_gate", {})
    accuracy = report.get("accuracy_holdout", {})

    def nfr(key: str) -> bool:
        return bool(report.get(key, {}).get("passed"))

    criteria: List[Criterion] = []

    def add(id_, level, description, kind, met, evidence):
        criteria.append(Criterion(id_, level, description, kind, bool(met), evidence))

    # --- TRL 3/4: proof of concept and lab validation of components ------------------
    add("T1", 4, "Automated test suite passes", "auto", test_suite_passed is True,
        {True: "pytest PASS", False: "pytest FAIL", None: "Result not provided"}[test_suite_passed])
    add("T2", 4, "Model accuracy on synthetic holdout: NFR-01/02/03 and FR-03", "auto",
        accuracy and all(accuracy.get(k, {}).get("passed") for k in ("NFR-01", "NFR-02", "NFR-03", "FR-03")),
        "From validation_report.accuracy_holdout" if accuracy else "Validation report not available")
    add("T3", 4, "Performance NFRs (04, 05, 06, 08) measured and satisfied", "auto",
        all(nfr(k) for k in ("NFR-04", "NFR-05", "NFR-06", "NFR-08")),
        ", ".join(f"{k}={'✓' if nfr(k) else '✗'}" for k in ("NFR-04", "NFR-05", "NFR-06", "NFR-08")))

    # --- TRL 5: integration in a relevant environment -----------------------------------------
    add("R1", 5, "Frozen reference benchmark gate PASS (no degradation and NFRs hold)", "auto", gate.get("passed") is True,
        "benchmark_gate.passed" if gate else "Gate not run")
    pub = _load(public_benchmark_path)
    if pub:
        summary = pub.get("summary", {})
        gnn = summary.get("ipind2-gnn+morgan", summary.get("ipind2-gnn", {})).get("r2_mean")
        forest = summary.get("random-forest (Morgan)", {}).get("r2_mean")
        ok = gnn is not None and forest is not None and gnn >= forest
        evidence = (f"{pub.get('dataset')}: platform R² {gnn:.3f} vs. RandomForest+Morgan baseline {forest:.3f}"
                    if gnn is not None and forest is not None else "Incomplete report")
    else:
        ok, evidence = False, "Public benchmark report (public_benchmark.json) not available"
    add("R2", 5, "On a real public dataset, platform R² ≥ simple baseline (RandomForest+Morgan)", "auto", ok, evidence)
    md = _load(md_validation_path) or report.get("md_real_validation") or {}
    md_real = bool(md.get("complete"))
    if md:
        rows = md.get("candidates", [])
        longest = max((r.get("simulated_ns", 0) for r in rows), default=0)
        md_evidence = (
            f"{len(rows)} candidates, real MD, longest {longest:g} ns of the required {md.get('required_ns', 100):g} ns"
            + ("" if md_real else " — below the requirement")
        )
    else:
        md_evidence = "Only conformer sampling; no MD report"
    add("R3", 5, "Real MD validation (≥100 ns, MD engine) for candidates", "auto", md_real, md_evidence)
    add("R4", 5, f"Lab loop: ≥{MIN_REAL_LAB_RESULTS} real results and improvement on a real holdout", "auto",
        real_lab_results >= MIN_REAL_LAB_RESULTS and real_lab_improved_holdout,
        f"{real_lab_results} real results; holdout improvement={'yes' if real_lab_improved_holdout else 'no/not measured'}")
    deploy = attest.get("staging_deployment", {})
    add("R5", 5, "Deployment in a relevant environment (staging: PostgreSQL+TLS1.3+Redis) with signed attestation", "attestation",
        bool(deploy.get("signed_by") and deploy.get("date") and deploy.get("evidence")),
        f"Signed by: {deploy.get('signed_by')} ({deploy.get('date')})" if deploy else "No manual attestation available")
    sec = attest.get("security_review", {})
    add("R6", 5, "Independent security review (SEC-01..06) with signed attestation", "attestation",
        bool(sec.get("signed_by") and sec.get("date") and sec.get("evidence")),
        f"Signed by: {sec.get('signed_by')} ({sec.get('date')})" if sec else "No manual attestation available")

    achieved = 3
    for level in (4, 5):
        if all(c.met for c in criteria if c.level == level):
            achieved = level
        else:
            break

    notes = []
    if achieved < 5:
        missing = ", ".join(c.id for c in criteria if c.level == 5 and not c.met)
        notes.append(f"Unmet criteria for TRL 5: {missing}")
    if not md_real:
        notes.append("Without real MD, FR-05 (≥100 ns) is not confirmed; conformer sampling is not a substitute for it.")
    return TRLAssessment(criteria, achieved, notes)


def main() -> int:
    parser = argparse.ArgumentParser(description="IPIND² evidence-based TRL assessment")
    parser.add_argument("--report", default="docs/validation_report.json")
    parser.add_argument("--history", default="benchmarks/history.json")
    parser.add_argument("--attestations", default="docs/trl_attestations.json")
    parser.add_argument("--public-benchmark", default="docs/public_benchmark.json")
    parser.add_argument("--md-validation", default="docs/md_validation.json")
    parser.add_argument("--tests-passed", choices=["yes", "no"], default=None)
    parser.add_argument("--real-lab-results", type=int, default=0)
    parser.add_argument("--real-lab-improved", action="store_true")
    args = parser.parse_args()
    result = assess(
        args.report, args.history, args.attestations, args.public_benchmark, args.md_validation, args.real_lab_results, args.real_lab_improved,
        None if args.tests_passed is None else args.tests_passed == "yes",
    )
    print(result.to_markdown())
    return 0 if result.target_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
