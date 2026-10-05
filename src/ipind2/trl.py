"""
ارزیاب شواهدمحور سطح آمادگی فناوری (TRL).

TRL ۵ یعنی «اعتبارسنجی مؤلفه‌های یکپارچه در **محیط مرتبط**». برای این محصول، محیط مرتبط یعنی
داده/آزمایش واقعی، نه فقط داده سنتتیک. این ماژول سطح را **از روی شواهد قابل‌بازرسی**
محاسبه می‌کند، نه از ادعا:

    python -m ipind2.trl --report docs/validation_report.json --history benchmarks/history.json

* معیارهای خودکار از ``validation_report.json``، تاریخچه بنچمارک و پایگاه داده خوانده می‌شوند؛
* معیارهایی که با کد قابل‌اثبات نیستند (استقرار در محیط مرتبط) **تأیید دستی امضاشده** می‌خواهند
  (``docs/trl_attestations.json``: نام، تاریخ، شاهد)؛ بدون آن «برآورده نشده» می‌ماند؛
* حاصل: بالاترین TRL که **همه** معیارهای آن سطح و سطوح پایین‌تر برقرارند.

هیچ معیاری با داده سنتتیک نمی‌تواند معیار «داده واقعی» را برآورده کند (تست‌شده).
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

MIN_REAL_LAB_RESULTS = 10  # حداقل یک دور fine-tuning طبق SRS §4.6


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
        lines = [f"## TRL محاسبه‌شده از شواهد: **{self.achieved_level}**", "",
                 "| سطح | معیار | وضعیت | شاهد |", "|---|---|---|---|"]
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
    real_lab_results: int = 0,
    real_lab_improved_holdout: bool = False,
    test_suite_passed: Optional[bool] = None,
) -> TRLAssessment:
    """
    Args:
        report_path: خروجی ``ipind2.training.validate``.
        history_path: تاریخچه بنچمارک (``BenchmarkHistory``).
        public_benchmark_path: خروجی ``ipind2.benchmarking.public`` (دیتاست عمومی واقعی).
        attestations_path: تأیید دستی امضاشده معیارهای غیرخودکار.
        real_lab_results: تعداد نتایج **آزمایشگاهی واقعی** (نه سنتتیک) ثبت‌شده در پایگاه داده.
        real_lab_improved_holdout: آیا یک دور به‌روزرسانی روی holdout واقعی خطا را کم کرد.
        test_suite_passed: نتیجه آخرین اجرای pytest (فراخوان گزارش می‌دهد).
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

    # --- TRL 3/4: اثبات مفهوم و اعتبارسنجی آزمایشگاهی مؤلفه‌ها ------------------
    add("T1", 4, "مجموعه تست خودکار گذراست", "auto", test_suite_passed is True,
        {True: "pytest PASS", False: "pytest FAIL", None: "نتیجه ارائه نشده"}[test_suite_passed])
    add("T2", 4, "دقت مدل روی holdout سنتتیک: NFR-01/02/03 و FR-03", "auto",
        accuracy and all(accuracy.get(k, {}).get("passed") for k in ("NFR-01", "NFR-02", "NFR-03", "FR-03")),
        "از validation_report.accuracy_holdout" if accuracy else "گزارش اعتبارسنجی موجود نیست")
    add("T3", 4, "NFR عملکردی (۰۴، ۰۵، ۰۶، ۰۸) اندازه‌گیری و برقرار", "auto",
        all(nfr(k) for k in ("NFR-04", "NFR-05", "NFR-06", "NFR-08")),
        ", ".join(f"{k}={'✓' if nfr(k) else '✗'}" for k in ("NFR-04", "NFR-05", "NFR-06", "NFR-08")))

    # --- TRL 5: یکپارچگی در محیط مرتبط -----------------------------------------
    add("R1", 5, "دروازه بنچمارک مرجع منجمد PASS (بدون افت و NFR برقرار)", "auto", gate.get("passed") is True,
        "benchmark_gate.passed" if gate else "دروازه اجرا نشده")
    pub = _load(public_benchmark_path)
    if pub:
        summary = pub.get("summary", {})
        gnn = summary.get("ipind2-gnn", {}).get("r2_mean")
        forest = summary.get("random-forest (Morgan)", {}).get("r2_mean")
        ok = gnn is not None and forest is not None and gnn >= forest
        evidence = (f"{pub.get('dataset')}: R² پلتفرم {gnn:.3f} در برابر خط پایه RandomForest+Morgan {forest:.3f}"
                    if gnn is not None and forest is not None else "گزارش ناقص")
    else:
        ok, evidence = False, "گزارش بنچمارک عمومی (public_benchmark.json) موجود نیست"
    add("R2", 5, "روی دیتاست عمومی واقعی، R² پلتفرم ≥ خط پایه ساده (RandomForest+Morgan)", "auto", ok, evidence)
    md_real = bool(report.get("md_real_validation", {}).get("complete"))
    add("R3", 5, "اعتبارسنجی MD واقعی (≥۱۰۰ ns، موتور MD) برای کاندیداها", "auto", md_real,
        "md_real_validation.complete" if md_real else "فقط نمونه‌برداری کانفورمری؛ موتور MD اجرا نشده")
    add("R4", 5, f"حلقه آزمایشگاهی: ≥{MIN_REAL_LAB_RESULTS} نتیجه واقعی و بهبود روی holdout واقعی", "auto",
        real_lab_results >= MIN_REAL_LAB_RESULTS and real_lab_improved_holdout,
        f"{real_lab_results} نتیجه واقعی؛ بهبود holdout={'بله' if real_lab_improved_holdout else 'خیر/سنجیده نشده'}")
    deploy = attest.get("staging_deployment", {})
    add("R5", 5, "استقرار در محیط مرتبط (staging: PostgreSQL+TLS1.3+Redis) با تأیید امضاشده", "attestation",
        bool(deploy.get("signed_by") and deploy.get("date") and deploy.get("evidence")),
        f"امضا: {deploy.get('signed_by')} ({deploy.get('date')})" if deploy else "تأیید دستی موجود نیست")
    sec = attest.get("security_review", {})
    add("R6", 5, "بازبینی امنیتی مستقل (SEC-01..06) با تأیید امضاشده", "attestation",
        bool(sec.get("signed_by") and sec.get("date") and sec.get("evidence")),
        f"امضا: {sec.get('signed_by')} ({sec.get('date')})" if sec else "تأیید دستی موجود نیست")

    achieved = 3
    for level in (4, 5):
        if all(c.met for c in criteria if c.level == level):
            achieved = level
        else:
            break

    notes = []
    if achieved < 5:
        missing = ", ".join(c.id for c in criteria if c.level == 5 and not c.met)
        notes.append(f"برای TRL ۵ معیارهای برآورده‌نشده: {missing}")
    if not md_real:
        notes.append("بدون MD واقعی، FR-05 (≥۱۰۰ ns) تأیید نشده؛ نمونه‌برداری کانفورمری جایگزین آن نیست.")
    return TRLAssessment(criteria, achieved, notes)


def main() -> int:
    parser = argparse.ArgumentParser(description="IPIND² evidence-based TRL assessment")
    parser.add_argument("--report", default="docs/validation_report.json")
    parser.add_argument("--history", default="benchmarks/history.json")
    parser.add_argument("--attestations", default="docs/trl_attestations.json")
    parser.add_argument("--public-benchmark", default="docs/public_benchmark.json")
    parser.add_argument("--tests-passed", choices=["yes", "no"], default=None)
    parser.add_argument("--real-lab-results", type=int, default=0)
    parser.add_argument("--real-lab-improved", action="store_true")
    args = parser.parse_args()
    result = assess(
        args.report, args.history, args.attestations, args.public_benchmark, args.real_lab_results, args.real_lab_improved,
        None if args.tests_passed is None else args.tests_passed == "yes",
    )
    print(result.to_markdown())
    return 0 if result.target_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
