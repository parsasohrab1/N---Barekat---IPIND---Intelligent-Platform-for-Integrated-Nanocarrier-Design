"""
بنچمارک هر انتشار مدل + دروازه CI (FR-12).

* **مجموعه مرجع منجمد داخلی** — داده سنتتیک قطعی (بذر ثابت) که بدون هیچ دانلودی در CI قابل
  اجراست. اثرانگشت محتوای آن در نام دیتاست ثبت می‌شود؛ اگر قوانین تولید داده عوض شود، تاریخچه
  به‌طور خودکار از نو شروع می‌شود (مقایسه با مرجع تغییرکرده بی‌معناست).
* **دیتاست‌های عمومی** (LNP-622، LANCE) — از طریق ``load_reference_dataset`` با فایل محلی.
* **مقالات منتشرشده** — ``published_results.json`` را خودتان با ارجاع دقیق پر کنید؛ این
  ماژول هیچ عدد «منتشرشده‌ای» از خودش اختراع نمی‌کند.

دروازه ``gate`` در صورت افت دقت نسبت به نسخه قبل **یا** نقض آستانه‌های NFR، کد خروج غیرصفر
می‌دهد (برای pipeline انتشار مدل)::

    python -m ipind2.benchmarking.release --model-dir models/v1 --history benchmarks/history.json
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..biological import BIO_TARGET_COLUMNS
from ..physicochemical import PHYSICO_TARGET_COLUMNS
from .runner import BenchmarkHistory, BenchmarkResult, RegressionReport, check_regression, run_benchmark

FROZEN_SEED = 20240601
FROZEN_SIZE = 500

# آستانه‌های NFR (SRS §2.2) برای دروازه انتشار
NFR_GATES = {
    "phys_size_nm": ("rmse", 5.0),
    "phys_zeta_potential_mV": ("rmse", 2.0),
    "phys_drug_loading_efficiency_percent": ("r2", 0.85),
}


def frozen_reference_dataset(n: int = FROZEN_SIZE, seed: int = FROZEN_SEED) -> pd.DataFrame:
    """مجموعه مرجع قطعی (با بذر جدا از هر بذر آموزش)."""
    from ..data_generation.synthetic_data_generator import SyntheticDataGenerator

    return SyntheticDataGenerator(seed).generate_dataset(n, include_pareto_labels=False)


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """اثرانگشت محتوای مرجع (SMILES + مقادیر هدف) — نام دیتاست را نسخه‌دار می‌کند."""
    columns = ["smiles", *PHYSICO_TARGET_COLUMNS, *BIO_TARGET_COLUMNS]
    payload = frame[columns].round(6).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


@dataclass
class ReleaseReport:
    results: List[BenchmarkResult]
    regressions: List[RegressionReport]
    nfr_violations: List[str]
    published: Dict[str, dict]

    @property
    def passed(self) -> bool:
        return not self.nfr_violations and not any(r.regressed for r in self.regressions)

    def to_markdown(self) -> str:
        lines = ["| دیتاست/هدف | RMSE | R² | تغییر نسبت به نسخه قبل |", "|---|---|---|---|"]
        for result, regression in zip(self.results, self.regressions):
            lines.append(f"| {result.dataset} | {result.rmse:.4f} | {result.r2:.4f} | {regression.message} |")
        if self.nfr_violations:
            lines += ["", "**نقض NFR:**"] + [f"- {v}" for v in self.nfr_violations]
        return "\n".join(lines)


def load_published_results(path: Optional[str]) -> Dict[str, dict]:
    """بارگذاری ``published_results.json`` (کلید = هدف، مقدار = {metric, value, citation})."""
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, entry in data.items():
        if "citation" not in entry or "value" not in entry:
            raise ValueError(f"ورودی منتشرشده «{key}» باید value و citation داشته باشد (بدون ارجاع پذیرفته نمی‌شود)")
    return data


def benchmark_release(
    bundle,
    history: BenchmarkHistory,
    reference: Optional[pd.DataFrame] = None,
    rel_rmse_tolerance: float = 0.05,
    r2_tolerance: float = 0.02,
    published_path: Optional[str] = None,
    record: bool = True,
) -> ReleaseReport:
    """
    سنجش یک نسخه مدل روی مجموعه مرجع، مقایسه با نسخه قبل و بررسی آستانه‌های NFR.

    تحمل افت: RMSE تا ``rel_rmse_tolerance`` (نسبی) بالا و R² تا ``r2_tolerance`` (مطلق) پایین
    مجاز است — نویز آموزش تصادفی مدل‌ها را نباید «regression» شمرد.
    """
    reference = frozen_reference_dataset() if reference is None else reference
    tag = dataset_fingerprint(reference)
    predictors = {**{c: bundle.physico for c in PHYSICO_TARGET_COLUMNS}, **{c: bundle.bio for c in BIO_TARGET_COLUMNS}}

    results: List[BenchmarkResult] = []
    regressions: List[RegressionReport] = []
    violations: List[str] = []
    for target, predictor in predictors.items():
        column = predictor.target_names.index(target)

        def predict(frame: pd.DataFrame, _p=predictor, _c=column) -> np.ndarray:
            mean = _p.predict(frame["smiles"].tolist()).reindex(range(len(frame)))
            return mean.iloc[:, _c].to_numpy()

        result = run_benchmark(
            predict, reference, target, feature_columns=["smiles"],
            model_version=bundle.version, dataset_name=f"{target}@{tag}",
        )
        previous = history.latest_for(result.dataset)
        tolerance = rel_rmse_tolerance * previous.rmse if previous else 0.0
        regressions.append(check_regression(history, result, rmse_tolerance=tolerance, r2_tolerance=r2_tolerance))
        results.append(result)

        if target in NFR_GATES:
            metric, threshold = NFR_GATES[target]
            value = result.rmse if metric == "rmse" else result.r2
            ok = value < threshold if metric == "rmse" else value > threshold
            if not ok:
                violations.append(f"{target}: {metric}={value:.4f} (آستانه {'<' if metric == 'rmse' else '>'} {threshold})")

    if record:
        for result in results:
            history.append(result)
    return ReleaseReport(results, regressions, violations, load_published_results(published_path))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="IPIND² release benchmark gate (FR-12)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--published", default=None)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args(argv)

    from ..training import ModelBundle

    report = benchmark_release(
        ModelBundle.load(args.model_dir), BenchmarkHistory(args.history),
        published_path=args.published, record=not args.no_record,
    )
    print(report.to_markdown())
    print("\nنتیجه دروازه:", "PASS" if report.passed else "FAIL")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
