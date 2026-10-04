"""
اعتبارسنجی انتشار: اندازه‌گیری واقعی NFRهای عملکردی و دروازه بنچمارک.

    python -m ipind2.training.validate --model-dir models/v1 --out docs/validation_report.json

هر عدد در گزارش از یک اجرای واقعی روی همین ماشین می‌آید (با مشخصات سخت‌افزار ثبت‌شده)؛
هیچ مقداری دستی وارد نمی‌شود. اندازه کتابخانه seed (NFR-10) فقط با ``--seed-library`` ساخته
می‌شود چون چند دقیقه طول می‌کشد.
"""

import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

from ..benchmarking import BenchmarkHistory, benchmark_release
from ..generation import generate_library, theoretical_library_size
from ..pipeline import DesignPipeline
from .bundle import ModelBundle


def hardware() -> Dict[str, Any]:
    import torch

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
    }


def measure(bundle: ModelBundle, n_library: int = 100_000, build_seed_library: bool = False) -> Dict[str, Any]:
    report: Dict[str, Any] = {"hardware": hardware(), "model_version": bundle.version}

    # NFR-04: زمان تولید ۱۰۰k ساختار (<۱۰ دقیقه)
    started = time.perf_counter()
    structures, stats = generate_library(n_library, seed=1)
    elapsed = time.perf_counter() - started
    report["NFR-04"] = {
        "n": len(structures), "seconds": round(elapsed, 1), "limit_seconds": 600,
        "validity_rate": stats.validity_rate, "passed": bool(elapsed < 600 and len(structures) >= n_library),
    }

    # NFR-05: زمان پیش‌بینی هر ساختار (<۱۰۰ ms)، هر دو مدل، ensemble کامل
    sample = [s.smiles for s in structures[:256]]
    bundle.physico.predict(sample[:8]); bundle.bio.predict(sample[:8])  # گرم‌کردن
    started = time.perf_counter()
    bundle.physico.predict(sample); bundle.bio.predict(sample)
    per_structure_ms = 1000 * (time.perf_counter() - started) / len(sample)
    report["NFR-05"] = {"ms_per_structure": round(per_structure_ms, 2), "limit_ms": 100, "ensemble_size": bundle.physico.n_ensemble,
                        "passed": bool(per_structure_ms < 100)}

    # NFR-06: بهینه‌سازی ۱۰۰۰ کاندیدا (<۱ ساعت)
    pipeline = DesignPipeline(bundle, seed=0)
    started = time.perf_counter()
    result = pipeline.design("نانوحامل لیپیدی برای تومور، اندازه بین ۸۰ تا ۱۲۰ نانومتر", n_generate=1000, n_pareto=15, n_final=5,
                             optimize_iterations=40, optimize_batch=25, run_md=False, explain=True)
    elapsed = time.perf_counter() - started
    report["NFR-06"] = {
        "candidates_evaluated": result.stats["optimization"]["evaluations"] + result.stats["generation"]["unique"],
        "seconds": round(elapsed, 1), "limit_seconds": 3600, "passed": bool(elapsed < 3600),
        "stage_seconds": result.stats["timings_seconds"], "final_candidates": len(result.final_candidates),
    }

    # NFR-08: مقیاس‌پذیری فضای ساختاری و NFR-10
    report["NFR-08"] = {"theoretical_space": theoretical_library_size(), "required": 1_000_000,
                        "passed": bool(theoretical_library_size() >= 1_000_000)}
    if build_seed_library:
        from ..generation.seed_library import build_seed_library as build

        with tempfile.TemporaryDirectory() as directory:
            seed = build(str(Path(directory) / "seed.parquet"), n=1_000_000, seed=0)
        report["NFR-10"] = {
            "n_total": seed.n_total, "n_combinatorial": seed.n_combinatorial, "n_public": seed.n_public,
            "seconds": round(seed.elapsed_seconds, 1), "meets_size": seed.meets_size_requirement,
            "meets_public_source": seed.meets_public_source_requirement,
            "note": "مبدأ ساختارها ترکیبیاتی است؛ شرط «پایگاه عمومی» فقط با فایل‌های PubChem/ZINC برآورده می‌شود.",
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="IPIND² release validation")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history", default=None)
    parser.add_argument("--seed-library", action="store_true")
    args = parser.parse_args()

    bundle = ModelBundle.load(args.model_dir)
    report = measure(bundle, build_seed_library=args.seed_library)
    history = BenchmarkHistory(args.history or str(Path(args.out).with_name("benchmark_history.json")))
    release = benchmark_release(bundle, history)
    report["accuracy_holdout"] = bundle.metrics.get("nfr", {})
    report["noise_ceiling_r2"] = bundle.metrics.get("noise_ceiling_r2", {})
    report["benchmark_gate"] = {"passed": release.passed, "nfr_violations": release.nfr_violations,
                                "markdown": release.to_markdown()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k.startswith("NFR")}, ensure_ascii=False, indent=2))
    print("benchmark gate:", "PASS" if release.passed else "FAIL")


if __name__ == "__main__":
    main()
