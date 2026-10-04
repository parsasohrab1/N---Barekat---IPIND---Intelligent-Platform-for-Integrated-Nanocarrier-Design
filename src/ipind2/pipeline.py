"""
خط لوله انتها‌به‌انتهای طراحی (جریان داده SRS §3.2).

۱. ورودی: پرس‌وجوی زبان طبیعی یا پارامترهای ساختاریافته  → ``TargetParameters``
۲. تولید: مولد شرطی CVAE+CGAN چندین ساختار معتبر می‌سازد
۳. پیش‌بینی: GNN فیزیکوشیمیایی + Transformer زیستی (با عدم‌قطعیت ensemble)
۴. بهینه‌سازی: Pareto-Guided RL + جبهه پارتوی ساختارهای تولیدشده
۵. اعتبارسنجی: شبیه‌سازی برای کاندیداهای برتر (سطح واقعی در خروجی ثبت می‌شود)
۶. خروجی: ۳–۵ کاندیدای نهایی با پیش‌بینی، نمره اطمینان و تفسیر
(۷. بازخورد آزمایشگاهی در ``active_learning`` پس از ثبت نتایج انجام می‌شود.)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .generation import GenerationCondition, GeneratedStructure
from .interpretability import SurrogateExplainer, ensemble_confidence
from .md_simulation import ConformerEnsembleEngine, MDEngine, ValidationReport, validate_candidates
from .nlp_interface import TargetParameters, parse_query
from .optimization import (
    OBJECTIVE_COLUMNS,
    OBJECTIVE_NAMES,
    Constraints,
    ParetoGuidedRL,
    PredictorObjective,
    crowding_distance,
    objective_matrix,
    pareto_ranks,
)
from .optimization.objectives import DEFAULT_SIZE_RANGE_NM
from .training import ModelBundle

EXPLAIN_TARGETS = (
    "phys_size_nm",
    "phys_zeta_potential_mV",
    "phys_drug_loading_efficiency_percent",
    "bio_cytotoxicity_ic50_ug_ml",
)


@dataclass
class DesignResult:
    """خروجی کامل یک اجرای طراحی."""

    query: Optional[str]
    parameters: TargetParameters
    condition: GenerationCondition
    final_candidates: List[Dict[str, Any]]
    pareto_candidates: pd.DataFrame
    validation: Optional[ValidationReport]
    stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "parameters": self.parameters.to_dict(),
            "final_candidates": self.final_candidates,
            "n_pareto_candidates": int(len(self.pareto_candidates)),
            "stats": self.stats,
            "warnings": self.warnings,
            "md_complete": bool(self.validation.md_complete) if self.validation else False,
        }


def _predict_frame(bundle: ModelBundle, smiles: Sequence[str]):
    """پیش‌بینی هر دو مدل، هم‌ترتیب با ورودی (ردیف نامعتبر = NaN) + انحراف‌معیار ensemble."""
    smiles = list(smiles)
    index = range(len(smiles))
    pm, ps, _ = bundle.physico.predict_with_uncertainty(smiles)
    bm, bs, _ = bundle.bio.predict_with_uncertainty(smiles)
    mean = pd.concat([pm.reindex(index), bm.reindex(index)], axis=1)
    std = pd.concat([ps.reindex(index), bs.reindex(index)], axis=1)
    return mean, std


class DesignPipeline:
    """هماهنگ‌کننده واحدهای ۱ تا ۸ روی یک ``ModelBundle`` آموزش‌دیده."""

    def __init__(self, bundle: ModelBundle, md_engine: Optional[MDEngine] = None, seed: int = 0):
        self.bundle = bundle
        self.md_engine = md_engine or ConformerEnsembleEngine()
        self.seed = seed

    # ------------------------------------------------------------------
    def design(
        self,
        request: Union[str, TargetParameters],
        n_generate: int = 1500,
        n_pareto: int = 15,
        n_md: int = 8,
        n_final: int = 5,
        optimize_iterations: int = 120,
        optimize_batch: int = 24,
        run_md: bool = True,
        explain: bool = True,
        uncertainty_penalty: float = 0.0,
    ) -> DesignResult:
        if not 1 <= n_final <= n_pareto:
            raise ValueError("باید 1 ≤ n_final ≤ n_pareto باشد")
        timings: Dict[str, float] = {}
        warnings: List[str] = []
        clock = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timings[name] = round(now - clock, 3)
            clock = now

        # ۱) ورودی
        query = request if isinstance(request, str) else None
        params = parse_query(request) if isinstance(request, str) else request
        if params.unresolved_terms:
            warnings.append(f"پارامترهای نامشخص (با پیش‌فرض جایگزین شد): {', '.join(params.unresolved_terms)}")
        if params.target_tissue not in (None, "tumor"):
            warnings.append(
                f"بافت هدف «{params.target_tissue}» ثبت شد اما در توابع هدف فعلی اثری ندارد "
                "(اهداف بر پایه TBR/نفوذ سلولی عمومی‌اند)."
            )
        condition = GenerationCondition.from_target_parameters(params)
        size_range = params.size_range_nm or DEFAULT_SIZE_RANGE_NM
        lap("parse")

        # ۲) تولید
        generated, gen_stats = self.bundle.generator.generate(n_generate, condition, seed=self.seed)
        generated_smiles = [s.smiles for s in generated]
        if len(generated) < 0.5 * n_generate:
            warnings.append(
                f"مولد فقط {len(generated)} از {n_generate} ساختار یکتا برای این شرط تحویل داد "
                "(شرط بسیار باریک یا استخر کوچک)."
            )
        lap("generate")

        # ۳+۴) بهینه‌سازی روی ساختارهای تولیدشده + PG-RL در کل فضای قالب‌ها
        objective = PredictorObjective(self.bundle.physico, self.bundle.bio, uncertainty_penalty)
        constraints = Constraints.from_target_parameters(params)
        optimizer = ParetoGuidedRL(
            objective,
            scaffold_type=params.scaffold_type,
            constraints=constraints,
            size_range=size_range,
            batch_size=optimize_batch,
            seed=self.seed,
        )
        rl = optimizer.optimize(iterations=optimize_iterations)
        lap("optimize")

        pool = pd.DataFrame({"smiles": generated_smiles, "source": "generator"})
        generated_obj = objective(generated_smiles)
        pool = pd.concat([pool, generated_obj.reset_index(drop=True)], axis=1)
        rl_front = rl.candidates.copy()
        rl_front["source"] = "pg-rl"
        columns = ["smiles", "source", *OBJECTIVE_COLUMNS.values()]
        pool = pd.concat([pool[columns], rl_front[columns]], ignore_index=True).drop_duplicates("smiles")
        pool = pool[constraints.feasible(pool)] if len(pool) else pool
        if pool.empty:
            warnings.append("هیچ کاندیدایی قیود را برآورده نکرد؛ قیود را آسان‌تر کنید.")
            return DesignResult(query, params, condition, [], pool, None, self._stats(gen_stats, rl, timings), warnings)

        values = objective_matrix({n: pool[c].to_numpy() for n, c in OBJECTIVE_COLUMNS.items()}, size_range)
        front_mask = pareto_ranks(values) == 1
        front = pool[front_mask].reset_index(drop=True)
        front_values = values[front_mask]
        crowd = crowding_distance(front_values)
        order = np.argsort(-np.where(np.isinf(crowd), 1e9, crowd))[:n_pareto]
        pareto = front.iloc[order].reset_index(drop=True)
        pareto_values = front_values[order]
        lap("pareto_select")

        # ۵) شبیه‌سازی
        report: Optional[ValidationReport] = None
        stable: Dict[str, bool] = {}
        if run_md:
            report = validate_candidates(
                pareto["smiles"].tolist(), self.md_engine, max_candidates=n_md, fallback_to_conformers=True
            )
            stable = {r.smiles: r.stable for r in report.results}
            if not report.md_complete:
                warnings.append(
                    "اعتبارسنجی شبیه‌سازی با MD واقعی ≥۱۰۰ ns انجام نشد؛ نتیجه فقط غربالگری کانفورمری است "
                    "(نگاه کنید به fidelity در ValidationReport)."
                )
        lap("validate")

        # ۶) رتبه‌بندی نهایی: ترکیب هدف‌های نرمال‌شده + پایداری شبیه‌سازی
        span = np.maximum(pareto_values.max(axis=0) - pareto_values.min(axis=0), 1e-9)
        composite = ((pareto_values - pareto_values.min(axis=0)) / span).mean(axis=1)
        pareto["composite_score"] = composite
        pareto["md_stable"] = [stable.get(s) for s in pareto["smiles"]]
        ranked = pareto.assign(_rank_key=composite + np.where(pareto["md_stable"] == True, 0.5, 0.0))  # noqa: E712
        finalists = ranked.sort_values("_rank_key", ascending=False).head(n_final)

        # ۶ب) پیش‌بینی کامل + اطمینان + تفسیر
        mean, std = _predict_frame(self.bundle, finalists["smiles"].tolist())
        explainer = None
        if explain:
            # نمونه‌ی جانشین: ساختارهای تولیدشده + همه ساختارهایی که PG-RL ارزیابی کرد
            sample = list(dict.fromkeys(generated_smiles + optimizer.evaluated_smiles()))[:400]
            if len(sample) >= 30:
                explainer = SurrogateExplainer(self.bundle.physico, sample, targets=[t for t in EXPLAIN_TARGETS if t.startswith("phys_")])
                bio_explainer = SurrogateExplainer(self.bundle.bio, sample, targets=[t for t in EXPLAIN_TARGETS if t.startswith("bio_")])
            else:
                warnings.append("نمونه برای ساخت مدل جانشین تفسیر کافی نبود؛ تفسیر حذف شد.")

        final: List[Dict[str, Any]] = []
        for position, (_, row) in enumerate(finalists.iterrows()):
            smiles = row["smiles"]
            predictions = {c: float(mean.iloc[position][c]) for c in mean.columns if pd.notna(mean.iloc[position][c])}
            confidences = {}
            for predictor in (self.bundle.physico, self.bundle.bio):
                members, kept = predictor.member_predictions([smiles])
                if kept:
                    for j, name in enumerate(predictor.target_names):
                        score = ensemble_confidence(members[:, 0, j], scale=float(predictor.scaler.std[j]))
                        confidences[name] = score.confidence
            entry: Dict[str, Any] = {
                "rank": position + 1,
                "smiles": smiles,
                "source": row["source"],
                "predictions": predictions,
                "confidence": confidences,
                "overall_confidence": float(np.mean(list(confidences.values()))) if confidences else None,
                "composite_score": float(row["composite_score"]),
                "md_stable": None if pd.isna(row["md_stable"]) else bool(row["md_stable"]),
            }
            if explainer is not None:
                entry["explanations"] = [e.to_dict() for e in explainer.explain_many(smiles)] + [
                    e.to_dict() for e in bio_explainer.explain_many(smiles)
                ]
                attention = self.bundle.physico.atom_attention(smiles)
                if attention is not None:
                    top = np.argsort(-attention)[:5]
                    entry["top_attention_atoms"] = [{"atom_index": int(i), "weight": float(attention[i])} for i in top]
            final.append(entry)
        lap("finalize")

        return DesignResult(
            query=query,
            parameters=params,
            condition=condition,
            final_candidates=final,
            pareto_candidates=pareto,
            validation=report,
            stats=self._stats(gen_stats, rl, timings),
            warnings=warnings,
        )

    @staticmethod
    def _stats(gen_stats, rl, timings: Dict[str, float]) -> Dict[str, Any]:
        return {
            "generation": gen_stats.to_dict(),
            "optimization": {
                "iterations": rl.iterations,
                "evaluations": rl.evaluations,
                "converged": rl.converged,
                "archive_size": rl.archive_size,
            },
            "timings_seconds": timings,
            "total_seconds": round(sum(timings.values()), 3),
        }
