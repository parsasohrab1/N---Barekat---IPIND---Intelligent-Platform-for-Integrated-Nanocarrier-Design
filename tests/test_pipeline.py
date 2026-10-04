"""تست‌های خط لوله انتها‌به‌انتها، مدل جانشین تفسیر و حلقه کامل بازخورد (جریان داده SRS §3.2)."""

import numpy as np
import pytest

from ipind2.featurization import is_valid_smiles
from ipind2.interpretability import SurrogateExplainer
from ipind2.nlp_interface import TargetParameters
from ipind2.pipeline import DesignPipeline
from ipind2.training import ModelBundle

QUERY = "یک نانوحامل لیپیدی برای هدف‌گیری تومور، اندازه بین ۸۰ تا ۱۲۰ نانومتر"


@pytest.fixture(scope="module")
def pipeline(smoke_bundle):
    return DesignPipeline(smoke_bundle, seed=3)


@pytest.fixture(scope="module")
def result(pipeline):
    return pipeline.design(QUERY, n_generate=150, n_pareto=8, n_md=3, n_final=3, optimize_iterations=12, run_md=True, explain=True)


class TestEndToEnd:
    def test_parses_persian_query(self, result):
        assert result.parameters.scaffold_type == "lipid" and result.parameters.size_range_nm == (80.0, 120.0)
        assert result.condition.scaffold_type == "lipid" and result.condition.size_nm == 100.0

    def test_final_candidates_valid_ranked_and_within_request(self, result):
        candidates = result.final_candidates
        assert 1 <= len(candidates) <= 3
        assert [c["rank"] for c in candidates] == list(range(1, len(candidates) + 1))
        assert all(is_valid_smiles(c["smiles"]) for c in candidates)
        assert len({c["smiles"] for c in candidates}) == len(candidates)

    def test_predictions_cover_all_twelve_unit_outputs_with_confidence(self, result):
        candidate = result.final_candidates[0]
        assert len(candidate["predictions"]) == 14  # ۷ فیزیکوشیمیایی + ۷ زیستی (۳ رده سلولی)
        assert all(0.0 <= v <= 1.0 for v in candidate["confidence"].values())
        assert 0.0 <= candidate["overall_confidence"] <= 1.0

    def test_size_constraint_honoured_by_all_pareto_candidates(self, result):
        sizes = result.pareto_candidates["phys_size_nm"]
        assert len(sizes) > 0 and sizes.between(80, 120).all()

    def test_pareto_set_is_nondominated(self, result):
        from ipind2.optimization import OBJECTIVE_COLUMNS, OBJECTIVE_NAMES, objective_matrix, pareto_front_mask

        frame = result.pareto_candidates
        values = objective_matrix({n: frame[c].to_numpy() for n, c in OBJECTIVE_COLUMNS.items()}, (80, 120))
        assert pareto_front_mask(values).all()

    def test_explanations_have_fidelity_and_additive_shap(self, result):
        explanations = result.final_candidates[0]["explanations"]
        assert {e["target"] for e in explanations} >= {"phys_size_nm", "bio_cytotoxicity_ic50_ug_ml"}
        for explanation in explanations:
            assert -1.0 <= explanation["fidelity"] <= 1.0 and explanation["top_features"]
        attention = result.final_candidates[0]["top_attention_atoms"]
        assert attention and all(a["weight"] >= 0 for a in attention)

    def test_md_honesty_surfaces_in_warnings_and_flags(self, result):
        assert result.validation is not None and not result.validation.md_complete
        assert any("MD واقعی" in w for w in result.warnings)
        assert result.to_dict()["md_complete"] is False

    def test_stats_report_timings_and_generation_validity(self, result):
        assert result.stats["generation"]["validity_rate"] == 1.0
        assert set(result.stats["timings_seconds"]) >= {"generate", "optimize", "validate", "finalize"}

    def test_structured_input_equivalent_to_text(self, pipeline):
        params = TargetParameters(scaffold_type="metal", size_range_nm=(40.0, 80.0))
        out = pipeline.design(params, n_generate=60, n_pareto=4, n_final=2, optimize_iterations=6, run_md=False, explain=False)
        assert all(c["source"] in ("generator", "pg-rl") for c in out.final_candidates)
        assert out.pareto_candidates["phys_size_nm"].between(40, 80).all()

    def test_impossible_constraints_produce_clear_warning_not_crash(self, pipeline):
        params = TargetParameters(scaffold_type="lipid", size_range_nm=(1.0, 2.0))
        out = pipeline.design(params, n_generate=50, n_pareto=3, n_final=1, optimize_iterations=4, run_md=False, explain=False)
        assert out.final_candidates == [] and any("قیود" in w for w in out.warnings)

    def test_unknown_tissue_noted_not_silently_ignored(self, pipeline):
        params = TargetParameters(scaffold_type="lipid", target_tissue="spleen", size_range_nm=(60.0, 140.0))
        out = pipeline.design(params, n_generate=50, n_pareto=3, n_final=1, optimize_iterations=4, run_md=False, explain=False)
        assert any("spleen" in w for w in out.warnings)

    def test_argument_validation(self, pipeline):
        with pytest.raises(ValueError):
            pipeline.design(QUERY, n_final=9, n_pareto=3)

    def test_same_seed_reproducible(self, smoke_bundle):
        kwargs = dict(n_generate=60, n_pareto=4, n_final=2, optimize_iterations=5, run_md=False, explain=False)
        a = DesignPipeline(smoke_bundle, seed=11).design(QUERY, **kwargs).final_candidates
        b = DesignPipeline(smoke_bundle, seed=11).design(QUERY, **kwargs).final_candidates
        assert [c["smiles"] for c in a] == [c["smiles"] for c in b]


class TestBundlePersistence:
    def test_save_load_round_trip_gives_same_design(self, smoke_bundle, tmp_path):
        smoke_bundle.save(str(tmp_path / "bundle"))
        loaded = ModelBundle.load(str(tmp_path / "bundle"))
        assert loaded.version == smoke_bundle.version and set(loaded.metrics) >= {"physico", "bio", "nfr", "noise_ceiling_r2"}
        smiles = ["CCCCCCCCCCCCCCCC[N+](C)(C)C", "OCCOCCOCCO"]
        assert np.allclose(smoke_bundle.physico.predict(smiles).to_numpy(), loaded.physico.predict(smiles).to_numpy(), atol=1e-4)
        kwargs = dict(n_generate=40, n_pareto=3, n_final=1, optimize_iterations=3, run_md=False, explain=False)
        a = DesignPipeline(smoke_bundle, seed=2).design(QUERY, **kwargs).final_candidates
        b = DesignPipeline(loaded, seed=2).design(QUERY, **kwargs).final_candidates
        assert [c["smiles"] for c in a] == [c["smiles"] for c in b]

    def test_manifest_records_nfr_and_noise_ceiling(self, smoke_bundle):
        nfr = smoke_bundle.metrics["nfr"]
        assert set(nfr) == {"NFR-01", "NFR-02", "NFR-03", "FR-03"}
        ceilings = smoke_bundle.metrics["noise_ceiling_r2"]
        assert len(ceilings) == 14 and all(0.5 < v <= 1.0 for v in ceilings.values())


class TestSurrogateExplainer:
    def test_shap_values_are_additive_and_fidelity_reported(self, smoke_bundle):
        pool = [s.smiles for s in smoke_bundle.generator.pool[:200]]
        explainer = SurrogateExplainer(smoke_bundle.physico, pool, targets=["phys_size_nm"])
        explanation = explainer.explain(pool[0], "phys_size_nm")
        total = explanation.explanation.base_value + sum(a.shap_value for a in explanation.explanation.attributions)
        assert total == pytest.approx(explanation.explanation.prediction, abs=1e-3)
        assert explanation.fidelity > 0.5

    def test_invalid_inputs(self, smoke_bundle):
        pool = [s.smiles for s in smoke_bundle.generator.pool[:60]]
        explainer = SurrogateExplainer(smoke_bundle.physico, pool, targets=["phys_size_nm"])
        assert explainer.explain("bad(((", "phys_size_nm") is None
        with pytest.raises(KeyError):
            explainer.explain(pool[0], "unknown")
        with pytest.raises(ValueError):
            SurrogateExplainer(smoke_bundle.physico, pool[:5])
