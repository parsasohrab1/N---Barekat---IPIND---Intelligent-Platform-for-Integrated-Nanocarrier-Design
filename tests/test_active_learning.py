"""تست‌های واحد ۶ (FR-06): نمونه‌برداری، بافر، آستانه به‌روزرسانی و fine-tuning."""

import copy

import numpy as np
import pandas as pd
import pytest

from ipind2.active_learning import (
    EXPERIMENTAL_TO_TARGET,
    ActiveLearningLoop,
    committee_uncertainty,
    select_samples,
)
from ipind2.physicochemical import PHYSICO_TARGET_COLUMNS, PhysicochemicalPredictor


@pytest.fixture(scope="module")
def lipid_model_and_polymers():
    """مدلی که فقط لیپید دیده؛ داده جدید پلیمر است ⇒ shift توزیع واقعی."""
    from ipind2.data_generation.synthetic_data_generator import SyntheticDataGenerator

    generator = SyntheticDataGenerator(51)
    lipids = generator.generate_dataset(500, scaffold_type="lipid", include_pareto_labels=False)
    polymers = generator.generate_dataset(260, scaffold_type="polymer", include_pareto_labels=False)
    model = PhysicochemicalPredictor(n_ensemble=3, hidden_dim=40)
    model.fit(lipids.smiles.tolist(), lipids[list(PHYSICO_TARGET_COLUMNS)].to_numpy(), epochs=14, seed=5)
    return model, lipids, polymers


def _lab_frame(frame: pd.DataFrame, indices) -> pd.DataFrame:
    rows = frame.iloc[list(indices)]
    return pd.DataFrame({
        "molecule_id": list(indices),
        "experimental_size_nm": rows["phys_size_nm"].to_numpy(),
        "experimental_zeta_potential": rows["phys_zeta_potential_mV"].to_numpy(),
        "experimental_loading_efficiency": rows["phys_drug_loading_efficiency_percent"].to_numpy(),
        "experimental_cytotoxicity": np.nan,
    })


class TestUncertaintySampling:
    def test_uncertainty_higher_out_of_distribution(self, lipid_model_and_polymers):
        model, lipids, polymers = lipid_model_and_polymers
        in_dist, _ = committee_uncertainty([model], lipids.smiles.tolist()[:60])
        out_dist, _ = committee_uncertainty([model], polymers.smiles.tolist()[:60])
        assert out_dist.mean() > in_dist.mean()  # Query-by-Committee باید شیفت را حس کند

    def test_requires_a_committee(self, lipid_model_and_polymers):
        model, _, polymers = lipid_model_and_polymers
        single = PhysicochemicalPredictor(n_ensemble=1)
        single.models = model.models[:1]
        single.scaler, single.encoder = model.scaler, model.encoder
        with pytest.raises(ValueError):
            committee_uncertainty([single], polymers.smiles.tolist()[:5])
        with pytest.raises(ValueError):
            committee_uncertainty([], ["CCO"])

    @pytest.mark.parametrize("strategy", ["hybrid", "uncertainty", "diverse_uncertainty", "random"])
    def test_every_strategy_returns_unique_valid_indices(self, lipid_model_and_polymers, strategy):
        model, _, polymers = lipid_model_and_polymers
        pool = polymers.smiles.tolist()[:80] + ["bad((("]
        picks = select_samples(pool, [model], 12, strategy, seed=1)
        assert len(picks) == len(set(picks)) == 12 and 80 not in picks  # SMILES نامعتبر انتخاب نمی‌شود

    def test_uncertainty_strategy_picks_the_most_uncertain(self, lipid_model_and_polymers):
        model, _, polymers = lipid_model_and_polymers
        pool = polymers.smiles.tolist()[:60]
        scores, valid = committee_uncertainty([model], pool)
        picks = select_samples(pool, [model], 5, "uncertainty")
        assert set(picks) == {valid[i] for i in np.argsort(-scores)[:5]}

    def test_diverse_selection_spreads_over_feature_space(self, lipid_model_and_polymers):
        from ipind2.featurization import extended_matrix

        model, _, polymers = lipid_model_and_polymers
        pool = polymers.smiles.tolist()[:120]
        features, _ = extended_matrix(pool)
        features = (features - features.mean(0)) / np.where(features.std(0) < 1e-6, 1, features.std(0))

        def spread(indices):
            chosen = features[indices]
            return np.mean([np.linalg.norm(a - b) for i, a in enumerate(chosen) for b in chosen[i + 1 :]])

        diverse = select_samples(pool, [model], 8, "diverse_uncertainty", seed=0)
        greedy = select_samples(pool, [model], 8, "uncertainty", seed=0)
        assert spread(diverse) >= spread(greedy)

    def test_hybrid_is_reproducible_and_seeded(self, lipid_model_and_polymers):
        model, _, polymers = lipid_model_and_polymers
        pool = polymers.smiles.tolist()[:60]
        assert select_samples(pool, [model], 10, "hybrid", seed=3) == select_samples(pool, [model], 10, "hybrid", seed=3)

    def test_bad_arguments(self, lipid_model_and_polymers):
        model, _, polymers = lipid_model_and_polymers
        with pytest.raises(ValueError):
            select_samples(polymers.smiles.tolist()[:10], [model], 0)
        with pytest.raises(ValueError):
            select_samples(polymers.smiles.tolist()[:10], [model], 3, "oracle")


class TestLoopBuffering:
    def _loop(self, model, polymers, **kwargs):
        lookup = {i: s for i, s in enumerate(polymers.smiles)}
        return ActiveLearningLoop({"physico": copy.deepcopy(model)}, lookup, **kwargs)

    def test_threshold_follows_srs_10_to_50(self, lipid_model_and_polymers):
        model, _, polymers = lipid_model_and_polymers
        loop = self._loop(model, polymers)
        assert loop.min_batch == 10 and loop.max_batch == 50
        loop.ingest(_lab_frame(polymers, range(9)))
        assert not loop.ready() and loop.retrain() is None and loop.pending == 9
        loop.ingest(_lab_frame(polymers, [9]))
        assert loop.ready()

    def test_ingest_filters_unknown_and_empty_rows(self, lipid_model_and_polymers):
        model, _, polymers = lipid_model_and_polymers
        loop = self._loop(model, polymers)
        frame = pd.DataFrame([
            {"molecule_id": 0, "experimental_size_nm": 100.0},
            {"molecule_id": 999999, "experimental_size_nm": 100.0},  # شناخته‌نشده
            {"molecule_id": 1},                                     # بدون مقدار
            {"molecule_id": None, "experimental_size_nm": 5.0},      # بدون شناسه
        ])
        assert loop.ingest(frame) == 1

    def test_experimental_columns_map_to_model_targets(self):
        assert set(EXPERIMENTAL_TO_TARGET.values()) <= set(PHYSICO_TARGET_COLUMNS) | {"bio_cytotoxicity_ic50_ug_ml"}

    def test_invalid_batch_configuration(self, lipid_model_and_polymers):
        model, _, polymers = lipid_model_and_polymers
        with pytest.raises(ValueError):
            self._loop(model, polymers, min_batch=60, max_batch=50)
        with pytest.raises(ValueError):
            ActiveLearningLoop({}, {})


class TestFineTuningUnderShift:
    def test_few_lab_results_reduce_error_on_unseen_class(self, lipid_model_and_polymers):
        """رفتار اصلی FR-06: ~۳۰ نتیجه آزمایشگاهی خطای کلاس جدید را به‌طور معنادار کم می‌کند."""
        model, lipids, polymers = lipid_model_and_polymers
        holdout = polymers.iloc[150:].reset_index(drop=True)
        pool = polymers.iloc[:150].reset_index(drop=True)
        loop = ActiveLearningLoop(
            {"physico": copy.deepcopy(model)}, {i: s for i, s in enumerate(pool.smiles)}, replay=lipids, seed=1
        )
        picks = select_samples(pool.smiles.tolist(), [loop.predictors["physico"]], 30, "hybrid", seed=0)
        loop.ingest(_lab_frame(pool, picks))
        report = loop.retrain(holdout=holdout, epochs=25, learning_rate=1e-3)

        assert report is not None and report.n_new == 30 and report.n_replay == 90 and loop.pending == 0
        assert report.improved()
        assert report.metrics_after["phys_size_nm"] < 0.6 * report.metrics_before["phys_size_nm"]
        assert len(loop.labeled) == 30 and loop.reports == [report]

    def test_partial_labels_do_not_corrupt_unmeasured_targets(self, lipid_model_and_polymers):
        model, lipids, polymers = lipid_model_and_polymers
        loop = ActiveLearningLoop(
            {"physico": copy.deepcopy(model)}, {i: s for i, s in enumerate(polymers.smiles)}, replay=lipids, seed=2
        )
        before = model.predict(polymers.smiles.tolist()[:20])
        loop.ingest(_lab_frame(polymers, range(12)))  # فقط ۳ ستون از ۷ هدف اندازه‌گیری شده
        loop.retrain(epochs=10)
        after = loop.predictors["physico"].predict(polymers.smiles.tolist()[:20])
        assert np.all(np.isfinite(after.to_numpy()))
        # PDI اندازه‌گیری نشده؛ باید در بازه فیزیکی بماند (نه منفجر/NaN)
        assert after["phys_pdi"].between(-0.2, 1.0).all()

    def test_second_retrain_without_new_data_is_noop(self, lipid_model_and_polymers):
        model, _, polymers = lipid_model_and_polymers
        loop = ActiveLearningLoop({"physico": copy.deepcopy(model)}, {i: s for i, s in enumerate(polymers.smiles)})
        loop.ingest(_lab_frame(polymers, range(10)))
        assert loop.retrain(epochs=3) is not None
        assert loop.retrain(epochs=3) is None
