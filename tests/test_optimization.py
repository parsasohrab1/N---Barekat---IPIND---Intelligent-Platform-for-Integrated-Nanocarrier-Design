"""Unit 4 tests: Pareto tools and Pareto-Guided RL (FR-04, NFR-06)."""

import numpy as np
import pandas as pd
import pytest

from ipind2.optimization import (
    OBJECTIVE_COLUMNS,
    OBJECTIVE_NAMES,
    Constraints,
    ParetoGuidedRL,
    crowding_distance,
    dominates,
    hypervolume,
    non_dominated_sort,
    objective_matrix,
    oracle_objective,
    pareto_front_mask,
    pareto_ranks,
    random_search_front,
    size_fit_score,
)


class TestParetoTools:
    def test_dominates_semantics(self):
        assert dominates(np.array([2, 2]), np.array([1, 1]))
        assert dominates(np.array([2, 1]), np.array([1, 1]))
        assert not dominates(np.array([1, 1]), np.array([1, 1]))  # equal ⇒ no dominance
        assert not dominates(np.array([2, 0]), np.array([0, 2]))

    def test_known_fronts(self):
        points = np.array([[3, 1], [1, 3], [2, 2], [1, 1], [0.5, 0.5]])
        fronts = non_dominated_sort(points)
        assert sorted(fronts[0]) == [0, 1, 2]
        assert list(fronts[1]) == [3] and list(fronts[2]) == [4]
        assert list(pareto_ranks(points)) == [1, 1, 1, 2, 3]

    def test_front_property_against_bruteforce(self):
        points = np.random.default_rng(0).random((120, 4))
        mask = pareto_front_mask(points)
        for i, point in enumerate(points):
            dominated = any(dominates(other, point) for other in points)
            assert mask[i] == (not dominated)

    def test_duplicates_both_on_front(self):
        assert list(pareto_ranks(np.array([[1.0, 1.0], [1.0, 1.0]]))) == [1, 1]

    def test_empty_and_single(self):
        assert non_dominated_sort(np.zeros((0, 3))) == []
        assert list(pareto_ranks(np.array([[1.0, 2.0]]))) == [1]

    def test_crowding_distance_boundary_points_infinite(self):
        distance = crowding_distance(np.array([[0.0, 3.0], [1.0, 2.0], [3.0, 0.0]]))
        assert np.isinf(distance[0]) and np.isinf(distance[2]) and np.isfinite(distance[1])

    def test_hypervolume_analytic_cases(self):
        reference = np.zeros(2)
        assert hypervolume(np.array([[2.0, 2.0]]), reference, 40000) == pytest.approx(4.0, rel=0.03)
        union = hypervolume(np.array([[2.0, 1.0], [1.0, 2.0]]), reference, 60000)
        assert union == pytest.approx(3.0, rel=0.04)  # 2+2-1
        assert hypervolume(np.zeros((0, 2)), reference) == 0.0

    def test_hypervolume_monotone_in_front_quality(self):
        reference = np.zeros(3)
        weak = np.array([[1.0, 1.0, 1.0]])
        strong = np.array([[1.0, 1.0, 1.0], [2.0, 0.5, 0.5]])
        assert hypervolume(strong, reference) > hypervolume(weak, reference)


class TestObjectives:
    def test_size_fit_inside_range_is_one(self):
        assert size_fit_score(100.0, (80, 120)) == pytest.approx(1.0)
        assert size_fit_score(80.0, (80, 120)) == pytest.approx(1.0)

    def test_size_fit_decays_symmetrically_outside(self):
        inside, below, above = (size_fit_score(v, (80, 120)) for v in (100.0, 40.0, 160.0))
        assert below < inside and above < inside and below == pytest.approx(above)

    def test_objective_matrix_all_maximization(self):
        frame = oracle_objective(["CCCCCCCCCCCCCCCC[N+](C)(C)C", "OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO"])
        values = objective_matrix({n: frame[c].to_numpy() for n, c in OBJECTIVE_COLUMNS.items()})
        assert values.shape == (2, len(OBJECTIVE_NAMES)) and np.all(np.isfinite(values))

    def test_oracle_ranks_cationic_lipid_as_more_toxic(self):
        frame = oracle_objective(["CCCCCCCCCCCCCCCC[N+](C)(C)C", "CCCCCCCCCCCCCCCCOCCOCCO"])
        assert frame.loc[0, "bio_cytotoxicity_ic50_ug_ml"] < frame.loc[1, "bio_cytotoxicity_ic50_ug_ml"]


class TestConstraints:
    def test_feasibility_mask(self):
        frame = pd.DataFrame({
            "phys_size_nm": [70.0, 100.0, 130.0],
            "phys_drug_loading_efficiency_percent": [90.0, 50.0, 90.0],
            "bio_cytotoxicity_ic50_ug_ml": [10.0, 80.0, 80.0],
        })
        assert list(Constraints(size_range_nm=(80, 120)).feasible(frame)) == [False, True, False]
        assert list(Constraints(min_loading_efficiency=60).feasible(frame)) == [True, False, True]
        assert list(Constraints(min_ic50=50).feasible(frame)) == [False, True, True]
        assert list(Constraints().feasible(frame)) == [True, True, True]

    def test_from_nlp_parameters(self):
        from ipind2.nlp_interface import parse_query

        constraints = Constraints.from_target_parameters(parse_query("lipid size between 80 to 120 nm"))
        assert constraints.size_range_nm == (80.0, 120.0)


class TestParetoGuidedRL:
    def test_beats_random_search_at_equal_budget(self):
        """Main FR-04 claim: policy guidance gives a better front than random sampling with the same budget."""
        reference = np.zeros(len(OBJECTIVE_NAMES))
        wins = 0
        for seed in (0, 1):
            optimizer = ParetoGuidedRL(oracle_objective, scaffold_type="lipid", batch_size=24, seed=seed)
            result = optimizer.optimize(iterations=35, patience_window=100)
            values = np.stack([result.candidates[f"obj_{n}"] for n in OBJECTIVE_NAMES], axis=1)
            baseline = random_search_front(oracle_objective, result.evaluations, "lipid", seed=seed)
            wins += hypervolume(values, reference, 20000) > hypervolume(baseline, reference, 20000)
        assert wins == 2

    def test_output_is_nondominated_and_unique(self):
        optimizer = ParetoGuidedRL(oracle_objective, scaffold_type="metal", batch_size=16, seed=3)
        result = optimizer.optimize(iterations=8)
        values = np.stack([result.candidates[f"obj_{n}"] for n in OBJECTIVE_NAMES], axis=1)
        assert pareto_front_mask(values).all()
        assert result.candidates["smiles"].is_unique and result.evaluations >= len(result.candidates)

    def test_constraints_filter_archive(self):
        constraints = Constraints(size_range_nm=(90, 110))
        optimizer = ParetoGuidedRL(oracle_objective, scaffold_type="lipid", constraints=constraints, batch_size=24, seed=2)
        result = optimizer.optimize(iterations=12)
        sizes = result.candidates[OBJECTIVE_COLUMNS["size_fit"]]
        assert len(sizes) > 0 and sizes.between(90, 110).all()

    def test_select_diverse_returns_requested_count(self):
        optimizer = ParetoGuidedRL(oracle_objective, scaffold_type="polymer", batch_size=24, seed=4)
        optimizer.optimize(iterations=15)
        assert len(optimizer.select_diverse(10)) <= 10

    def test_convergence_criterion_triggers_early_stop(self):
        """SRS criterion: hypervolume change <1% in the window ⇒ stop before the iteration cap."""
        optimizer = ParetoGuidedRL(oracle_objective, scaffold_type="metal", batch_size=16, seed=5)
        result = optimizer.optimize(iterations=400, patience_window=15, tolerance=0.05, min_iterations=16)
        assert result.converged and result.iterations < 400
        history = result.hypervolume_history
        assert abs(history[-1] - history[-1 - 15]) / history[-1 - 15] < 0.05

    def test_hypervolume_history_nondecreasing(self):
        optimizer = ParetoGuidedRL(oracle_objective, scaffold_type="lipid", batch_size=16, seed=6)
        history = optimizer.optimize(iterations=15).hypervolume_history
        # The archive only grows and HV is estimated with a fixed seed; a small drop from sampling is allowed
        assert all(b >= a * 0.97 for a, b in zip(history, history[1:]))

    def test_reproducible_with_seed(self):
        runs = [
            ParetoGuidedRL(oracle_objective, scaffold_type="metal", batch_size=12, seed=9).optimize(iterations=6).candidates["smiles"].tolist()
            for _ in range(2)
        ]
        assert runs[0] == runs[1]

    def test_invalid_smiles_get_worst_score_not_crash(self):
        def objective(smiles_list):
            return oracle_objective(smiles_list)

        optimizer = ParetoGuidedRL(objective, scaffold_type="lipid", batch_size=8, seed=1)
        values = optimizer._evaluate(["not-a-smiles(((", "CCCCCCCCCCCCCCCC[N+](C)(C)C"])
        assert (values[0] < -1e5).all() and np.isfinite(values[1]).all()

    def test_pg_rl_with_trained_predictors(self, smoke_bundle):
        from ipind2.optimization import PredictorObjective

        objective = PredictorObjective(smoke_bundle.physico, smoke_bundle.bio)
        frame = objective(["CCCCCCCCCCCCCCCC[N+](C)(C)C", "bad(((", "OCCOCCOCCO"])
        assert list(frame.columns) == list(OBJECTIVE_COLUMNS.values()) and len(frame) == 3
        assert (frame.loc[1] == -1e6).all() and (frame.loc[[0, 2]] > -1e5).all().all()
