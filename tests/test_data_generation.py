"""تست‌های featurization و تولید داده سنتتیک (شامل رگرسیون باگ‌های کشف‌شده)."""

import numpy as np
import pytest

from ipind2.data_generation.properties import biological_truth, physicochemical_truth
from ipind2.data_generation.synthetic_data_generator import (
    BIO_TARGETS,
    PHYSICO_TARGETS,
    SyntheticDataGenerator,
)
from ipind2.featurization import (
    DESCRIPTOR_DIM,
    EXTENDED_DIM,
    NODE_FEATURE_DIM,
    atom_token_ids,
    batch_graphs,
    canonical_smiles,
    count_distinct_skeletons,
    descriptor_vector,
    extended_dict,
    extended_matrix,
    generic_framework,
    group_counts,
    is_valid_smiles,
    parse_smiles,
    smiles_to_graph,
    validity_rate,
)

QUAT = "CCCCCCCCCCCCCCCC[N+](C)(C)C"
PHOSPHOLIPID = "CCCCCCCCCCCCCC(=O)OCC(OC(=O)CCCCCCCCCCCC)COP(=O)(O)OCCN"


class TestFeaturization:
    def test_descriptor_vector_matches_rdkit(self):
        from rdkit.Chem import Descriptors

        vector = descriptor_vector(QUAT)
        assert vector.shape == (DESCRIPTOR_DIM,)
        assert vector[0] == pytest.approx(Descriptors.MolWt(parse_smiles(QUAT)), rel=1e-5)

    def test_invalid_smiles_return_none_not_exceptions(self):
        for bad in ("", "xyz((", "C1CC", None):
            assert descriptor_vector(bad) is None
            assert smiles_to_graph(bad) is None
            assert atom_token_ids(bad) is None
            assert not is_valid_smiles(bad)

    def test_group_counts_detect_charge_and_functional_groups(self):
        assert group_counts(parse_smiles(QUAT))["n_quat_ammonium"] == 1
        counts = group_counts(parse_smiles(PHOSPHOLIPID))
        assert counts["n_ester"] == 2 and counts["n_acid"] == 1 and counts["n_amine"] == 1
        assert group_counts(parse_smiles("OCCOCCOCCS[Au]"))["n_metal"] == 1

    def test_graph_shapes_and_symmetry(self):
        graph = smiles_to_graph(QUAT)
        assert graph.node_features.shape == (graph.n_atoms, NODE_FEATURE_DIM)
        assert np.allclose(graph.adjacency, graph.adjacency.T)
        assert np.all(np.diag(graph.adjacency) == 1.0)  # self-loop

    def test_graph_truncation(self):
        assert smiles_to_graph("C" * 200, max_atoms=64).n_atoms == 64

    def test_batch_padding_and_mask(self):
        graphs = [smiles_to_graph("CCO"), smiles_to_graph(QUAT)]
        nodes, adjacency, mask = batch_graphs(graphs)
        assert nodes.shape[0] == 2 and mask[0].sum() == graphs[0].n_atoms and mask[1].sum() == graphs[1].n_atoms
        assert adjacency[0, graphs[0].n_atoms :, :].sum() == 0  # padding بدون یال

    def test_canonical_smiles_unifies_equivalent_inputs(self):
        assert canonical_smiles("OCC") == canonical_smiles("C(O)C")

    def test_generic_framework_separates_topologies_not_atoms(self):
        assert generic_framework("CCCCO") == generic_framework("CCCCN")  # فقط نوع اتم فرق دارد
        assert generic_framework("CCCCO") != generic_framework("CC(C)CO")  # توپولوژی متفاوت

    def test_skeleton_and_validity_metrics(self):
        assert count_distinct_skeletons(["CCCC", "CCCCC", "CC(C)C", "bad(("]) == 3
        assert validity_rate(["CCO", "bad(("]) == 0.5 and validity_rate([]) == 0.0

    def test_extended_matrix_skips_invalid(self):
        matrix, kept = extended_matrix(["CCO", "bad((", QUAT])
        assert matrix.shape == (2, EXTENDED_DIM) and kept == [0, 2]


class TestSyntheticDataGenerator:
    def test_regression_descriptors_come_from_the_actual_structure(self, small_dataset):
        """باگ قبلی: Descriptors.FractionCsp3 وجود نداشت و توصیف‌گرها تصادفی می‌شدند."""
        for _, row in small_dataset.head(30).iterrows():
            expected = descriptor_vector(row["smiles"])
            assert row["desc_mol_weight"] == pytest.approx(float(expected[0]), rel=1e-4)
            assert row["desc_logP"] == pytest.approx(float(expected[1]), abs=1e-3)
            assert row["desc_fraction_csp3"] == pytest.approx(float(expected[7]), abs=1e-4)

    def test_regression_structures_are_diverse_and_valid(self, small_dataset):
        assert small_dataset["smiles"].nunique() == len(small_dataset)  # قبلاً ~۱۵۰ مورد متمایز از ۱۰۰k
        assert all(is_valid_smiles(s) for s in small_dataset["smiles"])

    def test_all_three_scaffold_classes_represented(self, small_dataset):
        counts = small_dataset["scaffold_type"].value_counts()
        assert set(counts.index) == {"lipid", "polymer", "metal"} and counts.min() > 40

    def test_no_missing_values_and_all_columns_present(self, small_dataset):
        assert not small_dataset.isna().any().any()
        assert set(PHYSICO_TARGETS) | set(BIO_TARGETS) <= set(small_dataset.columns)

    def test_targets_are_not_saturated_at_clip_bounds(self, small_dataset):
        for column in (*PHYSICO_TARGETS, *BIO_TARGETS):
            values = small_dataset[column].to_numpy()
            assert np.mean(values == values.min()) < 0.05 and np.mean(values == values.max()) < 0.05, column

    def test_structure_property_signal_exists(self, small_dataset):
        correlation = np.corrcoef(small_dataset["desc_mol_weight"], small_dataset["phys_size_nm"])[0, 1]
        assert correlation > 0.2

    def test_physical_trends(self, small_dataset):
        quats = small_dataset[small_dataset["smiles"].str.contains(r"\[N\+\]")]
        others = small_dataset[~small_dataset["smiles"].str.contains(r"\[N\+\]")]
        assert len(quats) > 5
        assert quats["phys_zeta_potential_mV"].mean() > others["phys_zeta_potential_mV"].mean()  # کاتیونی ⇒ زتای بالاتر
        assert quats["bio_cytotoxicity_ic50_ug_ml"].mean() < others["bio_cytotoxicity_ic50_ug_ml"].mean()  # سمی‌تر

    def test_deterministic_for_same_seed(self):
        a = SyntheticDataGenerator(5).generate_dataset(60, include_pareto_labels=False)
        b = SyntheticDataGenerator(5).generate_dataset(60, include_pareto_labels=False)
        assert a.equals(b)
        assert not a.equals(SyntheticDataGenerator(6).generate_dataset(60, include_pareto_labels=False))

    def test_noise_zero_is_noise_free(self):
        features = extended_dict(parse_smiles(QUAT))
        rng = np.random.default_rng(0)
        first = physicochemical_truth("lipid", features, rng, noise=0.0)
        second = physicochemical_truth("lipid", features, np.random.default_rng(99), noise=0.0)
        assert first == second
        assert biological_truth("lipid", features, first, rng, 0.0) == biological_truth("lipid", features, second, rng, 0.0)

    def test_pareto_labels_are_real_nondominated_sets(self):
        df = SyntheticDataGenerator(3).generate_dataset(300)
        from ipind2.optimization import default_objective_matrix, dominates

        values = default_objective_matrix(df)
        front = values[df["is_pareto_optimal"].to_numpy() == 1]
        assert len(front) > 0 and df["pareto_rank"].min() == 1
        for point in front[:20]:
            assert not any(dominates(other, point) for other in values)

    def test_generate_for_gnn_has_per_molecule_targets(self):
        """باگ قبلی: هدف هر گراف، ردیف اول دیتافریم بود."""
        data = SyntheticDataGenerator(2).generate_for_gnn(40)
        assert data["targets"].shape == (len(data["graphs"]), len(PHYSICO_TARGETS))
        assert data["targets"][:, 0].std() > 0, "همه اهداف یکسان‌اند ⇒ باگ هدف ردیف اول"
        assert [g.smiles for g in data["graphs"]] == data["dataframe"]["smiles"].tolist()

    def test_single_molecule_api(self):
        molecule = SyntheticDataGenerator(1).generate_molecule("polymer")
        assert molecule["scaffold_type"] == "polymer" and set(molecule["physicochemical"]) >= {"size_nm", "zeta_potential_mV"}

    def test_csv_round_trip(self, tmp_path):
        generator = SyntheticDataGenerator(4)
        frame = generator.generate_dataset(30, include_pareto_labels=False)
        import pandas as pd

        path = generator.save_dataset(frame, str(tmp_path / "d.csv"))
        assert pd.read_csv(path).shape == frame.shape
