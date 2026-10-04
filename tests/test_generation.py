"""تست‌های واحد ۱: کتابخانه ترکیبیاتی، مولد شرطی، کتابخانه seed (FR-01، NFR-04، NFR-08، NFR-10)."""

import time

import numpy as np
import pandas as pd
import pytest

from ipind2.featurization import extended_dict, is_valid_smiles, parse_smiles
from ipind2.generation import (
    SCAFFOLD_TYPES,
    TEMPLATES,
    CombinatorialLibrary,
    ConditionalStructureGenerator,
    GenerationCondition,
    generate_library,
    theoretical_library_size,
)
from ipind2.generation.seed_library import build_seed_library, read_public_smiles
from ipind2.nlp_interface import parse_query


class TestCombinatorialLibrary:
    @pytest.mark.parametrize("scaffold_type", SCAFFOLD_TYPES)
    def test_validity_above_fr01_threshold_per_class(self, scaffold_type):
        structures, stats = generate_library(400, scaffold_type=scaffold_type, seed=1)
        assert stats.validity_rate > 0.95  # الزام FR-01
        assert all(s.scaffold_type == scaffold_type for s in structures)

    def test_every_template_yields_valid_smiles(self):
        rng = np.random.default_rng(0)
        for template in TEMPLATES:
            options = template.slot_options()
            for _ in range(40):
                choices = {slot: opts[int(rng.integers(len(opts)))] for slot, opts in options.items()}
                assert is_valid_smiles(template.build(choices)), (template.name, choices)

    def test_uniqueness_and_count(self):
        structures, stats = generate_library(500, seed=2)
        assert len({s.smiles for s in structures}) == stats.unique == len(structures) == 500

    def test_skeleton_diversity_meets_fr01(self):
        _, stats = generate_library(1200, seed=3, track_skeletons=True)
        assert stats.distinct_skeletons >= 500  # الزام FR-01: ≥۵۰۰ اسکلت متمایز

    def test_reproducible_with_seed(self):
        a, _ = generate_library(50, seed=7)
        b, _ = generate_library(50, seed=7)
        assert [s.smiles for s in a] == [s.smiles for s in b]
        assert [s.smiles for s in a] != [s.smiles for s in generate_library(50, seed=8)[0]]

    def test_class_balancing_by_default(self):
        structures, _ = generate_library(900, seed=4)
        counts = pd.Series([s.scaffold_type for s in structures]).value_counts(normalize=True)
        assert counts.min() > 0.2  # بدون تعادل، فلزی ≈ ۸٪ می‌شد

    def test_provenance_recorded(self):
        structure = generate_library(5, seed=1)[0][0]
        assert structure.template in {t.name for t in TEMPLATES} and structure.slots

    def test_space_size_supports_nfr08(self):
        assert theoretical_library_size() >= 1_000_000

    def test_throughput_supports_nfr04(self):
        """NFR-04: ۱۰۰k ساختار در <۱۰ دقیقه ⇒ حداقل ~۱۷۰ ساختار/ثانیه؛ با حاشیه امن."""
        started = time.perf_counter()
        _, stats = generate_library(3000, seed=5)
        rate = stats.unique / (time.perf_counter() - started)
        assert rate > 400, f"نرخ {rate:.0f}/s برای رساندن ۱۰۰k به <۱۰ دقیقه کافی نیست"

    def test_regression_small_templates_saturate_without_blocking_large_requests(self):
        """
        باگ کشف‌شده در اعتبارسنجی NFR-04: با وزن یکنواخت ایستا، قالب‌های کوچک (۴۸–۳۶۸ ترکیب)
        زود اشباع می‌شدند و درخواست ۱۰۰k فقط ~۷۹k ساختار یکتا می‌داد. دو قالب کوچک (جمعاً ۴۱۶
        ترکیب) را جدا می‌کنیم و ۱۸۰ ساختار یکتا می‌خواهیم؛ سامپلر ایستا روی هر ۶ بذر آزمایشی
        کمتر از ۱۸۰ می‌داد (۱۵۴–۱۶۹)، سامپلر وفقی همه را تحویل می‌دهد.
        """
        for seed in (1, 2, 3, 4):
            library = CombinatorialLibrary("metal", seed=seed)
            small = [t for t in library.templates if t.name in ("metal_peg_ligand", "metal_ligand")]
            assert sum(t.combination_count() for t in small) == 416
            library.templates = small
            library._weights = [0.5, 0.5]
            library._template_index = {t.name: i for i, t in enumerate(small)}
            library._combinations = [t.combination_count() for t in small]
            structures, stats = library.generate(180)
            assert stats.unique == len(structures) == 180, f"seed={seed}"

    def test_request_beyond_the_whole_space_stops_instead_of_hanging(self):
        library = CombinatorialLibrary("metal", seed=1)
        small = [t for t in library.templates if t.name == "metal_peg_ligand"]
        library.templates, library._weights = small, [1.0]
        library._template_index = {small[0].name: 0}
        library._combinations = [small[0].combination_count()]
        structures, stats = library.generate(500, max_attempt_factor=50.0)
        assert 0 < len(structures) <= 48 and stats.unique == len(structures)

    def test_enumerate_all_is_deterministic_and_duplicate_free(self):
        first = [s.smiles for s in CombinatorialLibrary("metal").enumerate_all(limit=300)]
        second = [s.smiles for s in CombinatorialLibrary("metal").enumerate_all(limit=300)]
        assert first == second and len(set(first)) == len(first)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            CombinatorialLibrary("ceramic")
        with pytest.raises(ValueError):
            generate_library(0)


@pytest.fixture(scope="module")
def trained_generator():
    from ipind2.data_generation.synthetic_data_generator import SyntheticDataGenerator

    data = SyntheticDataGenerator(31).generate_dataset(2500, include_pareto_labels=False)
    return ConditionalStructureGenerator(seed=2).fit(data, pool_size=5000, epochs=40)


class TestConditionalGenerator:
    def test_all_outputs_valid_and_unique(self, trained_generator):
        structures, stats = trained_generator.generate(150, GenerationCondition(scaffold_type="lipid", size_nm=100.0))
        assert stats.validity_rate == 1.0
        assert len({s.smiles for s in structures}) == len(structures) > 50
        assert all(s.scaffold_type == "lipid" for s in structures)

    def test_training_converged(self, trained_generator):
        history = trained_generator.history["vae_loss"]
        assert history[-1] < 0.5 * history[0]

    def test_condition_steers_output(self, trained_generator):
        """شرط اندازه باید توزیع اندازه خروجی را به سمت هدف جابه‌جا کند (نسبت به شرط مخالف)."""
        from ipind2.data_generation.properties import physicochemical_truth

        def mean_size(structures):
            rng = np.random.default_rng(0)
            return float(np.mean([
                physicochemical_truth(s.scaffold_type, extended_dict(parse_smiles(s.smiles)), rng, 0.0)["size_nm"]
                for s in structures
            ]))

        large, _ = trained_generator.generate(120, GenerationCondition("polymer", size_nm=140.0), seed=1)
        small, _ = trained_generator.generate(120, GenerationCondition("polymer", size_nm=70.0), seed=1)
        assert mean_size(large) > mean_size(small) + 3.0

    def test_conditional_is_tighter_than_unconditional(self, trained_generator):
        from ipind2.data_generation.properties import physicochemical_truth

        def sizes(structures):
            rng = np.random.default_rng(0)
            return np.array([
                physicochemical_truth(s.scaffold_type, extended_dict(parse_smiles(s.smiles)), rng, 0.0)["size_nm"]
                for s in structures
            ])

        conditioned, _ = trained_generator.generate(150, GenerationCondition("lipid", size_nm=104.0))
        baseline, _ = generate_library(150, scaffold_type="lipid", seed=9)
        target = 104.0
        assert np.mean(np.abs(sizes(conditioned) - target)) < np.mean(np.abs(sizes(baseline) - target))

    @pytest.mark.parametrize("model", ["vae", "gan", "ensemble"])
    def test_each_model_head_generates(self, trained_generator, model):
        structures, _ = trained_generator.generate(30, GenerationCondition("metal"), model=model)
        assert structures

    def test_narrow_condition_still_delivers_many_structures(self, trained_generator):
        """باگ قبلی: شرط باریک → ۴۸ از ۴۰۰ تحویل. اکنون جست‌وجو گسترش می‌یابد."""
        structures, stats = trained_generator.generate(200, GenerationCondition("lipid", size_nm=100.0, zeta_mV=5.0, loading_efficiency=65.0, ic50=60.0))
        assert len(structures) >= 120 and stats.attempts >= stats.unique

    def test_nlp_bridge(self, trained_generator):
        params = parse_query("نانوحامل لیپیدی برای تومور، اندازه بین ۸۰ تا ۱۲۰ نانومتر")
        condition = GenerationCondition.from_target_parameters(params)
        assert condition.scaffold_type == "lipid" and condition.size_nm == 100.0
        structures, _ = trained_generator.generate(20, condition)
        assert structures and all(s.scaffold_type == "lipid" for s in structures)

    def test_save_load_round_trip(self, trained_generator, tmp_path):
        trained_generator.save(str(tmp_path / "gen"))
        loaded = ConditionalStructureGenerator.load(str(tmp_path / "gen"))
        condition = GenerationCondition("lipid", size_nm=95.0)
        a, _ = trained_generator.generate(25, condition, seed=11)
        b, _ = loaded.generate(25, condition, seed=11)
        assert [s.smiles for s in a] == [s.smiles for s in b]

    def test_untrained_and_bad_arguments(self, trained_generator):
        with pytest.raises(RuntimeError):
            ConditionalStructureGenerator().generate(5)
        with pytest.raises(ValueError):
            trained_generator.generate(0)
        with pytest.raises(ValueError):
            trained_generator.generate(5, model="diffusion")
        with pytest.raises(ValueError):
            trained_generator.generate(5, GenerationCondition(scaffold_type="ceramic"))

    def test_fit_requires_columns_and_enough_rows(self):
        with pytest.raises(ValueError):
            ConditionalStructureGenerator().fit(pd.DataFrame({"smiles": ["CCO"]}))


class TestSeedLibrary:
    def test_builds_unique_valid_parquet(self, tmp_path):
        import pyarrow.parquet as pq

        report = build_seed_library(str(tmp_path / "seed.parquet"), n=3000, seed=1, chunk_size=1000)
        table = pq.read_table(report.path).to_pandas()
        assert report.n_total == len(table) == 3000 and table["smiles"].nunique() == 3000
        assert (table["source"] == "combinatorial").all() and report.n_public == 0
        assert all(is_valid_smiles(s) for s in table["smiles"].head(100))

    def test_public_files_merge_and_deduplicate(self, tmp_path):
        public = tmp_path / "pubchem.smi"
        public.write_text("# comment\nCCO ethanol\nOCC dup-of-ethanol\nCC(=O)O acetic\nbad(((\n", encoding="utf-8")
        report = build_seed_library(str(tmp_path / "s.parquet"), n=500, public_files=[str(public)], seed=2)
        assert report.n_public == 2 and report.n_rejected >= 2  # تکراری و نامعتبر رد شد
        assert report.n_total == 500
        assert not report.meets_public_source_requirement

    def test_csv_input_requires_smiles_column(self, tmp_path):
        bad = tmp_path / "x.csv"
        bad.write_text("name,formula\na,b\n", encoding="utf-8")
        with pytest.raises(ValueError):
            list(read_public_smiles([str(bad)]))
        good = tmp_path / "y.csv"
        good.write_text("id,SMILES\n1,CCO\n2,CCN\n", encoding="utf-8")
        assert list(read_public_smiles([str(good)])) == ["CCO", "CCN"]

    def test_requirement_flags(self, tmp_path):
        report = build_seed_library(str(tmp_path / "s.parquet"), n=1200, seed=3)
        assert not report.meets_size_requirement  # فقط وقتی ≥۱M
