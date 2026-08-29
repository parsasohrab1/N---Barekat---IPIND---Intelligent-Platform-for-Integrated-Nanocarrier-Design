import pandas as pd
import pytest

from ipind2.benchmarking import (
    REGISTRY,
    BenchmarkHistory,
    assert_no_regression,
    check_regression,
    list_reference_datasets,
    load_reference_dataset,
    r_squared,
    rmse,
    run_benchmark,
)


class TestMetrics:
    def test_rmse_perfect_prediction(self):
        assert rmse([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        assert rmse([0, 0], [3, 4]) == pytest.approx(3.5355339, rel=1e-5)

    def test_r_squared_perfect_prediction(self):
        assert r_squared([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)

    def test_r_squared_mean_prediction_is_zero(self):
        y_true = [1, 2, 3, 4]
        y_pred = [2.5, 2.5, 2.5, 2.5]
        assert r_squared(y_true, y_pred) == pytest.approx(0.0, abs=1e-9)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            rmse([1, 2, 3], [1, 2])


class TestDatasetRegistry:
    def test_registry_has_known_datasets(self):
        datasets = list_reference_datasets()
        assert "lnp-622" in datasets
        assert "lance" in datasets

    def test_load_unknown_dataset_raises(self):
        with pytest.raises(KeyError):
            load_reference_dataset("not-a-real-dataset", "irrelevant.csv")

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_reference_dataset("lnp-622", str(tmp_path / "missing.csv"))

    def test_load_missing_target_column_raises(self, tmp_path):
        csv_path = tmp_path / "lnp622.csv"
        pd.DataFrame({"some_other_column": [1, 2, 3]}).to_csv(csv_path, index=False)
        with pytest.raises(ValueError):
            load_reference_dataset("lnp-622", str(csv_path))

    def test_load_valid_dataset(self, tmp_path):
        csv_path = tmp_path / "lnp622.csv"
        target_column = REGISTRY["lnp-622"].target_column
        pd.DataFrame({"feature_a": [1, 2], target_column: [0.5, 0.9]}).to_csv(csv_path, index=False)
        df = load_reference_dataset("lnp-622", str(csv_path))
        assert target_column in df.columns
        assert len(df) == 2


@pytest.fixture
def toy_dataset():
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "target": [2.0, 4.0, 6.0, 8.0],
        }
    )


class TestRunBenchmark:
    def test_run_benchmark_perfect_model(self, toy_dataset):
        def predict_fn(X):
            return X["feature_a"] * 2

        result = run_benchmark(
            predict_fn,
            toy_dataset,
            target_column="target",
            model_version="v1",
            dataset_name="toy",
        )
        assert result.n_samples == 4
        assert result.rmse == pytest.approx(0.0, abs=1e-9)
        assert result.r2 == pytest.approx(1.0)

    def test_run_benchmark_missing_target_column_raises(self, toy_dataset):
        with pytest.raises(ValueError):
            run_benchmark(lambda X: X["feature_a"], toy_dataset, target_column="nope")


class TestBenchmarkHistoryAndRegression:
    def test_first_run_has_no_previous(self, tmp_path, toy_dataset):
        history = BenchmarkHistory(str(tmp_path / "history.json"))
        result = run_benchmark(
            lambda X: X["feature_a"] * 2, toy_dataset, "target", dataset_name="toy"
        )
        report = check_regression(history, result)
        assert report.regressed is False
        assert report.delta_rmse is None

    def test_detects_regression(self, tmp_path, toy_dataset):
        history_path = tmp_path / "history.json"
        history = BenchmarkHistory(str(history_path))

        good_result = run_benchmark(
            lambda X: X["feature_a"] * 2, toy_dataset, "target", dataset_name="toy", model_version="v1"
        )
        history.append(good_result)

        # مدل بدتر: ضریب اشتباه
        bad_result = run_benchmark(
            lambda X: X["feature_a"] * 1.5, toy_dataset, "target", dataset_name="toy", model_version="v2"
        )
        report = check_regression(history, bad_result)
        assert report.regressed is True
        assert report.delta_rmse > 0

    def test_no_regression_when_model_improves(self, tmp_path, toy_dataset):
        history = BenchmarkHistory(str(tmp_path / "history.json"))
        worse_result = run_benchmark(
            lambda X: X["feature_a"] * 1.5, toy_dataset, "target", dataset_name="toy", model_version="v1"
        )
        history.append(worse_result)

        better_result = run_benchmark(
            lambda X: X["feature_a"] * 2, toy_dataset, "target", dataset_name="toy", model_version="v2"
        )
        report = check_regression(history, better_result)
        assert report.regressed is False

    def test_assert_no_regression_raises_on_regression(self, tmp_path, toy_dataset):
        history = BenchmarkHistory(str(tmp_path / "history.json"))
        good_result = run_benchmark(
            lambda X: X["feature_a"] * 2, toy_dataset, "target", dataset_name="toy", model_version="v1"
        )
        history.append(good_result)

        bad_result = run_benchmark(
            lambda X: X["feature_a"] * 1.5, toy_dataset, "target", dataset_name="toy", model_version="v2"
        )
        with pytest.raises(AssertionError):
            assert_no_regression(history, bad_result)

    def test_history_persists_across_instances(self, tmp_path, toy_dataset):
        history_path = str(tmp_path / "history.json")
        result = run_benchmark(
            lambda X: X["feature_a"] * 2, toy_dataset, "target", dataset_name="toy", model_version="v1"
        )
        BenchmarkHistory(history_path).append(result)

        reloaded = BenchmarkHistory(history_path)
        latest = reloaded.latest_for("toy")
        assert latest is not None
        assert latest.model_version == "v1"
