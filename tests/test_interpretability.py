import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from ipind2.interpretability import (
    AttentionExtractor,
    SHAPExplainer,
    batch_ensemble_confidence,
    ensemble_confidence,
    explain_multi_output,
)


class TestEnsembleConfidence:
    def test_zero_variance_gives_full_confidence(self):
        score = ensemble_confidence([5.0, 5.0, 5.0], scale=1.0)
        assert score.mean == pytest.approx(5.0)
        assert score.std == pytest.approx(0.0)
        assert score.confidence == pytest.approx(1.0)

    def test_high_variance_lowers_confidence(self):
        low_var = ensemble_confidence([5.0, 5.1, 4.9], scale=1.0)
        high_var = ensemble_confidence([1.0, 5.0, 9.0], scale=1.0)
        assert high_var.confidence < low_var.confidence

    def test_requires_at_least_two_predictions(self):
        with pytest.raises(ValueError):
            ensemble_confidence([5.0], scale=1.0)

    def test_is_high_uncertainty_threshold(self):
        score = ensemble_confidence([1.0, 9.0], scale=1.0)
        assert score.is_high_uncertainty(threshold=0.99)
        assert not score.is_high_uncertainty(threshold=0.0)

    def test_batch_matches_scalar(self):
        predictions = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])  # (n_models=3, n_samples=2)
        batch = batch_ensemble_confidence(predictions, scale=2.0)
        assert len(batch) == 2
        assert batch[0].mean == pytest.approx(1.0)
        assert batch[1].mean == pytest.approx(5.0)


@pytest.fixture
def linear_regression_setup():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "mol_weight": rng.uniform(300, 800, size=40),
            "logP": rng.uniform(0, 5, size=40),
            "tpsa": rng.uniform(20, 150, size=40),
        }
    )
    # هدف کاملاً خطی و وابسته به mol_weight تا اهمیت ویژگی قابل پیش‌بینی باشد
    y = 0.1 * X["mol_weight"] + rng.normal(0, 0.01, size=40)

    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(X, y)

    def predict_fn(data):
        return model.predict(pd.DataFrame(data, columns=X.columns))

    return X, predict_fn


class TestSHAPExplainer:
    def test_explain_returns_one_result_per_row(self, linear_regression_setup):
        X, predict_fn = linear_regression_setup
        explainer = SHAPExplainer(predict_fn, background_data=X.iloc[:20], algorithm="permutation")
        results = explainer.explain(X.iloc[20:25])
        assert len(results) == 5
        for result in results:
            assert len(result.attributions) == 3

    def test_prediction_matches_model_output(self, linear_regression_setup):
        X, predict_fn = linear_regression_setup
        explainer = SHAPExplainer(predict_fn, background_data=X.iloc[:20], algorithm="permutation")
        sample = X.iloc[20:22]
        results = explainer.explain(sample)
        expected = predict_fn(sample)
        for result, exp in zip(results, expected):
            assert result.prediction == pytest.approx(exp, rel=1e-3)

    def test_mol_weight_is_top_feature(self, linear_regression_setup):
        X, predict_fn = linear_regression_setup
        explainer = SHAPExplainer(predict_fn, background_data=X.iloc[:20], algorithm="permutation")
        results = explainer.explain(X.iloc[20:25])
        for result in results:
            top = result.top_features(n=1)[0]
            assert top.feature == "mol_weight"

    def test_rejects_empty_background(self, linear_regression_setup):
        X, predict_fn = linear_regression_setup
        with pytest.raises(ValueError):
            SHAPExplainer(predict_fn, background_data=X.iloc[:0])


def test_explain_multi_output(linear_regression_setup):
    X, predict_fn = linear_regression_setup
    results = explain_multi_output(
        predict_fn_per_target={"size_nm": predict_fn, "zeta_mV": predict_fn},
        background_data=X.iloc[:20],
        X=X.iloc[20:23],
        algorithm="permutation",
    )
    assert set(results.keys()) == {"size_nm", "zeta_mV"}
    assert len(results["size_nm"]) == 3


class _ToyAttentionLayer(nn.Module):
    """لایه ساختگی که یک تاپل (خروجی، وزن‌های attention) برمی‌گرداند، مثل nn.MultiheadAttention."""

    def forward(self, x):
        weights = torch.softmax(x.sum(dim=-1, keepdim=True).expand(-1, x.shape[1]), dim=-1)
        return x, weights


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_layer = _ToyAttentionLayer()

    def forward(self, x):
        return self.attn_layer(x)


class TestAttentionExtractor:
    def test_extracts_attention_weights(self):
        model = _ToyModel()
        x = torch.randn(2, 4)
        with AttentionExtractor(model, ["attn_layer"]) as extractor:
            captured = extractor.extract(x)
        assert "attn_layer" in captured
        assert captured["attn_layer"].shape == (2, 4)

    def test_unknown_layer_raises(self):
        model = _ToyModel()
        with pytest.raises(ValueError):
            AttentionExtractor(model, ["does_not_exist"])

    def test_close_removes_hooks(self):
        model = _ToyModel()
        extractor = AttentionExtractor(model, ["attn_layer"])
        extractor.close()
        assert extractor._handles == []
