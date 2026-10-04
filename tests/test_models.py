"""تست‌های واحد ۲ و ۳ (GNN، Transformer+GNN)، پیش‌بین ensemble و آموزش (FR-02، FR-03، FR-06)."""

import numpy as np
import pytest
import torch

from ipind2.biological import BIO_TARGET_COLUMNS, BiologicalPredictor, GraphTransformer
from ipind2.featurization import EXTENDED_DIM, NODE_FEATURE_DIM, batch_graphs, smiles_to_graph
from ipind2.nn import AttentionReadout, DenseMessagePassing, EarlyStopping, TargetScaler, masked_softmax
from ipind2.nn.training import masked_huber_loss, split_indices
from ipind2.physicochemical import PHYSICO_TARGET_COLUMNS, MultiTaskGNN, PhysicochemicalPredictor


class TestLayers:
    def test_masked_softmax_ignores_padding(self):
        scores = torch.tensor([[1.0, 2.0, 100.0]])
        weights = masked_softmax(scores, torch.tensor([[1.0, 1.0, 0.0]]))
        assert weights[0, 2] == 0 and weights.sum().item() == pytest.approx(1.0)

    def test_masked_softmax_all_padding_row_is_zero(self):
        weights = masked_softmax(torch.zeros(1, 3), torch.zeros(1, 3))
        assert torch.all(weights == 0)

    def test_message_passing_zeroes_padding(self):
        nodes, adjacency, mask = (torch.from_numpy(a) for a in batch_graphs([smiles_to_graph("CCO"), smiles_to_graph("CCCCCC")]))
        out = DenseMessagePassing(NODE_FEATURE_DIM, 16)(nodes, adjacency, mask)
        assert torch.all(out[0, 3:] == 0)  # اتم‌های padding نمونه کوتاه‌تر

    def test_readout_weights_sum_to_one_over_real_atoms(self):
        nodes, adjacency, mask = (torch.from_numpy(a) for a in batch_graphs([smiles_to_graph("CCO"), smiles_to_graph("CCCCCC")]))
        hidden = DenseMessagePassing(NODE_FEATURE_DIM, 16)(nodes, adjacency, mask)
        _, weights = AttentionReadout(16)(hidden, mask)
        assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5) and torch.all(weights[0, 3:] == 0)

    def test_attention_extractor_integration(self):
        """AttentionReadout با AttentionExtractor واحد ۷ (hook) سازگار است."""
        from ipind2.interpretability import AttentionExtractor

        model = MultiTaskGNN(n_tasks=2, hidden_dim=16)
        batch = batch_graphs([smiles_to_graph("CCCCO")])
        tensors = [torch.from_numpy(a) for a in batch] + [torch.zeros(1, EXTENDED_DIM)]
        with AttentionExtractor(model, ["readout"]) as extractor:
            captured = extractor.extract(*tensors)
        assert captured["readout"].shape == (1, batch[0].shape[1])


class TestTrainingUtilities:
    def test_scaler_round_trip_and_constant_column(self):
        data = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]], dtype=np.float32)
        scaler = TargetScaler.fit(data)
        assert np.allclose(scaler.inverse_transform(scaler.transform(data)), data)
        assert np.all(np.isfinite(scaler.transform(data)))  # ستون ثابت ⇒ تقسیم بر صفر نشود
        restored = TargetScaler.from_state_dict(scaler.state_dict())
        assert np.allclose(restored.mean, scaler.mean)

    def test_early_stopping_and_restore(self):
        model = torch.nn.Linear(1, 1)
        stopper = EarlyStopping(patience=2, min_delta=0.0)
        assert not stopper.step(1.0, 0, model)
        best = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            model.weight.add_(10.0)
        assert not stopper.step(2.0, 1, model) and stopper.step(3.0, 2, model)
        stopper.restore(model)
        assert torch.equal(model.weight, best["weight"]) and stopper.best_epoch == 0

    def test_masked_loss_ignores_nan_targets(self):
        predictions = torch.tensor([[1.0, 100.0]])
        assert masked_huber_loss(predictions, torch.tensor([[1.0, float("nan")]])).item() == 0.0
        assert masked_huber_loss(predictions, torch.tensor([[float("nan")] * 2])).item() == 0.0  # همه NaN ⇒ گرادیان صفر
        assert masked_huber_loss(predictions, torch.tensor([[0.0, 100.0]])).item() > 0

    def test_split_is_disjoint_and_complete(self):
        train, test = split_indices(50, 0.2, seed=1)
        assert set(train).isdisjoint(test) and len(train) + len(test) == 50


@pytest.fixture(scope="module")
def physico_model(request):
    from ipind2.data_generation.synthetic_data_generator import SyntheticDataGenerator

    data = SyntheticDataGenerator(41).generate_dataset(1100, include_pareto_labels=False)
    train, test = data.iloc[:900], data.iloc[900:]
    model = PhysicochemicalPredictor(n_ensemble=2, hidden_dim=48)
    model.fit(train.smiles.tolist(), train[list(PHYSICO_TARGET_COLUMNS)].to_numpy(), epochs=14, seed=3)
    return model, train, test


class TestPhysicochemicalPredictor:
    def test_learns_structure_property_relationship(self, physico_model):
        """R² معنادار روی داده نادیده — اگر مدل ساختار را نبیند، R² ≈ ۰ می‌شود."""
        model, _, test = physico_model
        report = model.evaluate(test.smiles.tolist(), test[list(PHYSICO_TARGET_COLUMNS)].to_numpy())
        assert report.loc["phys_size_nm", "r2"] > 0.5
        assert report.loc["phys_zeta_potential_mV", "r2"] > 0.5

    def test_predicts_seven_targets(self, physico_model):
        model, _, test = physico_model
        frame = model.predict(test.smiles.tolist()[:5])
        assert list(frame.columns) == list(PHYSICO_TARGET_COLUMNS) and len(PHYSICO_TARGET_COLUMNS) == 7
        assert np.all(np.isfinite(frame.to_numpy()))

    def test_uncertainty_nonnegative_and_member_shape(self, physico_model):
        model, _, test = physico_model
        mean, std, kept = model.predict_with_uncertainty(test.smiles.tolist()[:6])
        assert (std.to_numpy() >= 0).all() and kept == list(range(6))
        members, _ = model.member_predictions(test.smiles.tolist()[:6])
        assert members.shape == (2, 6, 7)

    def test_invalid_smiles_dropped_with_index_alignment(self, physico_model):
        model, _, test = physico_model
        smiles = [test.smiles.iloc[0], "bad(((", test.smiles.iloc[1]]
        mean, _, kept = model.predict_with_uncertainty(smiles)
        assert kept == [0, 2] and list(mean.index) == [0, 2]
        assert model.predict(["bad((("]).empty

    def test_prediction_is_deterministic_in_eval_mode(self, physico_model):
        model, _, test = physico_model
        smiles = test.smiles.tolist()[:4]
        assert np.allclose(model.predict(smiles).to_numpy(), model.predict(smiles).to_numpy())

    def test_batch_composition_does_not_change_prediction(self, physico_model):
        """padding/ماسک درست ⇒ پیش‌بینی یک مولکول به همراهان دسته بستگی ندارد."""
        model, _, test = physico_model
        a, b = test.smiles.iloc[0], test.smiles.iloc[1]
        alone = model.predict([a]).iloc[0].to_numpy()
        batched = model.predict([a, b, "C" * 80]).iloc[0].to_numpy()
        assert np.allclose(alone, batched, rtol=1e-3, atol=1e-3)

    def test_atom_attention_is_distribution_over_atoms(self, physico_model):
        model, _, test = physico_model
        smiles = test.smiles.iloc[0]
        attention = model.atom_attention(smiles)
        assert attention.shape[0] == smiles_to_graph(smiles, 96).n_atoms
        assert attention.sum() == pytest.approx(1.0, abs=1e-4) and (attention >= 0).all()
        assert model.atom_attention("bad(((") is None

    def test_save_load_round_trip_preserves_predictions(self, physico_model, tmp_path):
        model, _, test = physico_model
        model.save(str(tmp_path / "m"))
        loaded = PhysicochemicalPredictor.load(str(tmp_path / "m"))
        assert type(loaded) is PhysicochemicalPredictor
        smiles = test.smiles.tolist()[:6]
        assert np.allclose(model.predict(smiles).to_numpy(), loaded.predict(smiles).to_numpy(), atol=1e-4)

    def test_fine_tune_adapts_to_partial_labels_without_forgetting(self, physico_model):
        import copy

        model, train, test = physico_model
        tuned = copy.deepcopy(model)
        rows = test.iloc[:30]
        targets = rows[list(PHYSICO_TARGET_COLUMNS)].to_numpy().copy()
        targets[:, 2:] = np.nan  # فقط اندازه و زتا اندازه‌گیری شده
        shifted = targets.copy()
        shifted[:, 0] += 25.0  # اندازه‌گیری آزمایشگاهی سیستماتیک ۲۵ nm بزرگ‌تر
        before = model.predict(rows.smiles.tolist())["phys_size_nm"].mean()
        replay = train.iloc[:60]
        merged_smiles = rows.smiles.tolist() + replay.smiles.tolist()
        merged = np.vstack([shifted, replay[list(PHYSICO_TARGET_COLUMNS)].to_numpy()])
        tuned.fine_tune(merged_smiles, merged, epochs=25, learning_rate=1e-3)
        after = tuned.predict(rows.smiles.tolist())["phys_size_nm"].mean()
        assert after > before + 5.0, "fine-tuning باید پیش‌بینی را به‌سمت داده جدید ببرد"
        assert np.all(np.isfinite(tuned.predict(test.smiles.tolist()[30:40]).to_numpy()))

    def test_untrained_and_bad_inputs(self):
        with pytest.raises(RuntimeError):
            PhysicochemicalPredictor().predict(["CCO"])
        model = PhysicochemicalPredictor(n_ensemble=1)
        with pytest.raises(ValueError):
            model.fit(["CCO"] * 20, np.zeros((19, 7)))  # طول ناهمخوان
        with pytest.raises(ValueError):
            model.fit(["CCO"] * 20, np.zeros((20, 3)))  # تعداد هدف ناهمخوان
        with pytest.raises(ValueError):
            model.fit(["bad((("] * 20, np.zeros((20, 7)))  # SMILES معتبر کافی نیست

    def test_inference_speed_supports_nfr05(self, physico_model):
        """NFR-05: <۱۰۰ ms برای هر ساختار (ensemble ۲ عضوی روی CPU)."""
        import time

        model, _, test = physico_model
        smiles = test.smiles.tolist()[:64]
        model.predict(smiles[:4])  # گرم‌کردن
        started = time.perf_counter()
        model.predict(smiles)
        assert (time.perf_counter() - started) / len(smiles) < 0.1


class TestBiologicalPredictor:
    @pytest.fixture(scope="class")
    def bio_model(self):
        from ipind2.data_generation.synthetic_data_generator import SyntheticDataGenerator

        data = SyntheticDataGenerator(43).generate_dataset(900, include_pareto_labels=False)
        model = BiologicalPredictor(n_ensemble=2, hidden_dim=32)
        model.fit(data.smiles.iloc[:750].tolist(), data.iloc[:750][list(BIO_TARGET_COLUMNS)].to_numpy(), epochs=10, seed=1)
        return model, data.iloc[750:]

    def test_three_cell_lines_and_four_other_targets(self):
        assert len([c for c in BIO_TARGET_COLUMNS if "ic50" in c]) == 3  # FR-03: ≥۳ رده سلولی
        assert len(BIO_TARGET_COLUMNS) == 7

    def test_learns_signal(self, bio_model):
        model, test = bio_model
        report = model.evaluate(test.smiles.tolist(), test[list(BIO_TARGET_COLUMNS)].to_numpy())
        assert report.loc["bio_cytotoxicity_ic50_ug_ml", "r2"] > 0.4

    def test_transformer_masks_padding(self):
        model = GraphTransformer(n_tasks=2, hidden_dim=16, n_heads=2).eval()
        short = batch_graphs([smiles_to_graph("CCO")])
        padded = batch_graphs([smiles_to_graph("CCO"), smiles_to_graph("CCCCCCCCCC")])
        globals_ = torch.zeros(1, EXTENDED_DIM)
        with torch.no_grad():
            alone = model(*(torch.from_numpy(a) for a in short), globals_)
            together = model(*(torch.from_numpy(a) for a in padded), torch.zeros(2, EXTENDED_DIM))[:1]
        assert torch.allclose(alone, together, atol=1e-4)

    def test_save_load_uses_correct_architecture(self, bio_model, tmp_path):
        model, test = bio_model
        model.save(str(tmp_path / "b"))
        loaded = BiologicalPredictor.load(str(tmp_path / "b"))
        assert loaded.architecture == "graph_transformer"
        smiles = test.smiles.tolist()[:5]
        assert np.allclose(model.predict(smiles).to_numpy(), loaded.predict(smiles).to_numpy(), atol=1e-4)


class TestConcurrency:
    def test_parallel_predictions_match_serial(self, physico_model):
        """باگ بالقوه: GraphEncoder حالت per-call دارد؛ قفل باید نتیجه هم‌زمان را درست نگه دارد."""
        from concurrent.futures import ThreadPoolExecutor

        model, _, test = physico_model
        batches = [test.smiles.tolist()[i : i + 5] for i in range(0, 30, 5)]
        serial = [model.predict(b).to_numpy() for b in batches]
        with ThreadPoolExecutor(max_workers=4) as pool:
            parallel = list(pool.map(lambda b: model.predict(b).to_numpy(), batches * 3))
        for index, result in enumerate(parallel):
            assert np.allclose(result, serial[index % len(batches)], atol=1e-4)
