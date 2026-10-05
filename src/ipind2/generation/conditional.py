"""
Conditional structure generator (ensemble of CVAE and CGAN) — Unit 1 (FR-01).

Flow: models are trained on (molecule features | target features + scaffold type); at generation time,
for a "condition" (e.g., size 100 nm, positive zeta, lipid) a target feature vector is sampled
and mapped to a real SMILES by nearest neighbor in the *pool of valid structures* (built from the templates of
``building_blocks``). Therefore:

* the chemical validity of structures is 100% (and is re-checked with RDKit),
* the deep models determine *which* region of the structural space is desirable,
* outputs are traceable (template + building blocks in ``GeneratedStructure.slots``).

See docs/SRS.md §4.1 (FR-01) and §4.8 (input from the natural-language interface).
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

# joblib warns on Windows without ``wmic``; we give the core count explicitly.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.neighbors import NearestNeighbors

from ..featurization import EXTENDED_DIM, extended_matrix, is_valid_smiles
from .building_blocks import SCAFFOLD_TYPES
from .library import CombinatorialLibrary, GeneratedStructure, LibraryStats
from .models import ConditionalGAN, ConditionalVAE

CONDITION_COLUMNS: Tuple[str, ...] = (
    "phys_size_nm",
    "phys_zeta_potential_mV",
    "phys_drug_loading_efficiency_percent",
    "bio_cytotoxicity_ic50_ug_ml",
)
CONDITION_KEYS: Tuple[str, ...] = ("size_nm", "zeta_mV", "loading_efficiency", "ic50")


@dataclass
class GenerationCondition:
    """
    Generation condition. A ``None`` value means "don't care" and is replaced by the training median.

    If ``scaffold_type`` is ``None``, the scaffold type is sampled uniformly.
    """

    scaffold_type: Optional[str] = None
    size_nm: Optional[float] = None
    zeta_mV: Optional[float] = None
    loading_efficiency: Optional[float] = None
    ic50: Optional[float] = None

    def values(self) -> Tuple[Optional[float], ...]:
        return (self.size_nm, self.zeta_mV, self.loading_efficiency, self.ic50)

    @classmethod
    def from_target_parameters(cls, params) -> "GenerationCondition":
        """
        Convert the natural-language interface output (``nlp_interface.TargetParameters``) to a generation condition.

        The size range is mapped to its midpoint; the minimum loading efficiency is taken with a small
        margin above the minimum.
        """
        size = None
        if params.size_range_nm is not None:
            low, high = params.size_range_nm
            size = (low + high) / 2.0 if low > 0 else high * 0.75
        loading = (
            params.min_loading_efficiency + 5.0
            if params.min_loading_efficiency is not None
            else None
        )
        return cls(scaffold_type=params.scaffold_type, size_nm=size, loading_efficiency=loading)


class ConditionalStructureGenerator:
    """Conditional CVAE+CGAN ensemble with retrieval from the pool of valid structures."""

    def __init__(self, latent_dim: int = 8, hidden: int = 128, seed: int = 0):
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.seed = seed
        self.vae: Optional[ConditionalVAE] = None
        self.gan: Optional[ConditionalGAN] = None
        self.feature_mean = np.zeros(EXTENDED_DIM, dtype=np.float32)
        self.feature_std = np.ones(EXTENDED_DIM, dtype=np.float32)
        self.cond_mean = np.zeros(len(CONDITION_KEYS), dtype=np.float32)
        self.cond_std = np.ones(len(CONDITION_KEYS), dtype=np.float32)
        self.cond_median: Dict[str, np.ndarray] = {}
        self.pool: List[GeneratedStructure] = []
        self._pool_features: Optional[np.ndarray] = None
        self._indexes: Dict[str, Tuple[NearestNeighbors, np.ndarray]] = {}
        self.history: Dict[str, List[float]] = {"vae_loss": [], "gan_d_loss": [], "gan_g_loss": []}

    # ------------------------------------------------------------------
    @property
    def cond_dim(self) -> int:
        return len(SCAFFOLD_TYPES) + len(CONDITION_KEYS)

    def _encode_condition(self, scaffold_type: str, values: Sequence[Optional[float]]) -> np.ndarray:
        one_hot = np.zeros(len(SCAFFOLD_TYPES), dtype=np.float32)
        one_hot[SCAFFOLD_TYPES.index(scaffold_type)] = 1.0
        filled = np.array(
            [
                v if v is not None else self.cond_median[scaffold_type][i]
                for i, v in enumerate(values)
            ],
            dtype=np.float32,
        )
        return np.concatenate([one_hot, (filled - self.cond_mean) / self.cond_std])

    # ------------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        pool_size: int = 20000,
        epochs: int = 120,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        verbose: bool = False,
    ) -> "ConditionalStructureGenerator":
        """
        Train on a dataframe with columns ``smiles``, ``scaffold_type`` and ``CONDITION_COLUMNS``
        and build the retrieval pool.
        """
        missing = [c for c in ("smiles", "scaffold_type", *CONDITION_COLUMNS) if c not in df.columns]
        if missing:
            raise ValueError(f"Required columns are missing from the dataframe: {missing}")

        features, kept = extended_matrix(df["smiles"].tolist())
        if len(kept) < 50:
            raise ValueError("At least 50 valid structures are required to train the generator")
        rows = df.iloc[kept].reset_index(drop=True)
        torch.manual_seed(self.seed)

        self.feature_mean = features.mean(axis=0)
        std = features.std(axis=0)
        self.feature_std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        x = (features - self.feature_mean) / self.feature_std

        cond_raw = rows[list(CONDITION_COLUMNS)].to_numpy(dtype=np.float32)
        self.cond_mean = cond_raw.mean(axis=0)
        cstd = cond_raw.std(axis=0)
        self.cond_std = np.where(cstd < 1e-6, 1.0, cstd).astype(np.float32)
        self.cond_median = {
            t: np.median(cond_raw[rows["scaffold_type"].to_numpy() == t], axis=0)
            if (rows["scaffold_type"] == t).any()
            else np.median(cond_raw, axis=0)
            for t in SCAFFOLD_TYPES
        }
        cond = np.stack(
            [
                self._encode_condition(t, row)
                for t, row in zip(rows["scaffold_type"].tolist(), cond_raw.tolist())
            ]
        )

        self.vae = ConditionalVAE(EXTENDED_DIM, self.cond_dim, self.latent_dim, self.hidden)
        self.gan = ConditionalGAN(EXTENDED_DIM, self.cond_dim, self.latent_dim, self.hidden)
        self._train(torch.from_numpy(x), torch.from_numpy(cond), epochs, batch_size, learning_rate, verbose)
        self._build_pool(pool_size)
        return self

    def _train(self, x, cond, epochs, batch_size, learning_rate, verbose):
        n = x.shape[0]
        vae_opt = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        g_opt = torch.optim.Adam(self.gan.generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        d_opt = torch.optim.Adam(self.gan.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        bce = torch.nn.BCEWithLogitsLoss()
        self.history = {"vae_loss": [], "gan_d_loss": [], "gan_g_loss": []}

        for epoch in range(epochs):
            order = torch.randperm(n)
            vae_losses, d_losses, g_losses = [], [], []
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                xb, cb = x[idx], cond[idx]
                if xb.shape[0] < 4:
                    continue

                vae_opt.zero_grad()
                recon, mu, log_var = self.vae(xb, cb)
                loss = ConditionalVAE.loss(recon, xb, mu, log_var)
                loss.backward()
                vae_opt.step()
                vae_losses.append(float(loss.item()))

                real_label = torch.ones(xb.shape[0])
                fake_label = torch.zeros(xb.shape[0])
                z = torch.randn(xb.shape[0], self.latent_dim)
                fake = self.gan.generate(z, cb)
                d_opt.zero_grad()
                d_loss = bce(self.gan.discriminate(xb, cb), real_label) + bce(
                    self.gan.discriminate(fake.detach(), cb), fake_label
                )
                d_loss.backward()
                d_opt.step()
                d_losses.append(float(d_loss.item()))

                g_opt.zero_grad()
                g_loss = bce(self.gan.discriminate(fake, cb), real_label)
                g_loss.backward()
                g_opt.step()
                g_losses.append(float(g_loss.item()))

            self.history["vae_loss"].append(float(np.mean(vae_losses)))
            self.history["gan_d_loss"].append(float(np.mean(d_losses)))
            self.history["gan_g_loss"].append(float(np.mean(g_losses)))
            if verbose and epoch % 10 == 0:
                print(f"epoch {epoch}: vae={self.history['vae_loss'][-1]:.3f}")

    def _build_pool(self, pool_size: int) -> None:
        library = CombinatorialLibrary(seed=self.seed + 1)
        structures, _ = library.generate(pool_size)
        features, kept = extended_matrix([s.smiles for s in structures])
        self.pool = [structures[i] for i in kept]
        self._pool_features = (features - self.feature_mean) / self.feature_std
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self._indexes = {}
        types = np.array([s.scaffold_type for s in self.pool])
        for scaffold_type in SCAFFOLD_TYPES:
            member_idx = np.where(types == scaffold_type)[0]
            if member_idx.size == 0:
                continue
            nn_index = NearestNeighbors(n_neighbors=min(64, member_idx.size)).fit(
                self._pool_features[member_idx]
            )
            self._indexes[scaffold_type] = (nn_index, member_idx)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _decode_targets(self, cond: np.ndarray, use: str) -> np.ndarray:
        c = torch.from_numpy(cond.astype(np.float32))
        half = c.shape[0] // 2
        if use == "vae":
            out = self.vae.sample(c)
        elif use == "gan":
            out = self.gan.sample(c)
        else:  # ensemble: half of the samples from each model
            out = torch.cat([self.vae.sample(c[:half]), self.gan.sample(c[half:])], dim=0)
        return out.numpy()

    def generate(
        self,
        n: int,
        condition: Optional[GenerationCondition] = None,
        model: str = "ensemble",
        neighbours: int = 8,
        seed: Optional[int] = None,
    ) -> Tuple[List[GeneratedStructure], LibraryStats]:
        """
        Generate ``n`` unique, valid structures for the given condition.

        Args:
            model: ``'vae'`` | ``'gan'`` | ``'ensemble'``.
            neighbours: among the k nearest neighbors of each target vector, one is chosen at random
                (for diversity; ``1`` means the most deterministic mapping).
        """
        if self.vae is None or not self._indexes:
            raise RuntimeError("Generator is not trained; call fit() or load() first")
        if n <= 0:
            raise ValueError("n must be positive")
        if model not in ("vae", "gan", "ensemble"):
            raise ValueError("model must be one of vae/gan/ensemble")

        condition = condition or GenerationCondition()
        rng = np.random.default_rng(self.seed if seed is None else seed)
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
        started = time.perf_counter()

        available = [t for t in SCAFFOLD_TYPES if t in self._indexes]
        if condition.scaffold_type is not None:
            if condition.scaffold_type not in available:
                raise ValueError(f"Invalid scaffold type or no pool: {condition.scaffold_type!r}")
            available = [condition.scaffold_type]

        structures: List[GeneratedStructure] = []
        seen = set()
        proposed = 0
        max_rounds = 60
        batch = max(64, n)
        k_effective = max(1, neighbours)

        for _ in range(max_rounds):
            if len(structures) >= n:
                break
            added_this_round = 0
            types = rng.choice(available, size=batch)
            cond = np.stack([self._encode_condition(t, condition.values()) for t in types])
            targets = self._decode_targets(cond, model)
            for scaffold_type in available:
                rows = np.where(types == scaffold_type)[0]
                if rows.size == 0:
                    continue
                index, member_idx = self._indexes[scaffold_type]
                k = min(k_effective, index.n_neighbors)
                distances, neighbour_ids = index.kneighbors(targets[rows], n_neighbors=k)
                for row, dist, ids in zip(rows, distances, neighbour_ids):
                    pick = int(rng.integers(0, k))
                    proposed += 1
                    structure = self.pool[int(member_idx[ids[pick]])]
                    if structure.smiles in seen:
                        continue
                    seen.add(structure.smiles)
                    added_this_round += 1
                    structures.append(
                        GeneratedStructure(
                            smiles=structure.smiles,
                            scaffold_type=structure.scaffold_type,
                            template=structure.template,
                            slots=dict(structure.slots),
                            condition_distance=float(dist[pick]),
                        )
                    )
                    if len(structures) >= n:
                        break
                if len(structures) >= n:
                    break
            # Narrow condition ⇒ target vectors cluster and duplicate neighbors are returned; instead of
            # silently under-delivering, we widen the search radius (diversity versus condition precision).
            if added_this_round < max(1, batch // 20):
                k_effective = min(k_effective * 2, 64)

        structures = structures[:n]
        valid = sum(1 for s in structures if is_valid_smiles(s.smiles))
        # Validity is measured on the *final output* (not on the number of raw retrievals).
        stats = LibraryStats(
            requested=n,
            proposed=len(structures),
            valid=valid,
            unique=len(structures),
            elapsed_seconds=time.perf_counter() - started,
            attempts=proposed,
        )
        return structures, stats

    # ------------------------------------------------------------------
    def save(self, directory: str) -> None:
        if self.vae is None:
            raise RuntimeError("Generator is not trained")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "latent_dim": self.latent_dim,
            "hidden": self.hidden,
            "seed": self.seed,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "cond_mean": self.cond_mean.tolist(),
            "cond_std": self.cond_std.tolist(),
            "cond_median": {k: v.tolist() for k, v in self.cond_median.items()},
            "pool": [
                {"smiles": s.smiles, "scaffold_type": s.scaffold_type, "template": s.template, "slots": s.slots}
                for s in self.pool
            ],
        }
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        np.save(path / "pool_features.npy", self._pool_features)
        torch.save({"vae": self.vae.state_dict(), "gan": self.gan.state_dict()}, path / "weights.pt")

    @classmethod
    def load(cls, directory: str) -> "ConditionalStructureGenerator":
        path = Path(directory)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        gen = cls(latent_dim=meta["latent_dim"], hidden=meta["hidden"], seed=meta["seed"])
        gen.feature_mean = np.asarray(meta["feature_mean"], dtype=np.float32)
        gen.feature_std = np.asarray(meta["feature_std"], dtype=np.float32)
        gen.cond_mean = np.asarray(meta["cond_mean"], dtype=np.float32)
        gen.cond_std = np.asarray(meta["cond_std"], dtype=np.float32)
        gen.cond_median = {k: np.asarray(v, dtype=np.float32) for k, v in meta["cond_median"].items()}
        gen.pool = [GeneratedStructure(**row) for row in meta["pool"]]
        gen._pool_features = np.load(path / "pool_features.npy")
        gen.vae = ConditionalVAE(EXTENDED_DIM, gen.cond_dim, gen.latent_dim, gen.hidden)
        gen.gan = ConditionalGAN(EXTENDED_DIM, gen.cond_dim, gen.latent_dim, gen.hidden)
        weights = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        gen.vae.load_state_dict(weights["vae"])
        gen.gan.load_state_dict(weights["gan"])
        gen._rebuild_indexes()
        return gen
