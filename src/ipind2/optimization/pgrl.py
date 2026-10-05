"""
Multi-objective optimization with Pareto-Guided Reinforcement Learning — FR-04.

Policy: a categorical distribution over the "structural template" and then over the block of each slot of that template
(action space = the structural space ``generation.building_blocks``). In each iteration a batch of structures is sampled from the
policy, evaluated with the objective function, and the policy is updated with REINFORCE.

The "Pareto-guided" reward is a combination of two components:
1. Rank reward: ``1/Pareto rank`` among the batch + archive (structures on the front get the highest reward);
2. Chebyshev scalarization with random Dirichlet weights in each iteration, so the policy does not converge to a single front
   point and the whole front is covered.

Convergence criterion per SRS §4.4: change in archive hypervolume < 1% over 100 iterations.

See docs/SRS.md §4.4 (FR-04), NFR-06.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..generation.building_blocks import StructureTemplate, templates_for
from ..featurization import canonical_smiles, is_valid_smiles
from .objectives import OBJECTIVE_COLUMNS, OBJECTIVE_NAMES, objective_matrix, size_fit_score
from .pareto import crowding_distance, hypervolume, pareto_ranks

# Objective function: list of SMILES -> dataframe with OBJECTIVE_COLUMNS (in input order)
ObjectiveFn = Callable[[Sequence[str]], pd.DataFrame]


@dataclass
class Constraints:
    """Hard design constraints; a structure violating a constraint is excluded from the front archive."""

    size_range_nm: Optional[Tuple[float, float]] = None
    min_loading_efficiency: Optional[float] = None
    min_ic50: Optional[float] = None

    @classmethod
    def from_target_parameters(cls, params) -> "Constraints":
        """Build constraints from the natural-language interface output (``nlp_interface.TargetParameters``)."""
        return cls(
            size_range_nm=params.size_range_nm,
            min_loading_efficiency=params.min_loading_efficiency,
        )

    def feasible(self, frame: pd.DataFrame) -> np.ndarray:
        ok = np.ones(len(frame), dtype=bool)
        if self.size_range_nm is not None:
            low, high = self.size_range_nm
            ok &= (frame[OBJECTIVE_COLUMNS["size_fit"]] >= low).to_numpy()
            ok &= (frame[OBJECTIVE_COLUMNS["size_fit"]] <= high).to_numpy()
        if self.min_loading_efficiency is not None:
            ok &= (
                frame[OBJECTIVE_COLUMNS["loading_efficiency"]] >= self.min_loading_efficiency
            ).to_numpy()
        if self.min_ic50 is not None:
            ok &= (frame[OBJECTIVE_COLUMNS["safety_ic50"]] >= self.min_ic50).to_numpy()
        return ok


@dataclass
class OptimizationResult:
    """Result of one optimization run."""

    candidates: pd.DataFrame  # Pareto front (smiles + objective columns + ...)
    archive_size: int
    iterations: int
    converged: bool
    hypervolume_history: List[float] = field(default_factory=list)
    evaluations: int = 0
    elapsed_seconds: float = 0.0

    def top(self, n: int) -> pd.DataFrame:
        return self.candidates.head(n)


class _FactorizedPolicy:
    """Categorical policy: first layer template, second layer the block of each slot; updated with REINFORCE."""

    def __init__(self, templates: Sequence[StructureTemplate]):
        self.templates = list(templates)
        self.template_logits = np.zeros(len(self.templates))
        self.slot_logits: List[Dict[str, np.ndarray]] = [
            {slot: np.zeros(len(options)) for slot, options in t.slots} for t in self.templates
        ]
        # Adam state for stable updates
        self._m: Dict[Tuple, np.ndarray] = {}
        self._v: Dict[Tuple, np.ndarray] = {}
        self._t = 0

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    def sample(self, rng: np.random.Generator) -> Tuple[int, Dict[str, int]]:
        probs = self._softmax(self.template_logits)
        t = int(rng.choice(len(self.templates), p=probs))
        choice = {}
        for slot, options in self.templates[t].slots:
            p = self._softmax(self.slot_logits[t][slot])
            choice[slot] = int(rng.choice(len(options), p=p))
        return t, choice

    def smiles(self, t: int, choice: Dict[str, int]) -> str:
        template = self.templates[t]
        options = template.slot_options()
        return template.build({slot: options[slot][idx] for slot, idx in choice.items()})

    def entropy(self) -> float:
        probs = self._softmax(self.template_logits)
        return float(-(probs * np.log(probs + 1e-12)).sum())

    def _adam(self, key: Tuple, grad: np.ndarray, lr: float) -> np.ndarray:
        m = self._m.get(key, np.zeros_like(grad))
        v = self._v.get(key, np.zeros_like(grad))
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        self._m[key], self._v[key] = m, v
        m_hat = m / (1 - 0.9**self._t)
        v_hat = v / (1 - 0.999**self._t)
        return lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def update(
        self,
        actions: List[Tuple[int, Dict[str, int]]],
        advantages: np.ndarray,
        lr: float,
        entropy_bonus: float,
    ) -> None:
        """REINFORCE gradient step (raising the log-prob of actions with positive advantage)."""
        self._t += 1
        batch = max(len(actions), 1)

        probs = self._softmax(self.template_logits)
        grad = np.zeros_like(self.template_logits)
        for (t, _), adv in zip(actions, advantages):
            one_hot = np.zeros_like(grad)
            one_hot[t] = 1.0
            grad += adv * (one_hot - probs)
        grad /= batch
        grad += entropy_bonus * (-probs * (np.log(probs + 1e-12) + self.entropy()))
        self.template_logits += self._adam(("tpl",), grad, lr)

        for t_index, template in enumerate(self.templates):
            members = [(c, a) for (t, c), a in zip(actions, advantages) if t == t_index]
            if not members:
                continue
            for slot, _ in template.slots:
                logits = self.slot_logits[t_index][slot]
                p = self._softmax(logits)
                slot_grad = np.zeros_like(logits)
                for choice, adv in members:
                    one_hot = np.zeros_like(logits)
                    one_hot[choice[slot]] = 1.0
                    slot_grad += adv * (one_hot - p)
                slot_grad /= batch
                slot_grad += entropy_bonus * (-p * (np.log(p + 1e-12) + float(-(p * np.log(p + 1e-12)).sum())))
                self.slot_logits[t_index][slot] = logits + self._adam(
                    ("slot", t_index, slot), slot_grad, lr
                )


class ParetoGuidedRL:
    """
    PG-RL multi-objective optimizer.

    Args:
        objective_fn: structure evaluator (Unit 2/3 predictors or the oracle).
        scaffold_type: restrict templates to one scaffold class (or ``None`` for all).
        constraints: hard constraints (size, minimum loading, minimum IC50).
        size_range: desired size range for the ``size_fit`` objective.
    """

    def __init__(
        self,
        objective_fn: ObjectiveFn,
        scaffold_type: Optional[str] = None,
        constraints: Optional[Constraints] = None,
        size_range: Tuple[float, float] = (80.0, 120.0),
        batch_size: int = 32,
        learning_rate: float = 0.15,
        entropy_bonus: float = 0.01,
        seed: int = 0,
    ):
        self.objective_fn = objective_fn
        self.constraints = constraints or Constraints()
        self.size_range = (
            self.constraints.size_range_nm if self.constraints.size_range_nm else size_range
        )
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.entropy_bonus = entropy_bonus
        self.rng = np.random.default_rng(seed)
        self.policy = _FactorizedPolicy(templates_for(scaffold_type))
        self._cache: Dict[str, np.ndarray] = {}
        self._frames: Dict[str, Dict[str, float]] = {}
        self._archive: Dict[str, np.ndarray] = {}  # smiles -> objective vector (feasible only)

    # ------------------------------------------------------------------
    def _evaluate(self, smiles_list: Sequence[str]) -> np.ndarray:
        """Objective matrix (n, 5) with cache; invalid SMILES get a very bad score."""
        new = [s for s in dict.fromkeys(smiles_list) if s not in self._cache and is_valid_smiles(s)]
        if new:
            frame = self.objective_fn(new)
            values = objective_matrix(
                {n: frame[c].to_numpy() for n, c in OBJECTIVE_COLUMNS.items()}, self.size_range
            )
            feasible = self.constraints.feasible(frame)
            for smiles, row, ok, (_, raw) in zip(new, values, feasible, frame.iterrows()):
                self._cache[smiles] = row
                self._frames[smiles] = {c: float(raw[c]) for c in OBJECTIVE_COLUMNS.values()}
                if ok:
                    self._archive[smiles] = row
        worst = np.full(len(OBJECTIVE_NAMES), -1e6)
        return np.stack([self._cache.get(s, worst) for s in smiles_list])

    def evaluated_smiles(self) -> List[str]:
        """All structures evaluated so far (for building surrogate interpretation models and analysis)."""
        return list(self._cache)

    def _front(self) -> Tuple[List[str], np.ndarray]:
        if not self._archive:
            return [], np.zeros((0, len(OBJECTIVE_NAMES)))
        smiles = list(self._archive)
        values = np.stack([self._archive[s] for s in smiles])
        mask = pareto_ranks(values) == 1
        return [s for s, m in zip(smiles, mask) if m], values[mask]

    def _reward_scale(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize each objective with the range observed in the cache (for Chebyshev)."""
        pool = np.stack(list(self._cache.values())) if self._cache else values
        return pool.min(axis=0), np.maximum(pool.max(axis=0) - pool.min(axis=0), 1e-9)

    # ------------------------------------------------------------------
    def optimize(
        self,
        iterations: int = 300,
        patience_window: int = 100,
        tolerance: float = 0.01,
        min_iterations: Optional[int] = None,
        verbose: bool = False,
    ) -> OptimizationResult:
        """
        Run the optimization.

        Stopping: when the relative change in archive hypervolume over the last ``patience_window`` iterations is
        less than ``tolerance`` (SRS: 1% over 100 iterations) or ``iterations`` is exhausted.
        """
        started = time.perf_counter()
        history: List[float] = []
        converged = False
        min_iterations = patience_window + 1 if min_iterations is None else min_iterations
        reference = np.zeros(len(OBJECTIVE_NAMES))
        iteration = 0

        for iteration in range(1, iterations + 1):
            actions = [self.policy.sample(self.rng) for _ in range(self.batch_size)]
            smiles = [self.policy.smiles(t, c) for t, c in actions]
            values = self._evaluate(smiles)

            low, span = self._reward_scale(values)
            normalized = (values - low) / span
            weights = self.rng.dirichlet(np.ones(len(OBJECTIVE_NAMES)))
            chebyshev = (weights * normalized).min(axis=1) * len(OBJECTIVE_NAMES)

            archive_values = (
                np.stack(list(self._archive.values())) if self._archive else np.zeros((0, values.shape[1]))
            )
            combined = np.vstack([values, archive_values])
            ranks = pareto_ranks(combined)[: len(values)]
            rank_reward = 1.0 / ranks

            reward = 0.5 * rank_reward + 0.5 * chebyshev
            advantages = reward - reward.mean()
            scale = advantages.std()
            if scale > 1e-9:
                advantages = advantages / scale
            self.policy.update(actions, advantages, self.learning_rate, self.entropy_bonus)

            front_smiles, front_values = self._front()
            history.append(hypervolume(front_values, reference, n_samples=4000) if len(front_values) else 0.0)
            if verbose and iteration % 20 == 0:
                print(f"iter {iteration}: front={len(front_smiles)} hv={history[-1]:.4f}")

            if iteration >= min_iterations and len(history) > patience_window:
                previous = history[-1 - patience_window]
                if previous > 0 and abs(history[-1] - previous) / previous < tolerance:
                    converged = True
                    break

        return OptimizationResult(
            candidates=self._front_frame(),
            archive_size=len(self._archive),
            iterations=iteration,
            converged=converged,
            hypervolume_history=history,
            evaluations=len(self._cache),
            elapsed_seconds=time.perf_counter() - started,
        )

    def _front_frame(self, max_rows: int = 200) -> pd.DataFrame:
        smiles, values = self._front()
        if not smiles:
            # The columns must match the non-empty case so the caller (pipeline) can
            # concat/filter without errors and impossible constraints are reported as a warning, not a KeyError.
            return pd.DataFrame(
                columns=[
                    "smiles", "canonical", *OBJECTIVE_COLUMNS.values(),
                    *[f"obj_{n}" for n in OBJECTIVE_NAMES], "crowding",
                ]
            )
        crowd = crowding_distance(values)
        order = np.argsort(-np.where(np.isinf(crowd), 1e9, crowd))[:max_rows]
        rows = []
        for i in order:
            row = {"smiles": smiles[i], "canonical": canonical_smiles(smiles[i])}
            row.update(self._frames[smiles[i]])
            row.update({f"obj_{n}": float(v) for n, v in zip(OBJECTIVE_NAMES, values[i])})
            row["crowding"] = float(min(crowd[i], 1e9))
            rows.append(row)
        return pd.DataFrame(rows)

    def select_diverse(self, n: int = 15) -> pd.DataFrame:
        """10–20 diverse candidates on the front (largest crowding distance) — the FR-04 output."""
        return self._front_frame(max_rows=n)


def random_search_front(
    objective_fn: ObjectiveFn,
    n_evaluations: int,
    scaffold_type: Optional[str] = None,
    size_range: Tuple[float, float] = (80.0, 120.0),
    seed: int = 0,
) -> np.ndarray:
    """Baseline: Pareto front of random search with the same evaluation budget (for comparison with PG-RL)."""
    from ..generation.library import CombinatorialLibrary

    library = CombinatorialLibrary(scaffold_type=scaffold_type, seed=seed)
    structures, _ = library.generate(n_evaluations)
    frame = objective_fn([s.smiles for s in structures])
    values = objective_matrix({n: frame[c].to_numpy() for n, c in OBJECTIVE_COLUMNS.items()}, size_range)
    return values[pareto_ranks(values) == 1]
