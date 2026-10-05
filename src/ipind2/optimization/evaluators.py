"""
Objective-function evaluators for the optimizer: Unit 2/3 predictors and the reference oracle.

* ``PredictorObjective`` — production evaluator: physicochemical GNN output + biological Transformer.
* ``oracle_objective`` — noise-free ground-truth rules of the synthetic data; only for
  optimizer benchmarking (is PG-RL better than random search?) and fast tests.
"""

from typing import Sequence

import numpy as np
import pandas as pd

from ..data_generation.properties import biological_truth, physicochemical_truth
from ..featurization import extended_dict, parse_smiles
from .objectives import OBJECTIVE_COLUMNS


def _scaffold_of(smiles: str) -> str:
    """Guess the scaffold type from elements/groups (for the oracle evaluator that cannot see the template)."""
    mol = parse_smiles(smiles)
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    if symbols & {"Au", "Fe", "Si", "Zn", "Mn", "Gd"}:
        return "metal"
    features = extended_dict(mol)
    # Polymers: many repeating ester/ether bonds; lipids: long tails without repetition
    repeat_signal = features["n_ether_ch2"] + features["n_ester"] + features["n_amide"]
    return "polymer" if repeat_signal >= 4 else "lipid"


def oracle_objective(smiles_list: Sequence[str]) -> pd.DataFrame:
    """Noise-free evaluator based on structure→property rules (for benchmark/test)."""
    rng = np.random.default_rng(0)
    rows = []
    for smiles in smiles_list:
        mol = parse_smiles(smiles)
        features = extended_dict(mol)
        scaffold = _scaffold_of(smiles)
        physico = physicochemical_truth(scaffold, features, rng, noise=0.0)
        bio = biological_truth(scaffold, features, physico, rng, noise=0.0)
        rows.append({**{f"phys_{k}": v for k, v in physico.items()}, **{f"bio_{k}": v for k, v in bio.items()}})
    return pd.DataFrame(rows)


class PredictorObjective:
    """
    Objective function based on trained predictors.

    ``__call__(smiles) -> DataFrame`` with ``OBJECTIVE_COLUMNS`` columns; rows of
    invalid SMILES are not dropped but filled with a bad value so the order is preserved.
    """

    def __init__(self, physico_predictor, bio_predictor, uncertainty_penalty: float = 0.0):
        self.physico = physico_predictor
        self.bio = bio_predictor
        self.uncertainty_penalty = uncertainty_penalty

    def __call__(self, smiles_list: Sequence[str]) -> pd.DataFrame:
        smiles_list = list(smiles_list)
        phys_mean, phys_std, phys_kept = self.physico.predict_with_uncertainty(smiles_list)
        bio_mean, bio_std, bio_kept = self.bio.predict_with_uncertainty(smiles_list)
        frame = pd.DataFrame(index=range(len(smiles_list)), columns=list(OBJECTIVE_COLUMNS.values()), dtype=float)
        frame[:] = np.nan
        for column in OBJECTIVE_COLUMNS.values():
            source_mean, source_std = (
                (phys_mean, phys_std) if column.startswith("phys_") else (bio_mean, bio_std)
            )
            values = source_mean[column]
            # Pessimism against uncertainty: k·σ is subtracted from maximization objectives. Size
            # is exempt because its objective is "closeness to the range", not linear maximization.
            if self.uncertainty_penalty and column != OBJECTIVE_COLUMNS["size_fit"]:
                values = values - self.uncertainty_penalty * source_std[column]
            frame.loc[values.index, column] = values.to_numpy()
        return frame.fillna(-1e6)
