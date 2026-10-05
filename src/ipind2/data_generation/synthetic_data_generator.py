"""
Synthetic Data Generator for IPIND² Platform
Integrated Platform for Intelligent Design of Drug Nanocarriers

Generates the synthetic training data for the models of Units 1 to 4. Structures come from the combinatorial library
``ipind2.generation`` (real, unique SMILES, validated with RDKit) and properties
are computed from the structure→property rules in ``properties.py`` together with noise.

Correction history: the initial version, due to calling ``Descriptors.FractionCsp3`` (correct name:
``FractionCSP3``) and a blanket ``except``, substituted random descriptors for *all* samples
(zero structure-property signal, only ~150 distinct SMILES). In this version the root problem is fixed and
the RDKit error is no longer silently swallowed.

See docs/SRS.md §8.
"""

import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..featurization import extended_dict, parse_smiles, smiles_to_graph
from ..generation.library import CombinatorialLibrary
from .properties import biological_truth, physicochemical_truth

# Physicochemical/biological target columns (shared reference for training Units 2 and 3)
PHYSICO_TARGETS = (
    "phys_size_nm",
    "phys_zeta_potential_mV",
    "phys_pdi",
    "phys_colloidal_stability_hours",
    "phys_drug_loading_efficiency_percent",
    "phys_drug_loading_content_percent",
    "phys_release_rate_constant",
)
BIO_TARGETS = (
    "bio_cytotoxicity_ic50_ug_ml",
    "bio_cellular_uptake_efficiency_percent",
    "bio_serum_protein_binding_percent",
    "bio_circulation_half_life_hours",
    "bio_tumor_to_background_ratio",
)
_DESCRIPTOR_KEYS = (
    "mol_weight",
    "logP",
    "tpsa",
    "num_rotatable_bonds",
    "num_h_donors",
    "num_h_acceptors",
    "num_rings",
    "fraction_csp3",
    "num_heavy_atoms",
)


class SyntheticDataGenerator:
    """
    Synthetic data generator for training IPIND² models.

    Args:
        random_seed: seed; two runs with the same seed give the same dataset.
        noise: measurement noise scale factor (1 = default, 0 = no noise).
    """

    def __init__(self, random_seed: int = 42, noise: float = 1.0):
        self.random_seed = random_seed
        self.noise = noise
        self._rng = np.random.default_rng(random_seed)
        random.seed(random_seed)

    # ------------------------------------------------------------------
    def _record_from_structure(self, index: int, structure) -> Optional[Dict]:
        mol = parse_smiles(structure.smiles)
        if mol is None:
            return None
        features = extended_dict(mol)
        physico = physicochemical_truth(structure.scaffold_type, features, self._rng, self.noise)
        bio = biological_truth(structure.scaffold_type, features, physico, self._rng, self.noise)

        record: Dict = {
            "id": index,
            "smiles": structure.smiles,
            "scaffold_name": structure.template,
            "scaffold_type": structure.scaffold_type,
        }
        for key in _DESCRIPTOR_KEYS:
            record[f"desc_{key}"] = features[key]
        for key, value in physico.items():
            record[f"phys_{key}"] = value
        for key, value in bio.items():
            record[f"bio_{key}"] = value
        return record

    def generate_molecule(self, scaffold_type: Optional[str] = None) -> Dict:
        """Generate one complete sample (structure + descriptors + properties) as a nested dict."""
        library = CombinatorialLibrary(scaffold_type, seed=int(self._rng.integers(0, 2**31 - 1)))
        structures, _ = library.generate(1)
        record = self._record_from_structure(0, structures[0])
        return {
            "smiles": record["smiles"],
            "scaffold_name": record["scaffold_name"],
            "scaffold_type": record["scaffold_type"],
            "descriptors": {k: record[f"desc_{k}"] for k in _DESCRIPTOR_KEYS},
            "physicochemical": {
                c[len("phys_"):]: record[c] for c in record if c.startswith("phys_")
            },
            "biological": {c[len("bio_"):]: record[c] for c in record if c.startswith("bio_")},
        }

    def generate_dataset(
        self,
        n_samples: int = 100000,
        scaffold_type: Optional[str] = None,
        include_pareto_labels: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Generate the full dataset with a given number of samples.

        Structures are unique; if the structural space is smaller than ``n_samples``
        (e.g., the metal class with a large number of samples) duplicates are allowed so the request is fulfilled.
        """
        library = CombinatorialLibrary(
            scaffold_type, seed=int(self._rng.integers(0, 2**31 - 1))
        )
        unique = n_samples <= library.space_size // 2
        structures, _ = library.generate(n_samples, unique=unique)

        rows: List[Dict] = []
        for i, structure in enumerate(structures):
            if verbose and i % 10000 == 0:
                print(f"Generating sample {i}/{n_samples}...")
            record = self._record_from_structure(i, structure)
            if record is not None:
                rows.append(record)

        df = pd.DataFrame(rows)
        if include_pareto_labels and len(df):
            df = self._add_pareto_labels(df)
        return df

    def _add_pareto_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pareto labels with true non-dominated sorting (not score thresholding).

        Objectives (maximize): loading, cellular uptake, safety (−toxicity) and closeness of size to
        the desired range (80–120 nm), and stability. ``pareto_score`` is the same previous weighted score and
        is retained for compatibility.
        """
        from ..optimization.pareto import pareto_ranks
        from ..optimization.objectives import default_objective_matrix

        objectives = default_objective_matrix(df)
        ranks = pareto_ranks(objectives)

        def norm(col):
            span = col.max() - col.min()
            return (col - col.min()) / span if span > 0 else col * 0.0

        loading_norm = norm(df["phys_drug_loading_efficiency_percent"])
        uptake_norm = norm(df["bio_cellular_uptake_efficiency_percent"])
        safety_norm = norm(df["bio_cytotoxicity_ic50_ug_ml"])
        size_score = np.exp(-(((df["phys_size_nm"] - 100.0) / 30.0) ** 2))

        df["pareto_score"] = (
            0.3 * loading_norm + 0.3 * uptake_norm + 0.25 * safety_norm + 0.15 * size_score
        )
        df["pareto_rank"] = ranks
        df["is_pareto_optimal"] = (ranks == 1).astype(int)
        return df

    def save_dataset(self, df: pd.DataFrame, filepath: str = "synthetic_dataset.csv") -> str:
        """Save the dataset to a CSV file."""
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        print(f"Total samples: {len(df)}")
        print(f"Features: {len(df.columns)}")
        if "is_pareto_optimal" in df.columns:
            print(f"Pareto-optimal samples: {int(df['is_pareto_optimal'].sum())}")
        return filepath

    def generate_for_gnn(self, n_samples: int = 50000) -> Dict:
        """
        Data ready for GNN training: the graph of each molecule + the target vector of *that same sample*.

        (The previous version assigned the first row's target to all graphs.)
        """
        df = self.generate_dataset(n_samples)
        graphs, kept = [], []
        for position, smiles in enumerate(df["smiles"].values):
            graph = smiles_to_graph(smiles)
            if graph is None:
                continue
            graphs.append(graph)
            kept.append(position)
        targets = df.iloc[kept][list(PHYSICO_TARGETS)].to_numpy(dtype=np.float32)
        return {"graphs": graphs, "targets": targets, "dataframe": df.iloc[kept].reset_index(drop=True)}


def main():
    """Main entry point for generating the dataset (with adjustable dimensions from the command line)."""
    import argparse

    parser = argparse.ArgumentParser(description="IPIND² synthetic dataset generator")
    parser.add_argument("--n-full", type=int, default=100000)
    parser.add_argument("--n-per-type", type=int, default=20000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    import os

    os.makedirs(args.out_dir, exist_ok=True)
    generator = SyntheticDataGenerator(random_seed=args.seed)

    print("=" * 60)
    print("IPIND² Synthetic Data Generator")
    print("=" * 60)
    df_full = generator.generate_dataset(n_samples=args.n_full, verbose=True)
    generator.save_dataset(df_full, os.path.join(args.out_dir, "ipind2_dataset_full.csv"))
    for scaffold_type in ("lipid", "polymer", "metal"):
        df_type = generator.generate_dataset(n_samples=args.n_per_type, scaffold_type=scaffold_type)
        generator.save_dataset(df_type, os.path.join(args.out_dir, f"ipind2_dataset_{scaffold_type}.csv"))
    df_test = generator.generate_dataset(n_samples=args.n_test)
    generator.save_dataset(df_test, os.path.join(args.out_dir, "ipind2_dataset_test.csv"))
    return df_full


if __name__ == "__main__":
    main()
