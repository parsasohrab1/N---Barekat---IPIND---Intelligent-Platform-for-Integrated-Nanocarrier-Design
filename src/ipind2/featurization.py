"""
Shared molecular featurization

This module is the only point of conversion from "SMILES" to the numerical representations used by Units 1 to 6:

* ``descriptor_vector`` — 9-descriptor RDKit vector (global molecule features)
* ``smiles_to_graph`` — atom/bond graph for the multi-task GNN (Unit 2)
* ``atom_token_ids`` — atom token sequence for the Transformer (Unit 3)
* ``generic_framework`` / ``murcko_scaffold`` — molecular scaffold identifier for measuring diversity (Unit 1)

All functions are safe against invalid SMILES (they return ``None`` or an explicit exception) because
their input is the output of a generative model and not hand-curated data.

See docs/SRS.md §4.1 (FR-01), §4.2 (FR-02), §4.3 (FR-03).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold

# RDKit logs to stderr for every invalid SMILES; in bulk generation (100k structures)
# these logs make the output unusable.
RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_NAMES: Tuple[str, ...] = (
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
DESCRIPTOR_DIM = len(DESCRIPTOR_NAMES)

# Covered atom types: lipid/polymer scaffolds (C,N,O,P,S) and metal (Si,Fe,Au,...)
ATOM_TYPES: Tuple[str, ...] = (
    "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "Fe", "Au", "Zn", "Mn", "Gd",
)
_ATOM_TYPE_INDEX = {symbol: i for i, symbol in enumerate(ATOM_TYPES)}
# one-hot atom type (+1 for "other") plus 6 numeric features
NODE_FEATURE_DIM = len(ATOM_TYPES) + 1 + 6

# Token vocabulary for Unit 3 Transformer: 0=padding, 1=unknown
TOKEN_PAD = 0
TOKEN_UNK = 1
_TOKEN_OFFSET = 2
TOKEN_VOCAB_SIZE = len(ATOM_TYPES) + _TOKEN_OFFSET


def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Convert SMILES to an RDKit molecule; ``None`` if invalid."""
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def is_valid_smiles(smiles: str) -> bool:
    """Validate a structure with RDKit — the "valid structure rate > 95%" criterion in FR-01."""
    return parse_smiles(smiles) is not None


def canonical_smiles(smiles: str) -> Optional[str]:
    """Canonical SMILES (for comparison and deduplication)."""
    mol = parse_smiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def inchikey(smiles: str) -> Optional[str]:
    """InChIKey for deduplication independent of SMILES syntax."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:  # pragma: no cover - depends on the RDKit build
        return Chem.MolToSmiles(mol)


def murcko_scaffold(smiles: str) -> Optional[str]:
    """Murcko scaffold (rings + linkers). Empty string for molecules without rings."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def generic_framework(smiles: str) -> Optional[str]:
    """
    "Molecular scaffold" identifier independent of atom type and bond order.

    The Murcko scaffold returns an empty string for lipid/polymer nanocarriers (which are mostly acyclic)
    and is ineffective for measuring FR-01 structural diversity ("coverage of ≥500 distinct scaffolds").
    This function instead keeps the scaffold topology: all atoms are converted to carbon and all bonds to
    single bonds, then the canonical SMILES is taken.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    skeleton = Chem.RWMol(mol)
    for atom in skeleton.GetAtoms():
        atom.SetAtomicNum(6)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        atom.SetIsAromatic(False)
    for bond in skeleton.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    try:
        frame = skeleton.GetMol()
        Chem.SanitizeMol(frame)
        return Chem.MolToSmiles(frame)
    except Exception:
        return None


def descriptor_dict(mol: Chem.Mol) -> Dict[str, float]:
    """RDKit molecular descriptors matching the ``desc_*`` columns in docs/SRS.md §8."""
    return {
        "mol_weight": float(Descriptors.MolWt(mol)),
        "logP": float(Descriptors.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "num_rotatable_bonds": float(Descriptors.NumRotatableBonds(mol)),
        "num_h_donors": float(Lipinski.NumHDonors(mol)),
        "num_h_acceptors": float(Lipinski.NumHAcceptors(mol)),
        "num_rings": float(Descriptors.RingCount(mol)),
        "fraction_csp3": float(Descriptors.FractionCSP3(mol)),
        "num_heavy_atoms": float(mol.GetNumHeavyAtoms()),
    }


# Functional groups affecting surface charge, toxicity and stability; the source of structure-property
# signal (generic RDKit descriptors do not see ionic charge/bond type).
_GROUP_SMARTS: Dict[str, str] = {
    "n_quat_ammonium": "[NX4+]",
    "n_amine": "[NX3;!$(N-C=O);!$(N=*);!$(N-[a]);!$(N-S);!$(N-P)]",
    "n_acid": "[$(C(=O)[OX2H1]),$(P(=O)[OX2H1]),$(S(=O)(=O)[OX2H1])]",
    "n_ester": "[CX3](=O)[OX2][#6]",
    "n_amide": "[CX3](=O)[NX3]",
    "n_ether_ch2": "[CH2][OX2][CH2]",
    "n_disulfide": "[SX2][SX2]",
    "n_metal": "[Au,Fe,Si,Zn,Mn,Gd]",
    "n_cc_double": "[CX3]=[CX3]",
    "n_hydroxyl": "[CX4][OX2H1]",
}
GROUP_NAMES: Tuple[str, ...] = tuple(_GROUP_SMARTS)
_GROUP_PATTERNS = {name: Chem.MolFromSmarts(sma) for name, sma in _GROUP_SMARTS.items()}
EXTENDED_NAMES: Tuple[str, ...] = DESCRIPTOR_NAMES + GROUP_NAMES
EXTENDED_DIM = len(EXTENDED_NAMES)


def group_counts(mol: Chem.Mol) -> Dict[str, float]:
    """Count of each functional group in the molecule."""
    return {
        name: float(len(mol.GetSubstructMatches(pattern)))
        for name, pattern in _GROUP_PATTERNS.items()
    }


def extended_dict(mol: Chem.Mol) -> Dict[str, float]:
    """RDKit descriptors + functional group counts."""
    values = descriptor_dict(mol)
    values.update(group_counts(mol))
    return values


def extended_vector(smiles_or_mol) -> Optional[np.ndarray]:
    """Global feature vector (descriptors + functional groups) in the order of ``EXTENDED_NAMES``."""
    mol = smiles_or_mol if isinstance(smiles_or_mol, Chem.Mol) else parse_smiles(smiles_or_mol)
    if mol is None:
        return None
    values = extended_dict(mol)
    return np.array([values[name] for name in EXTENDED_NAMES], dtype=np.float32)


def extended_matrix(smiles_list: Sequence[str]) -> Tuple[np.ndarray, List[int]]:
    """Extended feature matrix; like ``descriptor_matrix`` but with functional groups."""
    rows: List[np.ndarray] = []
    kept: List[int] = []
    for i, smiles in enumerate(smiles_list):
        vector = extended_vector(smiles)
        if vector is not None:
            rows.append(vector)
            kept.append(i)
    if not rows:
        return np.zeros((0, EXTENDED_DIM), dtype=np.float32), []
    return np.vstack(rows), kept


def descriptor_vector(smiles_or_mol) -> Optional[np.ndarray]:
    """Descriptor vector in the order of ``DESCRIPTOR_NAMES``; ``None`` for invalid input."""
    mol = smiles_or_mol if isinstance(smiles_or_mol, Chem.Mol) else parse_smiles(smiles_or_mol)
    if mol is None:
        return None
    values = descriptor_dict(mol)
    return np.array([values[name] for name in DESCRIPTOR_NAMES], dtype=np.float32)


def descriptor_matrix(smiles_list: Sequence[str]) -> Tuple[np.ndarray, List[int]]:
    """
    Descriptor matrix for a list of SMILES.

    Returns:
        (matrix of shape (n_valid, DESCRIPTOR_DIM), indices of valid samples in the input)
    """
    rows: List[np.ndarray] = []
    kept: List[int] = []
    for i, smiles in enumerate(smiles_list):
        vector = descriptor_vector(smiles)
        if vector is not None:
            rows.append(vector)
            kept.append(i)
    if not rows:
        return np.zeros((0, DESCRIPTOR_DIM), dtype=np.float32), []
    return np.vstack(rows), kept


def _atom_features(atom: Chem.Atom) -> List[float]:
    one_hot = [0.0] * (len(ATOM_TYPES) + 1)
    index = _ATOM_TYPE_INDEX.get(atom.GetSymbol())
    one_hot[index if index is not None else len(ATOM_TYPES)] = 1.0
    return one_hot + [
        atom.GetDegree() / 4.0,
        atom.GetTotalNumHs() / 4.0,
        float(atom.GetFormalCharge()),
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        atom.GetMass() / 200.0,
    ]


@dataclass
class MolGraph:
    """Dense molecular graph — no torch-geometric required."""

    node_features: np.ndarray  # (n_atoms, NODE_FEATURE_DIM)
    adjacency: np.ndarray  # (n_atoms, n_atoms) symmetric, with self-loop
    n_atoms: int
    smiles: str = ""


def smiles_to_graph(smiles: str, max_atoms: int = 128) -> Optional[MolGraph]:
    """
    Convert SMILES to a dense graph for the Unit 2 MPNN.

    Molecules larger than ``max_atoms`` are truncated to the first ``max_atoms`` atoms (polymers
    can have hundreds of atoms and the model needs a fixed size).
    """
    mol = parse_smiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    n_atoms = min(mol.GetNumAtoms(), max_atoms)
    nodes = np.zeros((n_atoms, NODE_FEATURE_DIM), dtype=np.float32)
    for i in range(n_atoms):
        nodes[i] = _atom_features(mol.GetAtomWithIdx(i))

    adjacency = np.eye(n_atoms, dtype=np.float32)  # self-loop
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i < n_atoms and j < n_atoms:
            weight = float(bond.GetBondTypeAsDouble())
            adjacency[i, j] = weight
            adjacency[j, i] = weight

    return MolGraph(node_features=nodes, adjacency=adjacency, n_atoms=n_atoms, smiles=smiles)


def batch_graphs(graphs: Sequence[MolGraph]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Padding of several graphs of different sizes.

    Returns:
        (nodes (B, N, F), adjacency (B, N, N), mask (B, N) where 1=real atom)
    """
    if not graphs:
        raise ValueError("batch_graphs requires at least one graph")
    batch = len(graphs)
    n_max = max(g.n_atoms for g in graphs)
    nodes = np.zeros((batch, n_max, NODE_FEATURE_DIM), dtype=np.float32)
    adjacency = np.zeros((batch, n_max, n_max), dtype=np.float32)
    mask = np.zeros((batch, n_max), dtype=np.float32)
    for b, graph in enumerate(graphs):
        n = graph.n_atoms
        nodes[b, :n] = graph.node_features
        adjacency[b, :n, :n] = graph.adjacency
        mask[b, :n] = 1.0
    return nodes, adjacency, mask


def atom_token_ids(smiles: str, max_len: int = 128) -> Optional[np.ndarray]:
    """Atom token sequence (with padding) for the Unit 3 Transformer."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    tokens = np.full(max_len, TOKEN_PAD, dtype=np.int64)
    for i, atom in enumerate(mol.GetAtoms()):
        if i >= max_len:
            break
        index = _ATOM_TYPE_INDEX.get(atom.GetSymbol())
        tokens[i] = TOKEN_UNK if index is None else index + _TOKEN_OFFSET
    return tokens


def count_distinct_skeletons(smiles_list: Sequence[str]) -> int:
    """Number of distinct molecular scaffolds — the "structural diversity" measure in FR-01."""
    return len({f for f in (generic_framework(s) for s in smiles_list) if f})


def validity_rate(smiles_list: Sequence[str]) -> float:
    """Ratio of RDKit-valid structures — the "valid structure rate > 95%" measure in FR-01."""
    if not smiles_list:
        return 0.0
    return sum(1 for s in smiles_list if is_valid_smiles(s)) / len(smiles_list)
