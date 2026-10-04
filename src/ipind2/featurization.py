"""
ویژگی‌سازی مشترک مولکول‌ها (Molecular Featurization)

این ماژول تنها نقطه تبدیل «SMILES» به نمایش‌های عددی مورد استفاده واحدهای ۱ تا ۶ است:

* ``descriptor_vector`` — بردار ۹ توصیف‌گر RDKit (ویژگی‌های global مولکول)
* ``smiles_to_graph`` — گراف اتم/پیوند برای GNN چندوظیفه‌ای (واحد ۲)
* ``atom_token_ids`` — دنباله توکن اتمی برای Transformer (واحد ۳)
* ``generic_framework`` / ``murcko_scaffold`` — شناسه اسکلت مولکولی برای سنجش تنوع (واحد ۱)

همه توابع در برابر SMILES نامعتبر امن‌اند (``None`` یا استثنای صریح برمی‌گردانند) چون
ورودی آن‌ها خروجی یک مدل مولد است و نه داده دست‌چین‌شده.

See docs/SRS.md §4.1 (FR-01), §4.2 (FR-02), §4.3 (FR-03).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold

# RDKit برای هر SMILES نامعتبر به stderr لاگ می‌دهد؛ در تولید انبوه (۱۰۰k ساختار)
# این لاگ‌ها خروجی را غیرقابل‌استفاده می‌کنند.
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

# انواع اتم پوشش‌داده‌شده: اسکلت‌های لیپیدی/پلیمری (C,N,O,P,S) و فلزی (Si,Fe,Au,...)
ATOM_TYPES: Tuple[str, ...] = (
    "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "Fe", "Au", "Zn", "Mn", "Gd",
)
_ATOM_TYPE_INDEX = {symbol: i for i, symbol in enumerate(ATOM_TYPES)}
# one-hot نوع اتم (+۱ برای «سایر») به‌علاوه ۶ ویژگی عددی
NODE_FEATURE_DIM = len(ATOM_TYPES) + 1 + 6

# واژگان توکن برای Transformer واحد ۳: ۰=padding، ۱=unknown
TOKEN_PAD = 0
TOKEN_UNK = 1
_TOKEN_OFFSET = 2
TOKEN_VOCAB_SIZE = len(ATOM_TYPES) + _TOKEN_OFFSET


def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """تبدیل SMILES به مولکول RDKit؛ ``None`` اگر نامعتبر باشد."""
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def is_valid_smiles(smiles: str) -> bool:
    """اعتبارسنجی ساختار با RDKit — معیار «نرخ ساختارهای معتبر > ۹۵٪» در FR-01."""
    return parse_smiles(smiles) is not None


def canonical_smiles(smiles: str) -> Optional[str]:
    """SMILES کانونیک (برای مقایسه و حذف تکراری)."""
    mol = parse_smiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def inchikey(smiles: str) -> Optional[str]:
    """InChIKey برای حذف تکراری مستقل از نحو SMILES."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:  # pragma: no cover - وابسته به بیلد RDKit
        return Chem.MolToSmiles(mol)


def murcko_scaffold(smiles: str) -> Optional[str]:
    """اسکلت Murcko (حلقه‌ها + لینکرها). برای مولکول‌های بدون حلقه رشته خالی است."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def generic_framework(smiles: str) -> Optional[str]:
    """
    شناسه «اسکلت مولکولی» مستقل از نوع اتم و مرتبه پیوند.

    اسکلت Murcko برای نانوحامل‌های لیپیدی/پلیمری (که اغلب آسیکلیک‌اند) رشته خالی
    برمی‌گرداند و برای سنجش تنوع ساختاری FR-01 («پوشش ≥۵۰۰ اسکلت متفاوت») بی‌اثر است.
    این تابع در عوض توپولوژی اسکلت را نگه می‌دارد: همه اتم‌ها به کربن و همه پیوندها به
    پیوند یگانه تبدیل می‌شوند، سپس SMILES کانونیک گرفته می‌شود.
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
    """توصیف‌گرهای مولکولی RDKit مطابق ستون‌های ``desc_*`` در docs/SRS.md §۸."""
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


# گروه‌های عاملی مؤثر بر بار سطحی، سمیت و پایداری؛ منبع سیگنال ساختار-خاصیت
# (توصیف‌گرهای عمومی RDKit بار یونی/نوع پیوند را نمی‌بینند).
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
    """شمار هر گروه عاملی در مولکول."""
    return {
        name: float(len(mol.GetSubstructMatches(pattern)))
        for name, pattern in _GROUP_PATTERNS.items()
    }


def extended_dict(mol: Chem.Mol) -> Dict[str, float]:
    """توصیف‌گرهای RDKit + شمار گروه‌های عاملی."""
    values = descriptor_dict(mol)
    values.update(group_counts(mol))
    return values


def extended_vector(smiles_or_mol) -> Optional[np.ndarray]:
    """بردار ویژگی global (توصیف‌گر + گروه عاملی) به ترتیب ``EXTENDED_NAMES``."""
    mol = smiles_or_mol if isinstance(smiles_or_mol, Chem.Mol) else parse_smiles(smiles_or_mol)
    if mol is None:
        return None
    values = extended_dict(mol)
    return np.array([values[name] for name in EXTENDED_NAMES], dtype=np.float32)


def extended_matrix(smiles_list: Sequence[str]) -> Tuple[np.ndarray, List[int]]:
    """ماتریس ویژگی گسترده؛ مانند ``descriptor_matrix`` اما با گروه‌های عاملی."""
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
    """بردار توصیف‌گر به ترتیب ``DESCRIPTOR_NAMES``؛ ``None`` برای ورودی نامعتبر."""
    mol = smiles_or_mol if isinstance(smiles_or_mol, Chem.Mol) else parse_smiles(smiles_or_mol)
    if mol is None:
        return None
    values = descriptor_dict(mol)
    return np.array([values[name] for name in DESCRIPTOR_NAMES], dtype=np.float32)


def descriptor_matrix(smiles_list: Sequence[str]) -> Tuple[np.ndarray, List[int]]:
    """
    ماتریس توصیف‌گر برای فهرستی از SMILES.

    Returns:
        (ماتریس به شکل (n_valid, DESCRIPTOR_DIM)، اندیس نمونه‌های معتبر در ورودی)
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
    """گراف مولکولی متراکم (dense) — بدون نیاز به torch-geometric."""

    node_features: np.ndarray  # (n_atoms, NODE_FEATURE_DIM)
    adjacency: np.ndarray  # (n_atoms, n_atoms) متقارن، با self-loop
    n_atoms: int
    smiles: str = ""


def smiles_to_graph(smiles: str, max_atoms: int = 128) -> Optional[MolGraph]:
    """
    تبدیل SMILES به گراف متراکم برای MPNN واحد ۲.

    مولکول‌های بزرگ‌تر از ``max_atoms`` به ``max_atoms`` اتم اول برش می‌خورند (پلیمرها
    می‌توانند صدها اتم داشته باشند و برای مدل اندازه ثابت لازم است).
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
    دسته‌بندی (padding) چند گراف با اندازه متفاوت.

    Returns:
        (nodes (B, N, F)، adjacency (B, N, N)، mask (B, N) که ۱=اتم واقعی)
    """
    if not graphs:
        raise ValueError("batch_graphs به حداقل یک گراف نیاز دارد")
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
    """دنباله توکن اتمی (با padding) برای Transformer واحد ۳."""
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
    """تعداد اسکلت‌های مولکولی متمایز — سنجه «تنوع ساختاری» در FR-01."""
    return len({f for f in (generic_framework(s) for s in smiles_list) if f})


def validity_rate(smiles_list: Sequence[str]) -> float:
    """نسبت ساختارهای معتبر RDKit — سنجه «نرخ ساختارهای معتبر > ۹۵٪» در FR-01."""
    if not smiles_list:
        return 0.0
    return sum(1 for s in smiles_list if is_valid_smiles(s)) / len(smiles_list)
