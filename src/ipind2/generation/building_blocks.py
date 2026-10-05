"""
Library of building blocks and structural templates for nanocarriers (input of Unit 1, FR-01).

Approach: structures are built as "template + blocks", not by character-by-character generation of
SMILES. Reason: FR-01 requires a validity rate > 95%; the templates here are written so that
every allowed combination of blocks gives a syntactically and chemically valid SMILES (in ``tests/test_generation.py``
the validity rate of all three scaffold classes is measured). The generative model (CVAE) works on the *selection* of blocks and the
descriptor space, not on SMILES syntax.

Blocks are combined by string concatenation; each intermediate block is written so that its last atom
has the bonding valence for the next block.

See docs/SRS.md §4.1 (FR-01).
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

SCAFFOLD_TYPES: Tuple[str, ...] = ("lipid", "polymer", "metal")


def _alkyl_tails() -> List[str]:
    """
    Hydrophobic tails: saturated, unsaturated, branched, hydroxylated and biodegradable (ester).

    Chain length/saturation diversity is the most important lever for tuning size and pKa in ionizable lipids,
    so its range (6 to 24 carbons) is covered.
    """
    tails: List[str] = []
    for n in range(6, 25):
        tails.append("C" * n)  # saturated
    for n in range(12, 25, 2):
        for position in (n // 3, n // 2):
            tails.append("C" * position + "C=C" + "C" * (n - position - 2))  # unsaturated
    for n in range(9, 22, 3):
        tails.append("C" * (n - 3) + "C(C)C")  # terminal branched
    for n in range(10, 22, 3):
        tails.append("C" * (n // 2) + "C(O)" + "C" * (n - n // 2 - 1))  # hydroxylated
    for n in range(10, 22, 3):
        tails.append("C" * (n // 2) + "C(=O)O" + "C" * (n - n // 2 - 1))  # biodegradable ester
    return tails


TAILS: Tuple[str, ...] = tuple(_alkyl_tails())

# Biodegradable/stable linkers between the tail and the polar head
LINKERS: Tuple[str, ...] = (
    "C(=O)O",      # ester
    "OC(=O)",      # reverse ester
    "C(=O)N",      # amide
    "NC(=O)",      # reverse amide
    "OC(=O)N",     # carbamate
    "SSC",         # disulfide (redox-responsive)
    "O",           # ether
    "N",           # secondary amine
    "S",           # thioether
    "OCCO",        # short ethylene glycol
    "CC(O)C",      # propanediol
    "C(=O)OCC",    # ester + spacer
)

# Polar/ionizable heads (terminal)
POLAR_HEADS: Tuple[str, ...] = (
    "[N+](C)(C)C",                   # quaternary ammonium (permanently cationic)
    "N(C)C",                         # tertiary amine (ionizable)
    "N(CC)CC",
    "NCCN(C)C",                      # polyamine
    "N",                             # primary amine
    "O",                             # hydroxyl
    "OCCO",
    "OCCOCCOCCO",                    # short PEG
    "C(=O)O",                        # carboxyl (anionic)
    "C(N)=O",                        # primary amide
    "N1CCOCC1",                      # morpholine
    "N1CCCC1",                       # pyrrolidine
    "N1CCN(C)CC1",                   # piperazine
    "c1ccncc1",                      # pyridine
    "OS(=O)(=O)O",                   # sulfate
    "OP(=O)(O)O",                    # phosphate
)

# Glycerol/phospholipid heads (for the two-tail template)
GLYCEROL_HEADS: Tuple[str, ...] = (
    "COP(=O)(O)OCC[N+](C)(C)C",      # phosphocholine
    "COP(=O)(O)OCCN",                # phosphoethanolamine
    "COP(=O)(O)OCC(N)C(=O)O",        # phosphoserine
    "COP(=O)(O)OCCOCCOCCO",          # PEG-phosphate
    "CO",                            # diacylglycerol
    "COC(=O)CCC(=O)O",               # succinyl
    "COP(=O)(O)OCC(O)CO",            # phosphoglycerol
    "COP(=O)(O)O",                   # phosphatidic acid
    "COCC(O)CO",                     # glyceryl ether
    "COC(=O)N",                      # carbamate
    "COS(=O)(=O)O",                  # sulfate
    "COCCN(C)C",                     # aminoethyl ether
)

# End cap for the dialkylamine template
AMINE_CAPS: Tuple[str, ...] = (
    "C",
    "CCO",
    "CC(=O)O",
    "CCN(C)C",
    "CCOCCO",
    "CC(O)CO",
    "CCS",
    "CC(N)=O",
    "CCOCCOCCO",
    "CC1CCOCC1",
    "CCN1CCOCC1",
    "CC(O)C(O)CO",
)

# Polymer repeat units (each unit is self-contained and repeatable)
POLYMER_REPEAT_UNITS: Dict[str, str] = {
    "PLA": "OC(C)C(=O)",
    "PGA": "OCC(=O)",
    "PEG": "OCC",
    "PCL": "OCCCCCC(=O)",
    "PAA": "CC(C(=O)O)",
    "PNIPAM": "CC(C(=O)NC(C)C)",
    "chitosan_like": "OC(CO)C(N)",
    "PBAE": "OCCOC(=O)CCN(C)CCC(=O)",
}

POLYMER_INITIATORS: Tuple[str, ...] = ("CC", "CCCC", "CO", "OCC", "NCC", "c1ccccc1C")
POLYMER_TERMINATORS: Tuple[str, ...] = ("O", "OC", "N", "OCCO", "C(=O)O", "OCCOCCOCCO")

# Surface anchors of metal/inorganic nanoparticles (terminal)
METAL_ANCHORS: Dict[str, str] = {
    "Au": "S[Au]",
    "Au_dithiol": "SC(S)[Au]",
    "MSN": "[Si](O)(O)O",
    "Fe3O4": "C(=O)O[Fe]",
    "ZnO": "OC(=O)[Zn]",
    "MnO": "OC(=O)[Mn]",
    "Gd_chelate": "N(CC(=O)O)CC(=O)O[Gd]",
    "Si_amino": "[Si](O)(O)OCCN",
}


@dataclass(frozen=True)
class StructureTemplate:
    """
    A structural template: a string pattern with named slots and the allowed range of each slot.

    Attributes:
        name: template identifier (reported as ``template`` in the output).
        scaffold_type: 'lipid' | 'polymer' | 'metal'.
        pattern: pattern with ``{slot}`` placeholders.
        slots: slot name -> list of allowed values.
    """

    name: str
    scaffold_type: str
    pattern: str
    slots: Tuple[Tuple[str, Tuple[str, ...]], ...]

    def slot_options(self) -> Dict[str, Tuple[str, ...]]:
        return dict(self.slots)

    def build(self, choices: Dict[str, str]) -> str:
        return self.pattern.format(**choices)

    def combination_count(self) -> int:
        total = 1
        for _, options in self.slots:
            total *= len(options)
        return total


_REPEAT_BLOCKS: Tuple[str, ...] = tuple(
    unit * n
    for unit in POLYMER_REPEAT_UNITS.values()
    for n in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)
)
_REPEAT_BLOCKS_SHORT: Tuple[str, ...] = tuple(
    unit * n for unit in POLYMER_REPEAT_UNITS.values() for n in (1, 2, 3, 4, 6, 8)
)


TEMPLATES: Tuple[StructureTemplate, ...] = (
    StructureTemplate(
        name="lipid_single_chain",
        scaffold_type="lipid",
        pattern="{tail1}{linker}{head}",
        slots=(("tail1", TAILS), ("linker", LINKERS), ("head", POLAR_HEADS)),
    ),
    StructureTemplate(
        name="lipid_head_first",
        scaffold_type="lipid",
        pattern="C{head}{linker}{tail1}",
        slots=(("head", ("N(C)", "N(CC)", "N(CCO)")), ("linker", LINKERS), ("tail1", TAILS)),
    ),
    StructureTemplate(
        name="phospholipid_two_tail",
        scaffold_type="lipid",
        pattern="{tail1}C(=O)OCC(OC(=O){tail2}){head}",
        slots=(("tail1", TAILS), ("tail2", TAILS), ("head", GLYCEROL_HEADS)),
    ),
    StructureTemplate(
        name="dialkyl_amine",
        scaffold_type="lipid",
        pattern="{tail1}N({tail2}){cap}",
        slots=(("tail1", TAILS), ("tail2", TAILS), ("cap", AMINE_CAPS)),
    ),
    StructureTemplate(
        name="lipidoid_three_tail",
        scaffold_type="lipid",
        pattern="{tail1}N({tail2})CCN({tail3}){cap}",
        slots=(("tail1", TAILS), ("tail2", TAILS), ("tail3", TAILS), ("cap", AMINE_CAPS)),
    ),
    StructureTemplate(
        name="cationic_dialkyl_quat",
        scaffold_type="lipid",
        pattern="{tail1}[N+]({tail2})(C)C",
        slots=(("tail1", TAILS), ("tail2", TAILS)),
    ),
    StructureTemplate(
        name="dotap_like_quat",
        scaffold_type="lipid",
        pattern="{tail1}C(=O)OCC(OC(=O){tail2})C[N+](C)(C)C",
        slots=(("tail1", TAILS), ("tail2", TAILS)),
    ),
    StructureTemplate(
        name="quat_single_chain",
        scaffold_type="lipid",
        pattern="{tail1}{linker}CC[N+]({cap})(C)C",
        slots=(("tail1", TAILS), ("linker", LINKERS), ("cap", ("C", "CC", "CCO", "CCCC", "CC(O)C"))),
    ),
    StructureTemplate(
        name="lipid_diester",
        scaffold_type="lipid",
        pattern="{tail1}OC(=O)CCC(=O)O{tail2}",
        slots=(("tail1", TAILS), ("tail2", TAILS)),
    ),
    StructureTemplate(
        name="polymer_homo",
        scaffold_type="polymer",
        pattern="{initiator}{repeat}{terminator}",
        slots=(
            ("initiator", POLYMER_INITIATORS),
            ("repeat", _REPEAT_BLOCKS),
            ("terminator", POLYMER_TERMINATORS),
        ),
    ),
    StructureTemplate(
        name="polymer_block_copolymer",
        scaffold_type="polymer",
        pattern="{initiator}{repeat_a}{repeat_b}{terminator}",
        slots=(
            ("initiator", POLYMER_INITIATORS),
            ("repeat_a", _REPEAT_BLOCKS_SHORT),
            ("repeat_b", _REPEAT_BLOCKS_SHORT),
            ("terminator", POLYMER_TERMINATORS),
        ),
    ),
    StructureTemplate(
        name="polymer_lipid_conjugate",
        scaffold_type="polymer",
        pattern="{tail1}C(=O)O{repeat}{terminator}",
        slots=(
            ("tail1", TAILS),
            ("repeat", _REPEAT_BLOCKS_SHORT),
            ("terminator", POLYMER_TERMINATORS),
        ),
    ),
    StructureTemplate(
        name="polymer_segmented",
        scaffold_type="polymer",
        pattern="{initiator}{repeat_a}{linker}{repeat_b}{terminator}",
        slots=(
            ("initiator", POLYMER_INITIATORS),
            ("repeat_a", _REPEAT_BLOCKS_SHORT),
            ("linker", LINKERS),
            ("repeat_b", _REPEAT_BLOCKS_SHORT),
            ("terminator", POLYMER_TERMINATORS),
        ),
    ),
    StructureTemplate(
        name="metal_mixed_ligand",
        scaffold_type="metal",
        pattern="{tail1}{linker}N({tail2}){anchor}",
        slots=(
            ("tail1", TAILS),
            ("linker", LINKERS),
            ("tail2", TAILS),
            ("anchor", tuple(METAL_ANCHORS.values())),
        ),
    ),
    StructureTemplate(
        name="metal_ligand",
        scaffold_type="metal",
        pattern="{tail1}{anchor}",
        slots=(("tail1", TAILS), ("anchor", tuple(METAL_ANCHORS.values()))),
    ),
    StructureTemplate(
        name="metal_ligand_linker",
        scaffold_type="metal",
        pattern="{tail1}{linker}{anchor}",
        slots=(
            ("tail1", TAILS),
            ("linker", LINKERS),
            ("anchor", tuple(METAL_ANCHORS.values())),
        ),
    ),
    StructureTemplate(
        name="metal_peg_ligand",
        scaffold_type="metal",
        pattern="{peg}{anchor}",
        slots=(
            ("peg", tuple("OCC" * n + "C" for n in (2, 4, 6, 8, 12, 16))),
            ("anchor", tuple(METAL_ANCHORS.values())),
        ),
    ),
)


def templates_for(scaffold_type: str = None) -> List[StructureTemplate]:
    """Templates of one scaffold class (or all templates if ``None``)."""
    if scaffold_type is None:
        return list(TEMPLATES)
    if scaffold_type not in SCAFFOLD_TYPES:
        raise ValueError(
            f"Invalid scaffold_type: {scaffold_type!r} (allowed: {', '.join(SCAFFOLD_TYPES)})"
        )
    return [t for t in TEMPLATES if t.scaffold_type == scaffold_type]


def theoretical_library_size(scaffold_type: str = None) -> int:
    """
    Theoretical size of the reachable structural space (upper bound of the virtual library).

    Basis of the NFR-08/FR-01 claim about library scale: the enumerable space of templates.
    """
    return sum(t.combination_count() for t in templates_for(scaffold_type))


def all_building_blocks() -> Dict[str, Sequence[str]]:
    """All building blocks, for documentation and chemical coverage reporting."""
    return {
        "tails": TAILS,
        "linkers": LINKERS,
        "polar_heads": POLAR_HEADS,
        "glycerol_heads": GLYCEROL_HEADS,
        "amine_caps": AMINE_CAPS,
        "polymer_repeat_units": tuple(POLYMER_REPEAT_UNITS.values()),
        "polymer_initiators": POLYMER_INITIATORS,
        "polymer_terminators": POLYMER_TERMINATORS,
        "metal_anchors": tuple(METAL_ANCHORS.values()),
    }
