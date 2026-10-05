"""
Structure→property model (simulation reference) for synthetic data.

These are the "ground truth" rules of the synthetic data, not a prediction model. The shape of the rules is inspired by
well-known trends in the nanomedicine literature (correct direction of effect, not calibrated coefficients):

* Size increases with molecular weight and hydrophobicity; polymers are larger than lipids and
  metal nanoparticles are smaller.
* Zeta potential comes from ionic charge (quaternary ammonium/amine +, acid −).
* Cationic toxicity (lower IC50 for quaternary ammonium and high hydrophobicity); PEG reduces toxicity.
* Cellular uptake is higher with positive zeta and size ~100 nm; circulation half-life is longer with PEG.

⚠️ Because these are artificial rules, the accuracy the models achieve on this data is not proof of accuracy on
real data (see docs/MODEL_VALIDATION.md). Models must be retrained with lab data;
this data confirms the pipeline and its ability to learn.

See docs/SRS.md §8.
"""

from typing import Dict

import numpy as np

# Base shift by scaffold class
_SIZE_OFFSET = {"lipid": 55.0, "polymer": 85.0, "metal": 30.0}
_ZETA_OFFSET = {"lipid": 0.0, "polymer": -3.0, "metal": -8.0}
_IC50_OFFSET = {"lipid": 0.0, "polymer": 18.0, "metal": 8.0}

CELL_LINES = ("hek293", "hepg2", "hela")
# Relative sensitivity of each cell line (multiplicative factor on IC50)
_CELL_LINE_FACTOR = {"hek293": 1.0, "hepg2": 1.25, "hela": 0.85}

NOISE_SD = {
    "size": 3.0,
    "zeta": 1.5,
    "pdi": 0.008,
    "stability": 2.0,
    "loading_efficiency": 2.0,
    "loading_content": 0.2,
    "release": 0.01,
    "ic50": 3.0,
    "uptake": 2.0,
    "binding": 2.0,
    "half_life": 0.25,
    "tbr": 0.04,
}


def _clip(values, low, high):
    return np.clip(values, low, high)


def physicochemical_truth(
    scaffold_type: str, features: Dict[str, float], rng: np.random.Generator, noise: float = 1.0
) -> Dict[str, float]:
    """The seven FR-02 physicochemical properties from structural features."""
    mw = features["mol_weight"]
    logp = features["logP"]
    cation = features["n_quat_ammonium"] + 0.5 * features["n_amine"]
    anion = features["n_acid"]
    peg = features["n_ether_ch2"]
    ester = features["n_ester"]

    def n(key):
        return rng.normal(0.0, NOISE_SD[key] * noise)

    size = _SIZE_OFFSET[scaffold_type] + 14.0 * np.log1p(mw / 150.0) + 1.5 * logp - 1.5 * np.sqrt(peg)
    size = _clip(size + n("size"), 10.0, 500.0)

    # Ionic charge saturates as the number of groups increases (counter-ion coverage/Debye screening)
    zeta = (
        _ZETA_OFFSET[scaffold_type]
        + 26.0 * np.tanh(cation / 2.0)
        - 26.0 * np.tanh(anion / 2.5)
        - 1.2 * np.sqrt(peg)
        - 0.4 * logp
    )
    zeta = _clip(zeta + n("zeta"), -50.0, 50.0)

    pdi = 0.04 + 0.0007 * size + 0.006 * features["num_rotatable_bonds"] ** 0.5 + 0.02 * (
        scaffold_type == "polymer"
    )
    pdi = _clip(pdi + n("pdi"), 0.01, 0.5)

    stability = 8.0 + 1.6 * abs(zeta) - 0.05 * size + 4.0 * (1 - pdi) + 1.5 * np.sqrt(peg) - 0.8 * ester
    stability = _clip(stability + n("stability"), 1.0, 100.0)

    loading_eff = 55.0 + 0.25 * abs(zeta) - 0.08 * size + 6.0 * (1 - pdi) + 0.9 * logp
    loading_eff = _clip(loading_eff + n("loading_efficiency"), 10.0, 99.0)

    loading_content = 2.0 + 0.12 * loading_eff + 0.3 * features["num_rings"]
    loading_content = _clip(loading_content + n("loading_content"), 0.5, 30.0)

    release = 0.015 + 0.0005 * size - 0.001 * abs(zeta) + 0.012 * ester + 0.008 * (1 - pdi)
    release = _clip(release + n("release"), 0.001, 0.5)

    return {
        "size_nm": float(size),
        "zeta_potential_mV": float(zeta),
        "pdi": float(pdi),
        "colloidal_stability_hours": float(stability),
        "drug_loading_efficiency_percent": float(loading_eff),
        "drug_loading_content_percent": float(loading_content),
        "release_rate_constant": float(release),
    }


def biological_truth(
    scaffold_type: str,
    features: Dict[str, float],
    physico: Dict[str, float],
    rng: np.random.Generator,
    noise: float = 1.0,
) -> Dict[str, float]:
    """The five FR-03 biological properties (toxicity on 3 cell lines + four other properties)."""
    logp = features["logP"]
    quat = features["n_quat_ammonium"]
    peg = features["n_ether_ch2"]
    size = physico["size_nm"]
    zeta = physico["zeta_potential_mV"]

    def n(key):
        return rng.normal(0.0, NOISE_SD[key] * noise)

    base_ic50 = (
        75.0
        + _IC50_OFFSET[scaffold_type]
        - 1.6 * logp
        - 18.0 * np.tanh(quat / 1.5)
        - 0.45 * max(zeta, 0.0)
        + 3.0 * np.sqrt(peg)
        - 5.0 * features["n_disulfide"]
    )
    ic50 = {
        line: float(_clip(base_ic50 * factor + n("ic50"), 2.0, 300.0))
        for line, factor in _CELL_LINE_FACTOR.items()
    }

    uptake = 38.0 + 0.7 * zeta - 0.12 * abs(size - 100.0) - 1.0 * np.sqrt(peg) + 0.6 * logp
    uptake = float(_clip(uptake + n("uptake"), 5.0, 95.0))

    binding = 28.0 + 1.5 * logp + 0.12 * size + 0.35 * abs(zeta) - 2.0 * np.sqrt(peg)
    binding = float(_clip(binding + n("binding"), 5.0, 95.0))

    half_life = 4.5 + 1.1 * np.sqrt(peg) - 0.012 * size - 0.05 * max(zeta, 0.0) - 0.02 * binding
    half_life = float(_clip(half_life + n("half_life"), 0.5, 24.0))

    tbr = 0.8 + 0.018 * uptake + 0.12 * half_life - 0.004 * abs(size - 110.0) - 0.002 * ic50["hek293"]
    tbr = float(_clip(tbr + n("tbr"), 0.1, 10.0))

    return {
        "cytotoxicity_ic50_ug_ml": ic50["hek293"],
        "cytotoxicity_ic50_hepg2_ug_ml": ic50["hepg2"],
        "cytotoxicity_ic50_hela_ug_ml": ic50["hela"],
        "cellular_uptake_efficiency_percent": uptake,
        "serum_protein_binding_percent": binding,
        "circulation_half_life_hours": half_life,
        "tumor_to_background_ratio": tbr,
    }
