"""
Real-MD validation campaign for the pipeline's top candidates (FR-05 / TRL criterion R3).

    python -m ipind2.md_simulation.campaign --model-dir models/v1 --n 3 --ns 2 --out docs/md_validation.json

Designs candidates with the trained bundle (no MD in the design step), simulates each with the
``CondaOpenMMEngine`` and writes a JSON report whose ``complete`` flag is true only if **every** candidate was
simulated by a real MD engine for at least ``REQUIRED_MD_NS`` — so a short validation run can never be mistaken
for the 100 ns requirement. The report states the force-field/solvent model and its limitations verbatim.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

from ..featurization import parse_smiles
from .engines import CondaOpenMMEngine, MDEngineUnavailable
from .validation import REQUIRED_MD_NS, analyze_trajectory

QUERY = "lipid nanocarrier for tumour targeting, size between 80 and 120 nanometers"
SAGE_ELEMENTS = {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}


def pick_candidates(model_dir: str, n: int, seed: int = 0) -> List[str]:
    """Top pipeline candidates whose elements Sage can parametrize (more are designed than needed)."""
    from ..pipeline import DesignPipeline
    from ..training import ModelBundle

    pipeline = DesignPipeline(ModelBundle.load(model_dir), seed=seed)
    result = pipeline.design(QUERY, n_generate=600, n_pareto=15, n_final=min(10, max(n, 5)),
                             optimize_iterations=40, run_md=False, explain=False)
    chosen = []
    for candidate in result.final_candidates:
        mol = parse_smiles(candidate["smiles"])
        if mol is not None and {a.GetSymbol() for a in mol.GetAtoms()} <= SAGE_ELEMENTS:
            chosen.append(candidate["smiles"])
    return chosen[:n]


def run_campaign(model_dir: str, n: int, ns: float, platform: str, out: str, python: Optional[str] = None,
                 smiles: Optional[List[str]] = None) -> dict:
    engine = CondaOpenMMEngine(python=python, platform=platform)
    if not engine.available():
        raise MDEngineUnavailable("set IPIND_MD_PYTHON to the ipind-md environment's python")
    candidates = smiles or pick_candidates(model_dir, n)
    started = time.time()
    rows, failures = [], []
    for smi in candidates:
        t0 = time.time()
        try:
            trajectory = engine.simulate(smi, duration_ns=ns)
        except (ValueError, RuntimeError) as exc:
            failures.append({"smiles": smi, "error": str(exc)[:300]})
            continue
        result = analyze_trajectory(trajectory)
        meta = trajectory.metadata
        rows.append({
            "smiles": smi, "n_atoms": meta["n_atoms"], "simulated_ns": result.simulated_ns, "n_frames": result.n_frames,
            "rg_mean_A": result.rg_mean, "rg_std_A": result.rg_std, "sasa_mean_A2": result.sasa_mean,
            "order_parameter": result.order_parameter, "potential_energy_mean_kcal": result.energy_mean_kcal,
            "stable": result.stable, "meets_100ns": result.meets_sim_time, "ns_per_day": meta["ns_per_day"],
            "wall_seconds": round(time.time() - t0, 1), "platform": meta["platform"],
        })
    complete = bool(rows) and not failures and all(r["meets_100ns"] for r in rows)
    report = {
        "complete": complete, "required_ns": REQUIRED_MD_NS, "requested_ns": ns,
        "model": "OpenFF 2.2.0 (Sage) + Gasteiger charges + OBC2 implicit solvent; single solute; Langevin 310 K, 2 fs",
        "limitations": [
            "single molecule in implicit solvent: no self-assembly, no bilayer/particle, no binding free energy",
            "Gasteiger charges instead of AM1-BCC (AmberTools unavailable on Windows)",
            "CHARMM36/OPLS-AA are not used; no MM-GBSA",
        ],
        "candidates": rows, "failures": failures, "total_wall_seconds": round(time.time() - started, 1),
        "host": {"cpu_count": os.cpu_count()},
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="IPIND² real-MD validation campaign")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--ns", type=float, default=2.0)
    parser.add_argument("--platform", default="OpenCL")
    parser.add_argument("--out", default="docs/md_validation.json")
    parser.add_argument("--smiles", nargs="*", default=None, help="skip the design step and simulate these")
    args = parser.parse_args()
    report = run_campaign(args.model_dir, args.n, args.ns, args.platform, args.out, smiles=args.smiles)
    print(json.dumps({k: report[k] for k in ("complete", "requested_ns", "total_wall_seconds")}, indent=2))
    for row in report["candidates"]:
        print(f"{row['smiles'][:50]:50s} atoms={row['n_atoms']:3d} {row['simulated_ns']}ns Rg={row['rg_mean_A']:.2f}Å "
              f"{row['ns_per_day']:.0f} ns/day stable={row['stable']}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
