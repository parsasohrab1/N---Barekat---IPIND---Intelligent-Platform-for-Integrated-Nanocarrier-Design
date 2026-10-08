"""
Standalone OpenMM MD runner — executed *inside the MD conda environment* (``ipind-md``), not the main env.

    python openmm_runner.py --smiles "CCCC..." --ns 1.0 --out traj.npz [--platform CPU|OpenCL]

Model (be explicit about what this is):
  * force field: OpenFF 2.2.0 "Sage" (SMIRNOFF) applied through OpenFF Interchange;
  * partial charges: RDKit **Gasteiger** (AM1-BCC needs AmberTools, which has no Windows build) —
    cruder than the usual AM1-BCC; electrostatics-sensitive results (zwitterions, quats) are less reliable;
  * solvent: **implicit** OBC2 generalized Born (element-based radii/scale, mbondi2-like), not explicit water;
  * single solute molecule: this is a conformational/dynamics test of one lipid or polymer chain, **not**
    a self-assembled nanoparticle or a binding free-energy calculation;
  * Langevin (middle) integrator, 310 K, 2 fs, H-bond constraints.

Unsupported elements (e.g. Au, Fe, Si, Zn) are rejected: Sage has no parameters for them.
"""

import argparse
import json
import sys
import time

import numpy as np

SUPPORTED = {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
# mbondi2-like radii (nm) and OBC2 screening scale factors by element
GB_RADIUS_NM = {"H": 0.12, "C": 0.17, "N": 0.155, "O": 0.15, "S": 0.18, "P": 0.185, "F": 0.15, "Cl": 0.17, "Br": 0.185, "I": 0.198}
GB_SCALE = {"H": 0.85, "C": 0.72, "N": 0.79, "O": 0.85, "S": 0.96, "P": 0.86, "F": 0.88, "Cl": 0.80, "Br": 0.80, "I": 0.80}


def build_system(smiles: str):
    import openmm
    from openff.interchange import Interchange
    from openff.toolkit import ForceField, Molecule
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    bad = sorted({a.GetSymbol() for a in mol.GetAtoms()} - SUPPORTED)
    if bad:
        raise ValueError(f"elements without Sage parameters: {bad}")

    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    molecule.generate_conformers(n_conformers=1)
    molecule.assign_partial_charges("gasteiger")
    force_field = ForceField("openff-2.2.0.offxml")
    interchange = Interchange.from_smirnoff(force_field, [molecule], charge_from_molecules=[molecule])
    system = interchange.to_openmm(combine_nonbonded_forces=True)
    positions = interchange.positions.to_openmm()

    # add implicit solvent (OBC2) using the nonbonded charges
    nonbonded = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    gb = openmm.GBSAOBCForce()
    gb.setSolventDielectric(78.5)
    gb.setSoluteDielectric(1.0)
    elements = [a.symbol for a in molecule.atoms]
    for index, element in enumerate(elements):
        charge, _, _ = nonbonded.getParticleParameters(index)
        gb.addParticle(charge, GB_RADIUS_NM[element], GB_SCALE[element])
    system.addForce(gb)
    bonds = [(b.atom1_index, b.atom2_index) for b in molecule.bonds]
    return system, positions, molecule, elements, bonds


def run(smiles: str, ns: float, out: str, platform_name: str, report_ps: float, equil_ps: float, seed: int):
    import openmm
    from openmm import unit

    system, positions, molecule, elements, bonds = build_system(smiles)
    integrator = openmm.LangevinMiddleIntegrator(310 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    integrator.setRandomNumberSeed(seed)
    platform = openmm.Platform.getPlatformByName(platform_name)

    # Simulation needs an app.Topology; build a minimal one with matching atom count
    topology = openmm.app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("MOL", chain)
    atoms = [topology.addAtom(e, openmm.app.Element.getBySymbol(e), residue) for e in elements]
    for i, j in bonds:
        topology.addBond(atoms[i], atoms[j])
    simulation = openmm.app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    start_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    simulation.minimizeEnergy(maxIterations=2000)
    minimized = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    simulation.context.setVelocitiesToTemperature(310 * unit.kelvin, seed)

    dt_ps = 0.002
    simulation.step(int(equil_ps / dt_ps))

    steps_total = int(round(ns * 1000.0 / dt_ps))
    steps_per_frame = max(int(round(report_ps / dt_ps)), 1)
    n_frames = max(steps_total // steps_per_frame, 1)
    frames, energies, temps = [], [], []
    started = time.perf_counter()
    for _ in range(n_frames):
        simulation.step(steps_per_frame)
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        frames.append(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
        energies.append(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
    wall = time.perf_counter() - started
    simulated_ns = n_frames * steps_per_frame * dt_ps / 1000.0

    coords = np.asarray(frames, dtype=np.float32)
    finite = bool(np.isfinite(coords).all() and np.isfinite(energies).all())
    meta = {
        "engine": "openmm", "version": openmm.__version__, "platform": platform_name,
        "forcefield": "openff-2.2.0 (Sage) + Gasteiger charges + OBC2 implicit solvent",
        "temperature_k": 310, "dt_fs": 2, "equilibration_ps": equil_ps, "frame_interval_ps": report_ps,
        "simulated_ns": simulated_ns, "wall_seconds": wall,
        "ns_per_day": simulated_ns / wall * 86400.0 if wall > 0 else None,
        "n_atoms": len(elements), "energy_start_kcal": start_energy, "energy_minimized_kcal": minimized,
        "finite": finite, "seed": seed,
    }
    np.savez_compressed(out, coords=coords, energies=np.asarray(energies), elements=np.asarray(elements),
                        bonds=np.asarray(bonds, dtype=np.int32), meta=json.dumps(meta))
    print(json.dumps(meta))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--ns", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--platform", default="CPU")
    parser.add_argument("--report-ps", type=float, default=10.0)
    parser.add_argument("--equil-ps", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    try:
        run(args.smiles, args.ns, args.out, args.platform, args.report_ps, args.equil_ps, args.seed)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
