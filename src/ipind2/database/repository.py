"""عملیات داده سطح بالا (FR-08): ذخیره ساختارها، پیش‌بینی‌ها و نتایج آزمایشگاهی."""

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..featurization import canonical_smiles, descriptor_dict, inchikey, parse_smiles
from .models import (
    AuditLog,
    BiologicalProperty,
    ExperimentalResult,
    Job,
    Molecule,
    PhysicochemicalProperty,
)

# نگاشت ستون‌های پیش‌بین به ستون‌های جدول
PHYSICO_COLUMN_MAP = {
    "phys_size_nm": "size_nm",
    "phys_zeta_potential_mV": "zeta_potential_mV",
    "phys_pdi": "pdi",
    "phys_colloidal_stability_hours": "colloid_stability_hours",
    "phys_drug_loading_efficiency_percent": "drug_loading_efficiency",
    "phys_drug_loading_content_percent": "drug_loading_content",
    "phys_release_rate_constant": "release_rate_constant",
}
BIO_COLUMN_MAP = {
    "bio_cytotoxicity_ic50_ug_ml": "cytotoxicity_ic50",
    "bio_cellular_uptake_efficiency_percent": "cellular_uptake_efficiency",
    "bio_serum_protein_binding_percent": "serum_protein_binding",
    "bio_circulation_half_life_hours": "circulation_half_life",
    "bio_tumor_to_background_ratio": "tumor_to_background_ratio",
}


class MoleculeRepository:
    """دسترسی به ساختارها و خواص آن‌ها روی یک ``Session``."""

    def __init__(self, session: Session):
        self.session = session

    def add_molecule(self, smiles: str, scaffold_type: Optional[str] = None) -> Optional[Molecule]:
        """درج مولکول (یکتا بر اساس SMILES کانونیک). ``None`` اگر SMILES نامعتبر باشد."""
        mol = parse_smiles(smiles)
        if mol is None:
            return None
        canonical = canonical_smiles(smiles)
        existing = self.session.scalar(select(Molecule).where(Molecule.canonical_smiles == canonical))
        if existing is not None:
            return existing
        descriptors = descriptor_dict(mol)
        molecule = Molecule(
            smiles=smiles,
            canonical_smiles=canonical,
            inchikey=inchikey(smiles),
            molecular_weight=descriptors["mol_weight"],
            logP=descriptors["logP"],
            tpsa=descriptors["tpsa"],
            num_rotatable_bonds=int(descriptors["num_rotatable_bonds"]),
            num_h_donors=int(descriptors["num_h_donors"]),
            num_h_acceptors=int(descriptors["num_h_acceptors"]),
            scaffold_type=scaffold_type,
        )
        self.session.add(molecule)
        self.session.flush()
        return molecule

    def add_molecules(self, smiles_list: Iterable[str], scaffold_type: Optional[str] = None) -> List[int]:
        """درج دسته‌ای؛ شناسه مولکول‌های (جدید یا موجود) معتبر را برمی‌گرداند."""
        ids = []
        for smiles in smiles_list:
            molecule = self.add_molecule(smiles, scaffold_type)
            if molecule is not None:
                ids.append(molecule.id)
        return ids

    def smiles_by_id(self) -> Dict[int, str]:
        """نگاشت ``molecule_id → smiles`` برای ``ActiveLearningLoop``."""
        return {row.id: row.smiles for row in self.session.scalars(select(Molecule))}

    def count(self) -> int:
        return int(self.session.scalar(select(func.count(Molecule.id))) or 0)

    def store_predictions(
        self,
        molecule_id: int,
        physico: Optional[Mapping[str, float]] = None,
        bio: Optional[Mapping[str, float]] = None,
        confidence: Optional[float] = None,
        model_version: Optional[str] = None,
        cell_line: Optional[str] = None,
    ) -> None:
        """ذخیره خروجی پیش‌بین‌ها (کلیدها با نام ستون‌های پیش‌بین)."""
        if physico:
            values = {PHYSICO_COLUMN_MAP[k]: float(v) for k, v in physico.items() if k in PHYSICO_COLUMN_MAP}
            self.session.add(
                PhysicochemicalProperty(
                    molecule_id=molecule_id,
                    prediction_confidence=confidence,
                    model_version=model_version,
                    **values,
                )
            )
        if bio:
            values = {BIO_COLUMN_MAP[k]: float(v) for k, v in bio.items() if k in BIO_COLUMN_MAP}
            self.session.add(
                BiologicalProperty(
                    molecule_id=molecule_id, cell_line=cell_line, model_version=model_version, **values
                )
            )

    def add_experimental_results(self, results: pd.DataFrame) -> int:
        """
        درج نتایج آزمایشگاهی (خروجی ``lab_automation.ingest_results``).

        ردیف‌هایی که ``molecule_id`` آن‌ها در پایگاه داده نیست رد می‌شوند.
        Returns: تعداد درج‌شده.
        """
        known = set(self.session.scalars(select(Molecule.id)))
        added = 0
        for _, row in results.iterrows():
            molecule_id = row.get("molecule_id")
            if pd.isna(molecule_id) or int(molecule_id) not in known:
                continue

            def num(key):
                value = row.get(key)
                return None if value is None or pd.isna(value) else float(value)

            experimental_date = row.get("experimental_date")
            if experimental_date is None or pd.isna(experimental_date):
                experimental_date = None
            elif isinstance(experimental_date, str):
                from datetime import date

                experimental_date = date.fromisoformat(experimental_date)

            self.session.add(
                ExperimentalResult(
                    molecule_id=int(molecule_id),
                    experimental_size_nm=num("experimental_size_nm"),
                    experimental_zeta_potential=num("experimental_zeta_potential"),
                    experimental_loading_efficiency=num("experimental_loading_efficiency"),
                    experimental_cytotoxicity=num("experimental_cytotoxicity"),
                    experimental_date=experimental_date,
                    lab_technician=row.get("lab_technician") or None,
                )
            )
            added += 1
        self.session.flush()
        return added

    def unconsumed_results(self) -> pd.DataFrame:
        """نتایج آزمایشگاهی که هنوز در به‌روزرسانی مدل مصرف نشده‌اند."""
        rows = self.session.scalars(
            select(ExperimentalResult).where(ExperimentalResult.consumed_by_training.is_(False))
        ).all()
        return pd.DataFrame(
            [
                {
                    "molecule_id": r.molecule_id,
                    "experimental_size_nm": r.experimental_size_nm,
                    "experimental_zeta_potential": r.experimental_zeta_potential,
                    "experimental_loading_efficiency": r.experimental_loading_efficiency,
                    "experimental_cytotoxicity": r.experimental_cytotoxicity,
                    "experimental_date": r.experimental_date,
                    "lab_technician": r.lab_technician,
                    "_row_id": r.id,
                }
                for r in rows
            ]
        )

    def mark_consumed(self, row_ids: Sequence[int]) -> None:
        for row_id in row_ids:
            record = self.session.get(ExperimentalResult, int(row_id))
            if record is not None:
                record.consumed_by_training = True
        self.session.flush()


class AuditRepository:
    """لاگ فعالیت کاربران (SEC-04)."""

    def __init__(self, session: Session):
        self.session = session

    def record(
        self,
        action: str,
        username: Optional[str] = None,
        resource: Optional[str] = None,
        status: str = "ok",
        detail: Optional[dict] = None,
        ip_address: Optional[str] = None,
    ) -> AuditLog:
        entry = AuditLog(
            username=username,
            action=action,
            resource=resource,
            status=status,
            detail=detail,
            ip_address=ip_address,
        )
        self.session.add(entry)
        self.session.flush()
        return entry

    def recent(self, limit: int = 100, username: Optional[str] = None) -> List[AuditLog]:
        query = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
        if username:
            query = query.where(AuditLog.username == username)
        return list(self.session.scalars(query))


class JobRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, job_id: str, kind: str, owner: Optional[str], parameters: Optional[dict]) -> Job:
        job = Job(id=job_id, kind=kind, owner=owner, parameters=parameters, status="queued")
        self.session.add(job)
        self.session.flush()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.session.get(Job, job_id)
