"""
ORM models of the data layer (FR-08) — SQLAlchemy 2.0; compatible with PostgreSQL (production) and SQLite (development/test).

The first four tables are column-for-column identical to ``sql/schema.sql`` (only ``canonical_smiles`` and
``inchikey`` were added for deduplication); the other tables cover SEC-01/04/05, model version
tracking (FR-12) and asynchronous jobs.
"""

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Molecule(Base):
    __tablename__ = "molecules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    smiles: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_smiles: Mapped[Optional[str]] = mapped_column(String(2048), unique=True, index=True)
    inchikey: Mapped[Optional[str]] = mapped_column(String(27), index=True)
    molecular_weight: Mapped[Optional[float]] = mapped_column(Float)
    logP: Mapped[Optional[float]] = mapped_column(Float)
    tpsa: Mapped[Optional[float]] = mapped_column(Float)
    num_rotatable_bonds: Mapped[Optional[int]] = mapped_column(Integer)
    num_h_donors: Mapped[Optional[int]] = mapped_column(Integer)
    num_h_acceptors: Mapped[Optional[int]] = mapped_column(Integer)
    scaffold_type: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    physicochemical: Mapped[List["PhysicochemicalProperty"]] = relationship(back_populates="molecule")
    biological: Mapped[List["BiologicalProperty"]] = relationship(back_populates="molecule")
    experiments: Mapped[List["ExperimentalResult"]] = relationship(back_populates="molecule")


class PhysicochemicalProperty(Base):
    __tablename__ = "physicochemical_properties"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    molecule_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), index=True)
    size_nm: Mapped[Optional[float]] = mapped_column(Float)
    zeta_potential_mV: Mapped[Optional[float]] = mapped_column(Float)
    pdi: Mapped[Optional[float]] = mapped_column(Float)
    colloid_stability_hours: Mapped[Optional[float]] = mapped_column(Float)
    drug_loading_efficiency: Mapped[Optional[float]] = mapped_column(Float)
    drug_loading_content: Mapped[Optional[float]] = mapped_column(Float)
    release_rate_constant: Mapped[Optional[float]] = mapped_column(Float)
    prediction_confidence: Mapped[Optional[float]] = mapped_column(Float)
    model_version: Mapped[Optional[str]] = mapped_column(String(64))

    molecule: Mapped[Molecule] = relationship(back_populates="physicochemical")


class BiologicalProperty(Base):
    __tablename__ = "biological_properties"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    molecule_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), index=True)
    cell_line: Mapped[Optional[str]] = mapped_column(String(50))
    cytotoxicity_ic50: Mapped[Optional[float]] = mapped_column(Float)
    cellular_uptake_efficiency: Mapped[Optional[float]] = mapped_column(Float)
    serum_protein_binding: Mapped[Optional[float]] = mapped_column(Float)
    circulation_half_life: Mapped[Optional[float]] = mapped_column(Float)
    tumor_to_background_ratio: Mapped[Optional[float]] = mapped_column(Float)
    model_version: Mapped[Optional[str]] = mapped_column(String(64))

    molecule: Mapped[Molecule] = relationship(back_populates="biological")


class ExperimentalResult(Base):
    __tablename__ = "experimental_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    molecule_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), index=True)
    experimental_size_nm: Mapped[Optional[float]] = mapped_column(Float)
    experimental_zeta_potential: Mapped[Optional[float]] = mapped_column(Float)
    experimental_loading_efficiency: Mapped[Optional[float]] = mapped_column(Float)
    experimental_cytotoxicity: Mapped[Optional[float]] = mapped_column(Float)
    experimental_date: Mapped[Optional[date]] = mapped_column(Date)
    lab_technician: Mapped[Optional[str]] = mapped_column(String(100))
    consumed_by_training: Mapped[bool] = mapped_column(Boolean, default=False)

    molecule: Mapped[Molecule] = relationship(back_populates="experiments")


class User(Base):
    """Platform user (SEC-01 two-factor authentication, SEC-05 roles)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="viewer")  # admin | researcher | viewer
    totp_secret_encrypted: Mapped[Optional[str]] = mapped_column(Text)
    totp_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    failed_logins: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AuditLog(Base):
    """Log of all user activities (SEC-04)."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    resource: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(16), default="ok")
    detail: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    ip_address: Mapped[Optional[str]] = mapped_column(String(64))


class Job(Base):
    """Asynchronous job (generation/optimization/validation)."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    owner: Mapped[Optional[str]] = mapped_column(String(64))
    parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class ModelVersion(Base):
    """Model version registry (basis of previous/new version comparison in FR-12)."""

    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(64))
    path: Mapped[Optional[str]] = mapped_column(String(512))
    metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    trained_on: Mapped[Optional[str]] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String(64), index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    n_samples: Mapped[int] = mapped_column(Integer)
    rmse: Mapped[float] = mapped_column(Float)
    r2: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
