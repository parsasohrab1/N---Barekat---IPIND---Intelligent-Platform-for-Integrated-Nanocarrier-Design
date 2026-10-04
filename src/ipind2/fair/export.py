"""ساخت، اعتبارسنجی، خروجی و ورودی رکوردهای FAIR/MIRIBEL (FR-13)."""

import csv
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .schema import (
    MIRIBEL_FIELDS,
    UNITS,
    NanoparticleRecord,
    Observation,
)

CONTEXT = {
    "@vocab": "https://schema.org/",
    "ipind": "https://ipind2.example.org/ns#",  # جای‌نگهدار: قبل از انتشار به دامنه واقعی تغییر دهید
    "qudt": "http://qudt.org/schema/qudt/",
    "prov": "http://www.w3.org/ns/prov#",
    "unit": {"@id": "qudt:unit", "@type": "@id"},
    "value": "qudt:value",
}

# نگاشت نام فیلد پیش‌بین → (دسته، فیلد MIRIBEL)
PREDICTION_FIELD_MAP = {
    "phys_size_nm": ("material", "hydrodynamic_size_nm"),
    "phys_zeta_potential_mV": ("material", "zeta_potential_mV"),
    "phys_pdi": ("material", "pdi"),
    "phys_drug_loading_efficiency_percent": ("material", "drug_loading_efficiency_percent"),
    "bio_cytotoxicity_ic50_ug_ml": ("biological", "cytotoxicity_ic50_ug_ml"),
    "bio_cellular_uptake_efficiency_percent": ("biological", "cellular_uptake_efficiency_percent"),
}
EXPERIMENT_FIELD_MAP = {
    "experimental_size_nm": ("material", "hydrodynamic_size_nm"),
    "experimental_zeta_potential": ("material", "zeta_potential_mV"),
    "experimental_loading_efficiency": ("material", "drug_loading_efficiency_percent"),
    "experimental_cytotoxicity": ("biological", "cytotoxicity_ic50_ug_ml"),
}


def new_identifier() -> str:
    return f"urn:uuid:{uuid.uuid4()}"


def record_from_candidate(
    candidate: Mapping[str, Any],
    model_version: str,
    scaffold_type: Optional[str] = None,
    license: str = "CC-BY-4.0",
) -> NanoparticleRecord:
    """رکورد از یک کاندیدای ``DesignResult.final_candidates`` (همه مقادیر predicted)."""
    record = NanoparticleRecord(
        identifier=new_identifier(),
        license=license,
        created=datetime.now(timezone.utc).isoformat(),
    )
    record.material["smiles"] = Observation(candidate["smiles"], "predicted", "design-pipeline", model_version)
    if scaffold_type:
        record.material["scaffold_type"] = Observation(scaffold_type, "predicted", "design-pipeline", model_version)
    confidence = candidate.get("confidence", {})
    for column, value in candidate.get("predictions", {}).items():
        mapped = PREDICTION_FIELD_MAP.get(column)
        if mapped is None:
            continue
        category, name = mapped
        observation = Observation(
            float(value),
            "predicted",
            method="ensemble-gnn" if category == "material" else "ensemble-graph-transformer",
            model_version=model_version,
        )
        getattr(record, category)[name] = observation
        if column in confidence:
            record.notes.append(f"confidence[{name}]={confidence[column]:.3f}")
    return record


def attach_measurements(
    record: NanoparticleRecord,
    experiment: Mapping[str, Any],
    method: str = "DLS",
) -> NanoparticleRecord:
    """افزودن نتایج آزمایشگاهی (ستون‌های ``experimental_results``) به رکورد."""
    for column, (category, name) in EXPERIMENT_FIELD_MAP.items():
        value = experiment.get(column)
        if value is not None and value == value:  # حذف NaN
            getattr(record, category)[name] = Observation(float(value), "measured", method=method)
    if experiment.get("lab_technician"):
        record.protocol["operator"] = Observation(experiment["lab_technician"], "measured")
    if experiment.get("experimental_date"):
        record.protocol["date"] = Observation(str(experiment["experimental_date"]), "measured")
    return record


@dataclass
class CompletenessReport:
    score: float  # ۰..۱: سهم فیلدهای MIRIBEL پر
    missing: Dict[str, List[str]]
    predicted_only: List[str]  # فیلدهایی که فقط پیش‌بینی دارند (بدون اندازه‌گیری)

    @property
    def complete(self) -> bool:
        return not any(self.missing.values())


def validate_record(record: NanoparticleRecord) -> CompletenessReport:
    """بررسی کامل‌بودن رکورد نسبت به فیلدهای سه دسته MIRIBEL."""
    missing: Dict[str, List[str]] = {}
    filled = total = 0
    predicted_only: List[str] = []
    for category, fields in MIRIBEL_FIELDS.items():
        section = getattr(record, category)
        missing[category] = [f for f in fields if f not in section]
        filled += len(fields) - len(missing[category])
        total += len(fields)
        predicted_only += [f for f, obs in section.items() if obs.provenance == "predicted"]
    return CompletenessReport(score=filled / total, missing=missing, predicted_only=sorted(predicted_only))


def _observation_to_jsonld(name: str, obs: Observation) -> Dict[str, Any]:
    node: Dict[str, Any] = {
        "@type": "ipind:Observation",
        "name": name,
        "value": obs.value,
        "prov:wasGeneratedBy": {"@type": "prov:Activity", "name": obs.method, "ipind:provenance": obs.provenance},
    }
    if name in UNITS:
        node["unit"] = UNITS[name]
    if obs.model_version:
        node["ipind:modelVersion"] = obs.model_version
    if obs.uncertainty is not None:
        node["ipind:uncertainty"] = obs.uncertainty
    return node


def to_jsonld(record: NanoparticleRecord) -> Dict[str, Any]:
    """رکورد JSON-LD (Interoperable: واژگان schema.org/QUDT/PROV)."""
    report = validate_record(record)
    return {
        "@context": CONTEXT,
        "@id": record.identifier,
        "@type": ["Dataset", "ipind:NanoparticleRecord"],
        "license": record.license,
        "dateCreated": record.created,
        "ipind:miribelCompleteness": round(report.score, 3),
        "ipind:miribelMissing": report.missing,
        "ipind:material": [_observation_to_jsonld(n, o) for n, o in record.material.items()],
        "ipind:biological": [_observation_to_jsonld(n, o) for n, o in record.biological.items()],
        "ipind:protocol": [_observation_to_jsonld(n, o) for n, o in record.protocol.items()],
        "description": "; ".join(record.notes) if record.notes else None,
    }


def export_records(records: Sequence[NanoparticleRecord], path: str, fmt: str = "jsonld") -> Path:
    """نوشتن رکوردها: ``jsonld`` (یک آرایه ``@graph``) یا ``csv`` (تخت، با ستون provenance)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonld":
        graph = []
        for record in records:
            node = to_jsonld(record)
            node.pop("@context")
            graph.append(node)
        out.write_text(
            json.dumps({"@context": CONTEXT, "@graph": graph}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif fmt == "csv":
        rows = []
        for record in records:
            for category in MIRIBEL_FIELDS:
                for name, obs in getattr(record, category).items():
                    rows.append(
                        {
                            "identifier": record.identifier,
                            "category": category,
                            "field": name,
                            "value": obs.value,
                            "unit": UNITS.get(name, ""),
                            "provenance": obs.provenance,
                            "method": obs.method or "",
                            "model_version": obs.model_version or "",
                            "license": record.license,
                        }
                    )
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["identifier", "category", "field", "value", "unit", "provenance", "method", "model_version", "license"],
            )
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError("fmt باید jsonld یا csv باشد")
    return out


def load_jsonld(path: str) -> List[NanoparticleRecord]:
    """بازخوانی خروجی ``export_records(..., 'jsonld')`` (تعامل‌پذیری رفت‌وبرگشتی)."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    records: List[NanoparticleRecord] = []
    for node in document.get("@graph", []):
        record = NanoparticleRecord(
            identifier=node["@id"], license=node.get("license", ""), created=node.get("dateCreated")
        )
        for category in ("material", "biological", "protocol"):
            section = getattr(record, category)
            for item in node.get(f"ipind:{category}", []):
                activity = item.get("prov:wasGeneratedBy", {})
                section[item["name"]] = Observation(
                    value=item["value"],
                    provenance=activity.get("ipind:provenance", "measured"),
                    method=activity.get("name"),
                    model_version=item.get("ipind:modelVersion"),
                    uncertainty=item.get("ipind:uncertainty"),
                )
        records.append(record)
    return records
