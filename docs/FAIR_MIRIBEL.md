# FAIR and MIRIBEL (FR-13)

Module: [`src/ipind2/fair`](../src/ipind2/fair). API: `GET /export/{job_id}?fmt=jsonld|csv`.

## Claim status

> **Aligned with the MIRIBEL three-category classification** — not "officially certified or compliant".

The fields of each category (material, biological, protocol) are the project's reading of the published MIRIBEL checklist (Faria et al., *Nature Nanotechnology* 2018) and **must be reconciled with the official checklist by a specialist before being presented to a partner/regulator**. The vocabulary mapping (`ipind:` namespace) is a placeholder and must be changed to the organization's real domain.

## FAIR principles

| Principle | Implementation |
|---|---|
| Findable | Persistent `urn:uuid:` identifier for each record; rich JSON-LD metadata |
| Accessible | Open JSON-LD and CSV formats; no proprietary tools |
| Interoperable | schema.org / QUDT (units) / PROV-O (provenance) vocabularies |
| Reusable | `license` (default CC-BY-4.0), **`provenance` for every value**, model version |

## Value provenance (the most important decision)

Every value is either `predicted` (with mandatory `model_version`) or `measured` (with method); the `Observation` constructor rejects a predicted value without a model version, and a test proves that the `/export` output never labels any prediction as "measured". Reason: a regulator must not confuse a model prediction with a measurement.

## Completeness

`validate_record` returns the share of filled MIRIBEL fields. Records built only from predictions are naturally incomplete (no protocol, cell line, or measurement method) and `ipind:miribelMissing` lists the missing fields; filling them requires lab data (`attach_measurements`).
