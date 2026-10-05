"""HTML report builder for design results (FR-07: "reporter"). All inserted values are escaped."""

from html import escape
from typing import Any, Dict, List, Mapping

_LABELS = {
    "phys_size_nm": "Size (nm)",
    "phys_zeta_potential_mV": "Zeta potential (mV)",
    "phys_pdi": "PDI",
    "phys_colloidal_stability_hours": "Colloidal stability (h)",
    "phys_drug_loading_efficiency_percent": "Loading efficiency (%)",
    "phys_drug_loading_content_percent": "Loading content (wt%)",
    "phys_release_rate_constant": "Release rate constant",
    "bio_cytotoxicity_ic50_ug_ml": "IC50 toxicity (µg/mL)",
    "bio_cytotoxicity_ic50_hepg2_ug_ml": "IC50 HepG2",
    "bio_cytotoxicity_ic50_hela_ug_ml": "IC50 HeLa",
    "bio_cellular_uptake_efficiency_percent": "Cellular uptake (%)",
    "bio_serum_protein_binding_percent": "Serum protein binding (%)",
    "bio_circulation_half_life_hours": "Circulation half-life (h)",
    "bio_tumor_to_background_ratio": "TBR",
}

_STYLE = (
    "body{font-family:Tahoma,Arial,sans-serif;margin:2rem;direction:rtl;color:#1a1a1a}"
    "table{border-collapse:collapse;margin:1rem 0;width:100%}"
    "th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:right}"
    "th{background:#f0f4f8}code{direction:ltr;unicode-bidi:embed;font-size:.85em;word-break:break-all}"
    ".warn{background:#fff4d6;border:1px solid #e0b84a;padding:.6rem;margin:.4rem 0}"
    ".meta{color:#555}"
)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
    return str(value)


def render_report(result: Mapping[str, Any], title: str = "Nanocarrier Design Report — IPIND²") -> str:
    """Standalone HTML (no script/external resource) from the ``DesignResult.to_dict()`` output."""
    parts: List[str] = [
        "<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'>",
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head><body>",
        f"<h1>{escape(title)}</h1>",
    ]
    query = result.get("query")
    if query:
        parts.append(f"<p class='meta'>Query: {escape(str(query))}</p>")
    params = result.get("parameters", {})
    if params:
        rows = "".join(
            f"<tr><td>{escape(str(k))}</td><td>{escape(_fmt(v))}</td></tr>" for k, v in params.items() if v not in (None, [], ())
        )
        parts.append(f"<h2>Target parameters</h2><table>{rows}</table>")

    for warning in result.get("warnings", []):
        parts.append(f"<div class='warn'>⚠ {escape(str(warning))}</div>")
    if not result.get("md_complete", False):
        parts.append(
            "<div class='warn'>Full MD simulation validation (≥100 ns with a real MD engine) was not performed; "
            "simulation results are screening only.</div>"
        )

    candidates: List[Dict[str, Any]] = result.get("final_candidates", [])
    parts.append(f"<h2>Final candidates ({len(candidates)})</h2>")
    for candidate in candidates:
        parts.append(f"<h3>Rank {escape(str(candidate.get('rank')))}</h3>")
        parts.append(f"<p><code>{escape(str(candidate.get('smiles')))}</code></p>")
        confidence = candidate.get("overall_confidence")
        stable = candidate.get("md_stable")
        parts.append(
            "<p class='meta'>Source: {src} · Overall confidence: {conf} · Simulation stability: {stable}</p>".format(
                src=escape(str(candidate.get("source"))),
                conf=escape(_fmt(confidence)) if confidence is not None else "—",
                stable="—" if stable is None else ("Stable" if stable else "Unstable"),
            )
        )
        predictions = candidate.get("predictions", {})
        conf_map = candidate.get("confidence", {})
        rows = "".join(
            f"<tr><td>{escape(_LABELS.get(k, k))}</td><td>{escape(_fmt(v))}</td>"
            f"<td>{escape(_fmt(conf_map[k])) if k in conf_map else '—'}</td></tr>"
            for k, v in predictions.items()
        )
        parts.append(f"<table><tr><th>Property (prediction)</th><th>Value</th><th>Confidence</th></tr>{rows}</table>")
        for explanation in candidate.get("explanations", []):
            features = ", ".join(
                f"{escape(str(f['feature']))} ({f['shap_value']:+.2f})" for f in explanation.get("top_features", [])
            )
            parts.append(
                f"<p class='meta'>Interpretation of {escape(_LABELS.get(explanation['target'], explanation['target']))} "
                f"(surrogate fidelity {explanation['fidelity']:.2f}): {features}</p>"
            )

    stats = result.get("stats", {})
    if stats:
        parts.append(f"<h2>Run statistics</h2><p class='meta'>Total time: {escape(_fmt(stats.get('total_seconds', 0)))} s</p>")
    parts.append("</body></html>")
    return "".join(parts)
