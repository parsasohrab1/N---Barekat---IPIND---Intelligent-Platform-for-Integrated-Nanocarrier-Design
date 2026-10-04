"""گزارش‌ساز HTML نتایج طراحی (FR-07: «گزارش‌گیر»). همه مقادیر درج‌شده escape می‌شوند."""

from html import escape
from typing import Any, Dict, List, Mapping

_LABELS = {
    "phys_size_nm": "اندازه (nm)",
    "phys_zeta_potential_mV": "پتانسیل زتا (mV)",
    "phys_pdi": "PDI",
    "phys_colloidal_stability_hours": "پایداری کلوئیدی (h)",
    "phys_drug_loading_efficiency_percent": "کارایی بارگذاری (٪)",
    "phys_drug_loading_content_percent": "میزان بارگذاری (٪ وزنی)",
    "phys_release_rate_constant": "ثابت نرخ رهایش",
    "bio_cytotoxicity_ic50_ug_ml": "IC50 سمیت (µg/mL)",
    "bio_cytotoxicity_ic50_hepg2_ug_ml": "IC50 HepG2",
    "bio_cytotoxicity_ic50_hela_ug_ml": "IC50 HeLa",
    "bio_cellular_uptake_efficiency_percent": "نفوذ سلولی (٪)",
    "bio_serum_protein_binding_percent": "اتصال پروتئین سرم (٪)",
    "bio_circulation_half_life_hours": "نیمه‌عمر گردش (h)",
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


def render_report(result: Mapping[str, Any], title: str = "گزارش طراحی نانوحامل — IPIND²") -> str:
    """HTML مستقل (بدون اسکریپت/منبع خارجی) از خروجی ``DesignResult.to_dict()``."""
    parts: List[str] = [
        "<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'>",
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head><body>",
        f"<h1>{escape(title)}</h1>",
    ]
    query = result.get("query")
    if query:
        parts.append(f"<p class='meta'>پرس‌وجو: {escape(str(query))}</p>")
    params = result.get("parameters", {})
    if params:
        rows = "".join(
            f"<tr><td>{escape(str(k))}</td><td>{escape(_fmt(v))}</td></tr>" for k, v in params.items() if v not in (None, [], ())
        )
        parts.append(f"<h2>پارامترهای هدف</h2><table>{rows}</table>")

    for warning in result.get("warnings", []):
        parts.append(f"<div class='warn'>⚠ {escape(str(warning))}</div>")
    if not result.get("md_complete", False):
        parts.append(
            "<div class='warn'>اعتبارسنجی شبیه‌سازی MD کامل (≥۱۰۰ ns با موتور MD واقعی) انجام نشده است؛ "
            "نتایج شبیه‌سازی فقط غربالگری‌اند.</div>"
        )

    candidates: List[Dict[str, Any]] = result.get("final_candidates", [])
    parts.append(f"<h2>کاندیداهای نهایی ({len(candidates)})</h2>")
    for candidate in candidates:
        parts.append(f"<h3>رتبه {escape(str(candidate.get('rank')))}</h3>")
        parts.append(f"<p><code>{escape(str(candidate.get('smiles')))}</code></p>")
        confidence = candidate.get("overall_confidence")
        stable = candidate.get("md_stable")
        parts.append(
            "<p class='meta'>منبع: {src} · اطمینان کلی: {conf} · پایداری شبیه‌سازی: {stable}</p>".format(
                src=escape(str(candidate.get("source"))),
                conf=escape(_fmt(confidence)) if confidence is not None else "—",
                stable="—" if stable is None else ("پایدار" if stable else "ناپایدار"),
            )
        )
        predictions = candidate.get("predictions", {})
        conf_map = candidate.get("confidence", {})
        rows = "".join(
            f"<tr><td>{escape(_LABELS.get(k, k))}</td><td>{escape(_fmt(v))}</td>"
            f"<td>{escape(_fmt(conf_map[k])) if k in conf_map else '—'}</td></tr>"
            for k, v in predictions.items()
        )
        parts.append(f"<table><tr><th>ویژگی (پیش‌بینی)</th><th>مقدار</th><th>اطمینان</th></tr>{rows}</table>")
        for explanation in candidate.get("explanations", []):
            features = "، ".join(
                f"{escape(str(f['feature']))} ({f['shap_value']:+.2f})" for f in explanation.get("top_features", [])
            )
            parts.append(
                f"<p class='meta'>تفسیر {escape(_LABELS.get(explanation['target'], explanation['target']))} "
                f"(fidelity جانشین {explanation['fidelity']:.2f}): {features}</p>"
            )

    stats = result.get("stats", {})
    if stats:
        parts.append(f"<h2>آمار اجرا</h2><p class='meta'>مدت کل: {escape(_fmt(stats.get('total_seconds', 0)))} s</p>")
    parts.append("</body></html>")
    return "".join(parts)
