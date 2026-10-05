"""
Natural-Language Query Interface

A rule-based, fully offline parser that extracts nanocarrier design target parameters
from free Persian/English text and converts them to a structured ``TargetParameters``
(input of the structure generation unit, FR-01). It works without dependence on any external service/LLM
so that its behavior is deterministic and offline.

It is designed so that an LLM-based ``QueryParser`` (e.g., for understanding more
complex phrases) can later be substituted/added without changing the calling code — see the
``QueryParser`` protocol at the bottom of this file.

See docs/SRS.md §4.8 (FR-10).
"""

import re
from typing import Dict, List, Optional, Protocol

from .schema import TargetParameters

_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

_SCAFFOLD_KEYWORDS: Dict[str, List[str]] = {
    "lipid": ["لیپیدی", "لیپید", "lipid", "lnp"],
    "polymer": ["پلیمری", "پلیمر", "polymer", "plga", "peg-pla", "chitosan", "کیتوزان"],
    "metal": ["فلزی", "فلز", "metal", "طلا", "gold", "سیلیکا", "silica", "مزوپروس"],
}

_TISSUE_KEYWORDS: Dict[str, List[str]] = {
    "liver": ["کبد", "liver"],
    "lung": ["ریه", "lung", "pulmonary"],
    "tumor": ["تومور", "سرطان", "tumor", "cancer"],
    "brain": ["مغز", "brain", "cns"],
    "spleen": ["طحال", "spleen"],
    "muscle": ["عضله", "muscle"],
}

_SIZE_RANGE_RE = re.compile(
    r"(?:بین|between)?\s*(\d+(?:\.\d+)?)\s*(?:تا|to|-)\s*(\d+(?:\.\d+)?)\s*(?:نانومتر|نانومتری|nm)",
    re.IGNORECASE,
)
_SIZE_MAX_RE = re.compile(
    r"(?:زیر|کمتر از|under|below|less than)\s*(\d+(?:\.\d+)?)\s*(?:نانومتر|نانومتری|nm)",
    re.IGNORECASE,
)
_TOXICITY_RE = re.compile(
    r"(?:سمیت|toxicity|ic50)[^\d]{0,12}?(?:زیر|کمتر از|<|under|below)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_LOADING_RE = re.compile(
    r"(?:کارایی بارگذاری|loading efficiency)[^\d]{0,12}?(?:بالای|بیشتر از|>|above|over)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return text.translate(_PERSIAN_DIGITS)


def _find_keyword_match(text: str, keyword_map: Dict[str, List[str]]) -> Optional[str]:
    lowered = text.lower()
    for label, keywords in keyword_map.items():
        if any(kw.lower() in lowered for kw in keywords):
            return label
    return None


def parse_query(text: str) -> TargetParameters:
    """
    Extract ``TargetParameters`` from a free Persian/English query.

    Example:
        >>> p = parse_query("یک نانوحامل لیپیدی برای هدف‌گیری تومور، اندازه بین ۸۰ تا ۱۲۰ نانومتر")
        >>> p.scaffold_type, p.target_tissue, p.size_range_nm
        ('lipid', 'tumor', (80.0, 120.0))
    """
    normalized = _normalize(text)
    params = TargetParameters(raw_query=text)

    params.scaffold_type = _find_keyword_match(normalized, _SCAFFOLD_KEYWORDS)
    params.target_tissue = _find_keyword_match(normalized, _TISSUE_KEYWORDS)

    range_match = _SIZE_RANGE_RE.search(normalized)
    if range_match:
        params.size_range_nm = (float(range_match.group(1)), float(range_match.group(2)))
    else:
        max_match = _SIZE_MAX_RE.search(normalized)
        if max_match:
            params.size_range_nm = (0.0, float(max_match.group(1)))

    toxicity_match = _TOXICITY_RE.search(normalized)
    if toxicity_match:
        params.max_toxicity_ic50 = float(toxicity_match.group(1))

    loading_match = _LOADING_RE.search(normalized)
    if loading_match:
        params.min_loading_efficiency = float(loading_match.group(1))

    if params.scaffold_type is None:
        params.unresolved_terms.append("scaffold_type")
    if params.target_tissue is None:
        params.unresolved_terms.append("target_tissue")

    return params


class QueryParser(Protocol):
    """Public interface of the query parser — for easy replacement with an LLM-based implementation."""

    def parse(self, text: str) -> TargetParameters: ...


class RuleBasedQueryParser:
    """Default QueryParser implementation using offline rules/regex."""

    def parse(self, text: str) -> TargetParameters:
        return parse_query(text)
