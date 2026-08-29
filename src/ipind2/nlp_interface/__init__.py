"""Natural-language query interface: parses target parameters from free text into structured input for the generation unit. See docs/SRS.md §4.8 (FR-10)."""

from .parser import QueryParser, RuleBasedQueryParser, parse_query
from .schema import TargetParameters

__all__ = ["TargetParameters", "parse_query", "QueryParser", "RuleBasedQueryParser"]
