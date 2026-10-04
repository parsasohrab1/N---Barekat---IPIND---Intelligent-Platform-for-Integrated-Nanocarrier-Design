"""UI/API layer: interactive dashboard, report generation, parameter configuration. See docs/SRS.md §2.1 (FR-07)."""

from .app import Settings, create_app

__all__ = ["create_app", "Settings"]
