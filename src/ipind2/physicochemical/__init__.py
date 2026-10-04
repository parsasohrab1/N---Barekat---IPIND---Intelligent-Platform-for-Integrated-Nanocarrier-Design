"""Unit 2: physicochemical property prediction (Multi-Task GNN). See docs/SRS.md §4.2 (FR-02)."""

from .gnn import PHYSICO_TARGET_COLUMNS, MultiTaskGNN, PhysicochemicalPredictor

__all__ = ["PHYSICO_TARGET_COLUMNS", "MultiTaskGNN", "PhysicochemicalPredictor"]
