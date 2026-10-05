"""Model bundle (ModelBundle): Unit 2/3 predictors + Unit 1 generator with a versioned manifest."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..biological import BiologicalPredictor
from ..generation import ConditionalStructureGenerator
from ..physicochemical import PhysicochemicalPredictor

MANIFEST = "manifest.json"


@dataclass
class ModelBundle:
    physico: PhysicochemicalPredictor
    bio: BiologicalPredictor
    generator: ConditionalStructureGenerator
    version: str = "unversioned"
    metrics: Dict[str, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.physico.save(str(path / "physico"))
        self.bio.save(str(path / "bio"))
        self.generator.save(str(path / "generator"))
        manifest = {
            "version": self.version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics": self.metrics,
            "info": self.info,
        }
        (path / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str) -> "ModelBundle":
        path = Path(directory)
        manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
        return cls(
            physico=PhysicochemicalPredictor.load(str(path / "physico")),
            bio=BiologicalPredictor.load(str(path / "bio")),
            generator=ConditionalStructureGenerator.load(str(path / "generator")),
            version=manifest["version"],
            metrics=manifest.get("metrics", {}),
            info=manifest.get("info", {}),
        )

    @staticmethod
    def read_manifest(directory: str) -> Optional[Dict[str, Any]]:
        file = Path(directory) / MANIFEST
        return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None
