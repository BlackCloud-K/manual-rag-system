from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGE_ORDER = ("parse", "normalize", "vlm", "inject", "embed", "upload")


def state_dir(project_root: Path) -> Path:
    return project_root / "data" / ".pipeline_import"


def state_path(project_root: Path, stem: str) -> Path:
    return state_dir(project_root) / f"{stem}.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PipelineState:
    """Per-manual checkpoints under data/.pipeline_import/<stem>.json."""

    project_root: Path
    stem: str
    pdf_name: str
    stages: dict[str, Any]

    @classmethod
    def load(cls, project_root: Path, stem: str, pdf_name: str) -> PipelineState:
        path = state_path(project_root, stem)
        if not path.exists():
            return cls(
                project_root=project_root,
                stem=stem,
                pdf_name=pdf_name,
                stages={},
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls(project_root=project_root, stem=stem, pdf_name=pdf_name, stages={})
        if not isinstance(raw, dict):
            return cls(project_root=project_root, stem=stem, pdf_name=pdf_name, stages={})
        stages = raw.get("stages")
        if not isinstance(stages, dict):
            stages = {}
        loaded_pdf = str(raw.get("pdf_name", "") or "").strip()
        if loaded_pdf:
            pdf_name = loaded_pdf
        return cls(project_root=project_root, stem=stem, pdf_name=pdf_name, stages=stages)

    def save(self) -> None:
        path = state_path(self.project_root, self.stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stem": self.stem,
            "pdf_name": self.pdf_name,
            "updated_at": _utc_now_iso(),
            "stages": dict(self.stages),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def is_done(self, stage: str) -> bool:
        entry = self.stages.get(stage)
        if not isinstance(entry, dict):
            return False
        return bool(entry.get("done"))

    def mark_done(self, stage: str, **extra: Any) -> None:
        payload: dict[str, Any] = {"done": True, "finished_at": _utc_now_iso()}
        payload.update(extra)
        self.stages[stage] = payload

    def mark_not_done_from(self, start_stage: str) -> None:
        if start_stage not in STAGE_ORDER:
            return
        idx = STAGE_ORDER.index(start_stage)
        for name in STAGE_ORDER[idx:]:
            self.stages.pop(name, None)


def restart_from(stage: str) -> str:
    s = stage.strip().lower()
    if s not in STAGE_ORDER:
        allowed = ", ".join(STAGE_ORDER)
        raise ValueError(f"unknown stage {stage!r}; allowed: {allowed}")
    return s
