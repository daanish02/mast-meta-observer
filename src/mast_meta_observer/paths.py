from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class OutputPaths:
    """Filesystem locations for generated code and observer artifacts."""

    root: Path
    project: str
    run_id: str
    project_root: Path
    project_dir: Path
    mast_dir: Path
    checkpoints_dir: Path
    trace_path: Path
    report_path: Path
    checkpoint_index_path: Path


def ensure_output_paths(root: Path, project: str, run_id: str) -> OutputPaths:
    """Create output directories and return canonical paths.

    Args:
        root: Project root directory for the wrapper.
        project: User-provided project name.

    Returns:
        OutputPaths with existing directories.
    """
    created_projects = root / "created-projects"
    project_root = created_projects / project
    project_dir = project_root / "runs" / run_id
    mast_dir = project_root / ".mast" / run_id
    checkpoints_dir = mast_dir / "checkpoints"

    project_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    return OutputPaths(
        root=root,
        project=project,
        run_id=run_id,
        project_root=project_root,
        project_dir=project_dir,
        mast_dir=mast_dir,
        checkpoints_dir=checkpoints_dir,
        trace_path=mast_dir / "trace.json",
        report_path=mast_dir / "report.json",
        checkpoint_index_path=checkpoints_dir / "index.json",
    )
