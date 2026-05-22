"""CodeCarbon EmissionsTracker context manager for TutorMind experiments.

All scripts route emissions through `track_emissions(...)`, which writes a
per-project CSV under the repo's central `emissions/` directory. If
`codecarbon` isn't installed or the tracker fails to start, the context
manager becomes a no-op so experiments still run.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
EMISSIONS_DIR = REPO_ROOT / "emissions"


@contextlib.contextmanager
def track_emissions(
    project_name: str,
    output_dir: Optional[Path | str] = None,
) -> Iterator[Optional[object]]:
    """Run a block with CodeCarbon emissions tracking.

    Each call writes/append to `<output_dir>/<project_name>.csv` so different
    experiments do not collide in a single shared `emissions.csv`.
    """
    target_dir = Path(output_dir) if output_dir is not None else EMISSIONS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    tracker = None
    try:
        from codecarbon import EmissionsTracker  # type: ignore

        tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=str(target_dir),
            output_file=f"{project_name}.csv",
            log_level="error",
        )
        tracker.start()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[codecarbon] tracker disabled ({exc!s})")
        tracker = None

    try:
        yield tracker
    finally:
        if tracker is not None:
            try:
                emissions = tracker.stop()
                print(
                    f"[codecarbon] {project_name}: "
                    f"{emissions} kg CO2 -> {target_dir / f'{project_name}.csv'}"
                )
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[codecarbon] failed to stop tracker: {exc!s}")
