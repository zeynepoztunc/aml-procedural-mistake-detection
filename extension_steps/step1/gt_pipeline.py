from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    # Pipeline entrypoint (GT/oracle): runs the simple Step 1 script that pools features using GT timestamps.
    # Run the exported notebook script as-is.
    script = Path(__file__).resolve().parents[1] / "step1_pipeline.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
