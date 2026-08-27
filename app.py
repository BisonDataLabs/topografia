"""Public deployment entrypoint."""

from pathlib import Path
from runpy import run_path


run_path(str(Path(__file__).with_name("universal_app.py")), run_name="__main__")
