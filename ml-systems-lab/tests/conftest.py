"""Shared pytest setup.

Each lab is a plain folder (no __init__.py), and several of them contain a file
called app.py. Importing by name would make Python load whichever app.py it
finds first, which is exactly the kind of silent bug this course is about.
So we load each module explicitly, by file path, under a unique name.
"""

import importlib.util
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

LAB1 = PACKAGE_ROOT / "lab1_break_notebook"
LAB3 = PACKAGE_ROOT / "lab3_drift"
LAB4 = PACKAGE_ROOT / "lab4_batch_vs_online"
LAB5 = PACKAGE_ROOT / "hf_space"
LAB6 = PACKAGE_ROOT / "lab6_vertex"

# Lab 3 and Lab 4 have unique file names, so plain folders on sys.path are safe.
for folder in [LAB3, LAB4]:
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def load_module(path: Path, name: str):
    """Import a .py file by path, under a name you choose. Cached like a normal import."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
