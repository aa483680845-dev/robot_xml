"""Reusable robot arm control primitives for MuJoCo + Pinocchio workflows."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

_VERSION = f"python{sys.version_info.major}.{sys.version_info.minor}"
_PYTHON_ABI = f"{sys.version_info.major}{sys.version_info.minor}"
_SITE_PACKAGES = Path(__file__).resolve().parent.parent / ".venv" / "lib" / _VERSION / "site-packages"
_CMEEL_SITE_PACKAGES = _SITE_PACKAGES / "cmeel.prefix" / "lib" / _VERSION / "site-packages"
_CMEEL_LIB = _SITE_PACKAGES / "cmeel.prefix" / "lib"

preferred_paths = [str(path) for path in (_CMEEL_SITE_PACKAGES, _SITE_PACKAGES) if path.exists()]
for preferred_path in reversed(preferred_paths):
    if preferred_path in sys.path:
        sys.path.remove(preferred_path)
    sys.path.insert(0, preferred_path)

preload_names = [
    f"libboost_python{_PYTHON_ABI}.so.1.90.0",
    "libeigenpy.so",
    "libcoal.so",
    "libpinocchio_default.so",
    "libpinocchio_collision.so",
    "libpinocchio_parsers.so",
]
for name in preload_names:
    library_path = _CMEEL_LIB / name
    if library_path.exists():
        ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
