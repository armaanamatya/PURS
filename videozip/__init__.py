"""VideoZip — training-free video-guided audio token compression.

Importing this package adds OmniZip-OG/omnizip/ to sys.path so modules like
omnizip_units, modeling_qwen2_5_omni become importable from VideoZip code without
copying upstream files.
"""

from __future__ import annotations

import sys
import pathlib
from typing import Optional

_PKG_ROOT = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_ROOT.parent


def _prepend(p: pathlib.Path) -> None:
    s = str(p)
    if p.is_dir() and s not in sys.path:
        sys.path.insert(0, s)


# IMPORTANT: order matters. We must end with PROJECT_ROOT at sys.path[0] so that
# the customized `eval_qwen_omni_zip.py` at the project root takes precedence over
# any vendored copy inside OmniZip-main/. We prepend OmniZip dirs first, then
# PROJECT_ROOT last (so it ends up at index 0).
_CANDIDATES = [
    _PROJECT_ROOT / "OmniZip-main" / "omnizip",
    _PROJECT_ROOT / "OmniZip-OG" / "omnizip",
    _PROJECT_ROOT / "OmniZip-main",
]

OMNIZIP_PATH: Optional[pathlib.Path] = None  # type: ignore[name-defined]
for _cand in _CANDIDATES:
    if (_cand / "omnizip_units.py").is_file():
        _prepend(_cand)            # so `import omnizip_units` works
        _prepend(_cand.parent)     # so `import omnizip.modeling_qwen2_5_omni` works
        OMNIZIP_PATH = _cand
        break

if OMNIZIP_PATH is None:
    import warnings
    warnings.warn(
        "VideoZip: could not locate omnizip_units.py under any of "
        + ", ".join(str(c) for c in _CANDIDATES)
        + ". Imports of omnizip_units etc. will fail.",
        stacklevel=2,
    )

# qwen-omni-utils is a local package, not pip-installed.
for _qwen_root in (
    _PROJECT_ROOT / "OmniZip-main" / "qwen-omni-utils" / "src",
    _PROJECT_ROOT / "OmniZip-OG" / "qwen-omni-utils" / "src",
):
    if _qwen_root.is_dir():
        _prepend(_qwen_root)
        break

# Prepend PROJECT_ROOT LAST so the project-root `eval_qwen_omni_zip.py` is found
# before any vendored copy inside OmniZip-main/.
_prepend(_PROJECT_ROOT)

PROJECT_ROOT = _PROJECT_ROOT

__all__ = ["PROJECT_ROOT", "OMNIZIP_PATH"]
