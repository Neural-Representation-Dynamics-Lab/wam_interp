"""Allow canonical imports when the checkout directory is named ``wam_interp``.

In notebooks, the checkout's parent directory is often on ``sys.path``. If the
checkout is named ``wam_interp``, Python can find this directory before the
editable-install mapping, but the actual package modules live under ``src/``.
Adding that source directory to this package's search path keeps imports such
as ``wam_interp.models`` working in both cases.
"""

from pathlib import Path

_source_package = str(Path(__file__).resolve().parent / "src")
if _source_package not in __path__:
    __path__.append(_source_package)

from .types import InferenceRequest, InferenceResult, PreparedObservation

__all__ = ["InferenceRequest", "InferenceResult", "PreparedObservation"]
__version__ = "0.1.0"
