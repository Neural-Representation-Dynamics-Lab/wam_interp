# wam_interp: vendored from OpenWAM@3950b6d9 (Apache-2.0); import paths rewritten from `openwam.*`.
"""Architecture package exports and side-effect registration.

wam_interp: trimmed to dual_system; single_system and tri_system are not vendored.
"""

from wam_interp.models.openwam.model_code.model.architectures import dual_system  # noqa: F401
from wam_interp.models.openwam.model_code.model.architectures.base import ActionState, BaseWAMArchitecture
from wam_interp.models.openwam.model_code.model.architectures.dual_system import (
    DualSystemCrossAttnArchitecture,
    DualSystemIDMArchitecture,
    DualSystemSelfAttnArchitecture,
)
from wam_interp.models.openwam.model_code.model.architectures.registry import (
    ARCHITECTURE_METADATA,
    ARCHITECTURE_REGISTRY,
    ARCHITECTURE_SUPPORT,
    CanonicalArchitectureSpec,
    build_architecture,
    get_architecture_support,
    list_supported_architectures,
    normalize_architecture_spec,
    register_architecture,
    resolve_architecture_config,
)

__all__ = [
    "ActionState",
    "BaseWAMArchitecture",
    "ARCHITECTURE_METADATA",
    "ARCHITECTURE_REGISTRY",
    "ARCHITECTURE_SUPPORT",
    "CanonicalArchitectureSpec",
    "build_architecture",
    "get_architecture_support",
    "list_supported_architectures",
    "normalize_architecture_spec",
    "register_architecture",
    "resolve_architecture_config",
    "DualSystemCrossAttnArchitecture",
    "DualSystemIDMArchitecture",
    "DualSystemSelfAttnArchitecture",
]
