# wam_interp: trimmed to the dual-system architectures; SharedMoEActionBackbone is not vendored.
# Import architecture packages to trigger @register_architecture decorators
from wam_interp.models.openwam.model_code.model import architectures  # noqa: F401
from wam_interp.models.openwam.model_code.model.action_backbone.separate_action_dit import ActionDiT, ActionDiTState
from wam_interp.models.openwam.model_code.model.architectures import (
    ARCHITECTURE_METADATA,
    ARCHITECTURE_REGISTRY,
    ARCHITECTURE_SUPPORT,
    ActionState,
    BaseWAMArchitecture,
    CanonicalArchitectureSpec,
    build_architecture,
    get_architecture_support,
    list_supported_architectures,
    normalize_architecture_spec,
    resolve_architecture_config,
)

__all__ = [
    "ActionDiT",
    "ActionDiTState",
    "BaseWAMArchitecture",
    "ActionState",
    "ARCHITECTURE_METADATA",
    "ARCHITECTURE_REGISTRY",
    "ARCHITECTURE_SUPPORT",
    "build_architecture",
    "get_architecture_support",
    "list_supported_architectures",
    "normalize_architecture_spec",
    "resolve_architecture_config",
    "CanonicalArchitectureSpec",
]
