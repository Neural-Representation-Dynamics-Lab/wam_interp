# wam_interp: vendored from OpenWAM@3950b6d9 (Apache-2.0); import paths rewritten from `openwam.*`.
"""DualSystem architecture family.

Three variants split into independent classes:

- :class:`DualSystemCrossAttnArchitecture` — bridge-collection plan, ActionDiT
  runs after the video DiT.
- :class:`DualSystemSelfAttnArchitecture` — interleaved plan, joint ActionDiT
  blocks run inside the video DiT loop.
- :class:`DualSystemIDMArchitecture` — two-stage IDM plan, video denoised first,
  then action denoised with frozen video as condition.
"""

from wam_interp.models.openwam.model_code.model.architectures.dual_system.idm import DualSystemIDMArchitecture
from wam_interp.models.openwam.model_code.model.architectures.dual_system.joint_cross_attn import DualSystemCrossAttnArchitecture
from wam_interp.models.openwam.model_code.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture

__all__ = [
    "DualSystemCrossAttnArchitecture",
    "DualSystemSelfAttnArchitecture",
    "DualSystemIDMArchitecture",
]
