# wam_interp: vendored from OpenWAM@3950b6d9 (Apache-2.0); import paths rewritten from `openwam.*`.
from .torch_device import (
    IS_CUDA_AVAILABLE as IS_CUDA_AVAILABLE,
)
from .torch_device import (
    get_available_device_type as get_available_device_type,
)
from .torch_device import (
    parse_device_type as parse_device_type,
)
from .torch_device import (
    parse_nccl_backend as parse_nccl_backend,
)
