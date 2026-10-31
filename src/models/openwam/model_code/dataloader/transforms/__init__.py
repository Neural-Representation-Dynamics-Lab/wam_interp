# wam_interp: upstream re-exported the full transform pipeline; only normalize/multiview are vendored.
from wam_interp.models.openwam.model_code.dataloader.transforms.normalize import Normalizer

__all__ = ["Normalizer"]
