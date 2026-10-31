# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: src/fastwam/utils/logging_config.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.
#
# wam_interp vendoring note (NVlabs/LongLive @ 242635807b4ebb89fa2609a84df7eacde6aed92f,
# Long-WAM/src/longwam/utils/logging_config.py): only `get_logger` and `_is_main_process`
# are kept, unchanged. The Rich-based `setup_logging` (and its `rich` import) was removed
# because the vendored inference code never calls it; the host application owns logging.

import logging
import os

import torch.distributed as dist


def _is_main_process() -> bool:
    """
    Best-effort check for main process without any synchronization.
    """
    # Prefer torch.distributed state if initialized.
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0

    # Fallback to environment variables commonly set by launchers.
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return os.environ.get(key, "0") in ("0", "0\n", "")

    return True


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Drop-in replacement for accelerate.logging.get_logger:
    - No implicit barriers.
    - Only the main process emits log records by default.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # if not logger.handlers and _is_main_process():
    #     handler = logging.StreamHandler()
    #     formatter = logging.Formatter(
    #         fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #         datefmt="%Y-%m-%d %H:%M:%S",
    #     )
    #     handler.setFormatter(formatter)
    #     logger.addHandler(handler)

    if not _is_main_process():
        logger.propagate = False
        logger.disabled = True

    return logger
