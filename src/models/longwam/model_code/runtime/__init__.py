# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/__init__.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.
#
# wam_interp vendoring note (NVlabs/LongLive @ 242635807b4ebb89fa2609a84df7eacde6aed92f):
# package marker only. Upstream's __init__ re-exports runtime/factory.py (training, datasets,
# LeRobot policy), which is not vendored; wam_interp.models.longwam.runtime ports create_longwam.

"""Shared model construction, training and inference execution runtime."""
