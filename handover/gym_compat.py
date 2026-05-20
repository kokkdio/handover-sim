# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

"""Compatibility import for the legacy Gym dependency."""

import contextlib
import io
import warnings


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="The distutils package is deprecated.*",
        category=DeprecationWarning,
    )
    with contextlib.redirect_stderr(io.StringIO()):
        import gym as gym  # noqa: F401
