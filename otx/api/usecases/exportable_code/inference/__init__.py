"""Initialization of inference interfaces."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .inference import (
    AsyncOpenVINOTask,
    BaseInferencer,
    BaseOpenVINOInferencer,
    IInferencer,
)

__all__ = [
    "AsyncOpenVINOTask",
    "BaseInferencer",
    "BaseOpenVINOInferencer",
    "IInferencer",
]
