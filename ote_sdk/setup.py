"""
setup file for OTE SDK
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from setuptools import find_packages, setup

install_requires = []

with open("requirements.txt", "r", encoding="UTF-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            install_requires.append(line)

setup(
    name="otx",
    version="1.0",
    packages=find_packages(),
    package_data={"otx": ["py.typed", "usecases/exportable_code/demo/*"]},
    url="",
    license="Copyright (c) 2021-2022 Intel Corporation. "
    "SPDX-License-Identifier: Apache-2.0",
    install_requires=install_requires,
    author="Intel",
    description="OTE SDK Package",
)
