# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This needs to be done
Installation script for the 'isaaclab' python package.
"""

import os
import platform
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "onnx==1.16.1",  # 1.16.2 throws access violation on Windows
]

# # Additional dependencies that are only available on Linux platforms
# if platform.system() == "Linux":
#     INSTALL_REQUIRES += [
#         "pin-pink==3.1.0",  # required by isaaclab.isaaclab.controllers.pink_ik
#         "dex-retargeting==0.4.6",  # required by isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1_t2_dex_retargeting_utils
#     ]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu118"]

# Installation operation
setup(
    name="leatherbackpolicyexample",
    author="Pappachuck",
    maintainer="Jesus",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    # dependency_links=PYTORCH_INDEX_URL,
    packages=["leatherback.policy.example"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
