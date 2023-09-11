"""Setup."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Tue 12 Sep 2023 12:36:45 AM CST
# ***
# ************************************************************************************/
#

from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="RefineMask",
    version="1.0.0",
    author="Dell Du",
    author_email="18588220928@163.com",
    description="Refine Mask Model Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/delldu/RefineMask.git",
    packages=["RefineMask"],
    package_data={"RefineMask": ["models/RefineMask.pth",]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch >= 1.9.0",
        "torchvision >= 0.10.0",
        "Pillow >= 7.2.0",
        "numpy >= 1.19.5",
        "todos >= 1.0.0",
    ],
)
