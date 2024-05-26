from setuptools import setup, find_packages
from pathlib import Path


def readme(root_path):
    """Returns the text content of the README.rst of the package

    Parameters
    ----------
    root_path : pathlib.Path
        path to the root of the package
    """
    with root_path.joinpath('README.md').open(encoding='UTF-8') as f:
        return f.read()


root_path = Path(__file__).parent
README = readme(root_path)

version = "0.0.1"
description = "Tensor decompositions"

config = {
    "name": "td",
    "version": version,
    "description": description,
    "long_description": README,
    "packages": find_packages("td"),
    "install_requires": ["numpy", "torch"],
}

setup(**config)
