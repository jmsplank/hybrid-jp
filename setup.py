"""Muldule installation script."""
from setuptools import find_packages, setup

packages = find_packages("src")

setup(
    name="hybrid_jp",
    version="0.2.3",
    packages=packages,
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "sdf",
        "rich",
        "typer",
        "sympy",
        "ruptures",
        "tqdm",
        "pandas",
        "joblib",
        "statsmodels",
        "lic",
    ],
    entry_points={
        "console_scripts": [
            "hybrid-jp = hybrid_jp:hybrid_jp",
        ]
    },
)
