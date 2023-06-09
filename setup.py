from setuptools import find_packages, setup

packages = find_packages("src")
print(packages)

setup(
    name="hybrid_jp",
    version="0.1",
    packages=packages,
    package_dir={"": "src"},
    install_requires=["numpy", "sdf", "rich", "typer", "sympy", "ruptures"],
    entry_points={
        "console_scripts": [
            "hybrid-jp = hybrid_jp:hybrid_jp",
        ]
    },
)
