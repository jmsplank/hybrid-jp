from setuptools import find_packages, setup

packages = find_packages("src")
print(packages)

setup(
    name="hybrid_jp",
    version="0.1",
    packages=packages,
    package_dir={"": "src"},
    install_requires=["numpy", "sdf", "rich", "typer"],
    entry_points={
        "console_scripts": [
            "sdf-vars = hybrid_jp:list_vars_app",
        ]
    },
)
