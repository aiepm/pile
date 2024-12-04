from setuptools import setup, find_packages

setup(
    name="tinylib",
    version="0.0.1",
    packages=find_packages(exclude=["pipelines", "tests"]),
    install_requires=[
        "tinygrad",
        "albumentations",
        "numpy",
        "PyYAML",
    ],
    #entry_points={
    #    "console_scripts": [
    #        "run_pipeline=pipelines.pipeline_example:main",
    #    ],
    #},
)
