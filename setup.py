from setuptools import setup, find_packages

setup(
    name="pile",
    version="0.0.1",
    packages=find_packages(exclude=["pipelines", "tests"]),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "PyYAML",
    ],
    #entry_points={
    #    "console_scripts": [
    #        "run_pipeline=pipelines.pipeline_example:main",
    #    ],
    #},
)
