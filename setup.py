from setuptools import setup

setup(
    name="panco2",
    version="0.1.0",
    description="panco2: pressure profile fitter from tSZ observations",
    author="Florian Keruzore",
    author_email="fkeruzore@anl.gov",
    packages=["panco2"],
    install_requires=[
        "numpy",
        "astropy",
        "scipy",
        "matplotlib",
        "pandas",
        "emcee",
        "tqdm",
        "dill",
        "chainconsumer",
        "sklearn",
    ],
)
