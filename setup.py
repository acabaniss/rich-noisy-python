import setuptools

setuptools.setup(
    name="richnoisy",
    version="0.1",
    author="Andrew Cabaniss",
    author_email="ahfc@umich.edu",
    description="Mark Newman's Rich Noisy Networks algorithm",
    long_description="""This package implements Mark Newman's (2018) algorithm for inferring network
structure from multiple (rich) error-prone (noisy) observations of network edges.

In this package, these observations of node existence (and the total number of
experiments or trials, either assuming a single set of experiments for all nodes
or for each node-node combination individually) are taken as numpy arrays. The
expectation-maximization algorithm then converges on the probability that each
edge actually exists. This probability, as well as the estimated parameters, 
are then available for further network analysis or an assessment of reliability.""",
    long_description_content_type="text/markdown",
    url="https://github.com/acabaniss/rich-noisy-python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)