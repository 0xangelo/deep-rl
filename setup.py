import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deep-rl-angelolovatto",
    version="0.1.0",
    author="Ângelo Gregório Lovatto",
    author_email="angelolovatto@gmail.com",
    description="PyTorch implementation of reinforcement learning algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/angelolovatto/deep-rl",
    packages=[
        package for package in setuptools.find_packages() if package.startswith("proj")
    ],
    install_requires=[
        "gym",
        "numpy",
        "tqdm",
        "flask",
        "plotly",
        "click",
        "cloudpickle",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
