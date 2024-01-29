import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="axgrad",
    version="0.1.0",
    author="Paweł Gąsiorowski",
    author_email="p.gasiorowski@axent.pl",
    description="Backpropagation engine for scalar values.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/axent-pl/axgrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)