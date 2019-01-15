import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="radical",
    version="0.0.1",
    author="Torrance Hodgson",
    author_email="torrance.hodgson@postgrad.curtin.edu.au",
    description="Radio Astronomy Directional Ionospheric Calibration: directional calibration for the Murchison Widefield Array",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/torrance/radical",
    packages=setuptools.find_packages(),
    scripts=['bin/radical'],
)
