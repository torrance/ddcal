import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ddcal",
    version="0.0.1",
    author="Torrance Hodgson",
    author_email="torrance.hodgson@postgrad.curtin.edu.au",
    description="A directional calibration tool for radio astronomy measurement sets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/torrance/ddcal",
    packages=setuptools.find_packages(),
    scripts=['bin/ddcal'],
)
