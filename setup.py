from setuptools import find_packages
from setuptools import setup

setup(
    name="booking",
    version="0.1.0",
    packages=find_packages(
        include=["scraper.booking", "scraper.booking.*", "core", "core.*"]
    ),
)
