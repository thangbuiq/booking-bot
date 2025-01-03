from setuptools import find_packages, setup

setup(
    name="booking",
    version="0.1.0",
    packages=find_packages(
        include=["scraper.booking", "scraper.booking.*", "core", "core.*"]
    ),
)
