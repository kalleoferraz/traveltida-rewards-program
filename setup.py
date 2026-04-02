from setuptools import setup, find_packages

setup(
    name="travel_tida",
    version="0.1",
    packages=find_packages(where="."), 
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "scipy"
    ],
)