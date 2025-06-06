from setuptools import setup, find_packages
import os

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="datasage",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        'datasage': [
            'styles/*.mplstyle',
            'assets/*.png',
        ],
    },
    include_package_data=True,
    
    # Metadata
    author="Carmine Minichini",
    author_email="carmine.mnc@gmail.com",
    description="A collection of statistics/data science tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carminemnc/datasage",
    license = 'MIT',
    
    # Requirements
    python_requires=">=3.6",
    install_requires=requirements,
)
