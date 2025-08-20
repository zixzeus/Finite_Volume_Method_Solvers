"""
Setup script for FVM Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fvm-framework",
    version="1.0.0",
    author="FVM Framework Development Team",
    author_email="",
    description="High-performance 2D finite volume method solver with pipeline data-driven architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.18.0",
    ],
    extras_require={
        "visualization": ["matplotlib>=3.0.0"],
        "profiling": ["psutil>=5.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "matplotlib>=3.0.0", 
            "psutil>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fvm-demo=demo_solver:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)