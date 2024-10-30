import ast
import re
import sys

from setuptools import setup

_version_re = re.compile(r"__version__\s+=\s+(.*)")

with open("auditok/__init__.py", "rt") as f:
    version = str(ast.literal_eval(_version_re.search(f.read()).group(1)))
    long_desc = open("README.rst", "rt").read()

setup(
    name="auditok",
    version=version,
    url="http://github.com/amsehili/auditok/",
    license="MIT",
    author="Amine Sehili",
    author_email="amine.sehili@gmail.com",
    description="A module for Audio/Acoustic Activity Detection",
    long_description=long_desc,
    long_description_content_type="text/x-rst",
    packages=["auditok"],
    include_package_data=True,
    package_data={"auditok": ["data/*"]},
    zip_safe=False,
    platforms="ANY",
    provides=["auditok"],
    install_requires=[
        "numpy",
        "matplotlib",
        "pydub",
        "pyaudio",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Telecommunications Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    entry_points={"console_scripts": ["auditok = auditok.cmdline:main"]},
)
