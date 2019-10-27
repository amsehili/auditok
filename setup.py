import sys
import re
import ast
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
    packages=["auditok"],
    include_package_data=True,
    package_data={"auditok": ["data/*"]},
    zip_safe=False,
    platforms="ANY",
    provides=["auditok"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Telecommunications Industry",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    entry_points={"console_scripts": ["auditok = auditok.cmdline:main"]},
)
