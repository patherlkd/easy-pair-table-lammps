"""
easy-pair-table-lammps (EPTL) a python tool to create any pair potential to use with the LAMMPS pair_style table option.
"""
import os
from setuptools import setup, find_packages

if not os.path.isfile("eptl/resources/lmp_executable"):
    file = open("eptl/resources/lmp_executable", "w")
    file.write("EXECPATHHERE")
    file.close()

setup(
    name='eptl',
    version='v2.2.0',
    description='easy-pair-table-lammps (EPTL) a python tool to create any pair potential to use with the LAMMPS pair_style table option.',
    author='Luke Davis',
    author_email='luke.davis@ucl.ac.uk',
    packages=find_packages("eptl"),
    install_requires=['numpy', "scipy", "autograd", "matplotlib"],
)
