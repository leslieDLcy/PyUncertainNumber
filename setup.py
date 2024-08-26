
from setuptools import find_packages, setup

setup(
    name='PyUncertainNumber',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    version='0.0.1',
    description='Uncertain Number in Python',
    author='Yu Chen (Leslie)',
    license='MIT',
)