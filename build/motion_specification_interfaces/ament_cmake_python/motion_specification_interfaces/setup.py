from setuptools import find_packages
from setuptools import setup

setup(
    name='motion_specification_interfaces',
    version='0.0.0',
    packages=find_packages(
        include=('motion_specification_interfaces', 'motion_specification_interfaces.*')),
)
