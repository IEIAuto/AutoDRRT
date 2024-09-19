from setuptools import find_packages
from setuptools import setup

setup(
    name='dma_customer_msg',
    version='0.0.0',
    packages=find_packages(
        include=('dma_customer_msg', 'dma_customer_msg.*')),
)
