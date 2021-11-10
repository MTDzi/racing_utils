from setuptools import setup, find_packages


setup(
    name='racing_utils',
    version='1.0.0',
    author='Maciej Dziubinski',
    author_email='ponadto@gmail.com',
    description='Small functions used in multiple places (f1tenth, offline_rl, ...) ',
    packages=find_packages(),
    install_requires=[
        'numpy >= 1.21.2',
        'torch >= 1.9.1',
        'scikit-learn >= 1.0',
    ],
)
