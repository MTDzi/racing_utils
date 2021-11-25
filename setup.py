from setuptools import setup, find_packages


setup(
    name='racing_utils',
    version='0.0.1',
    author='Maciej Dziubinski',
    author_email='mtdziubinski@gmail.com',
    description='Functions and classes used in multiple places (f1tenth_gym, offline_rl, ...) ',
    packages=find_packages(),
    install_requires=[
        'numpy >= 1.21.2',
        'torch >= 1.9.1',
        'scikit-learn >= 1.0',
    ],
)
