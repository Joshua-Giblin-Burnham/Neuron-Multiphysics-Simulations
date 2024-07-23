from setuptools import setup, find_packages

def get_readme():
    """
    Load README.md text for use as description.
    """
    with open('README.md') as f:
        return f.read()
    

setup(
    # Module name (lowercase)
    name='nuphysim',

    # Version
    version='1.0',

    description='A package to produce mutliphysics simulations of neurons',

    long_description=get_readme(),

    license='MIT license',

    # Author 
    author='J. Giblin-Burnham',
    author_email='j.giblin-burnham@hotmail.com'

    # Website
    url='https://neuron-multiphysics-simulations.readthedocs.io/en/latest/index.html',

    # Packages to include
    packages=find_packages(include=('nuphysim', 'nuphysim.*')),

    # List of dependencies
    install_requires = ["numpy", "matplotlib", "scipy", "absl-py", "biopython", "keras", "pypdf", "pypdf2", "scp", "paramiko", "sphinx_rtd_theme", "furo",],

    extras_require={
        'docs': [
            # Sphinx for doc generation. Version 1.7.3 has a bug:
            'sphinx>=1.5, !=1.7.3',
            # Nice theme for docs
            'sphinx_rtd_theme',
            'sphinx_autopackagesummary',
            'furo',
        ],
        'dev': [
            # Flake8 for code style checking
            'flake8>=3',
        ],
    }
    )