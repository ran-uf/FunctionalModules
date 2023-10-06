from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Functional Weights Implementation'
LONG_DESCRIPTION = 'Implement the functional weights in RKHS, including functional module and and functional optimizer.'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="KLinear",
    version=VERSION,
    author="Ran Dou",
    author_email="<dour@ufl.edu>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'

    keywords=['python', 'functional weights'],
    classifiers=[

    ]
)