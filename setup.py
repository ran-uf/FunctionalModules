from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Functional Weights Implementation'
LONG_DESCRIPTION = 'Implement the functional weights in RKHS, including functional module and optimizer.'

# Setting up
setup(
    name="FunctionalModules",
    version=VERSION,
    author="Ran Dou",
    author_email="<dour@ufl.edu>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    keywords=['python', 'functional weights'],
    classifiers=[

    ]
)
