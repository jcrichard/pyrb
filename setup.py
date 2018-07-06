from setuptools import setup, find_packages

DISTNAME = "pyrb"
VERSION = '1.0.0'
DESCRIPTION = """pyrb is a Python library to solve risk bugeting problem."""
LONG_DESCRIPTION = """pyrb is a Python library to solve risk bugeting problem."""
AUTHOR = "Jean-Charles Richard"
AUTHOR_EMAIL = "jcharles.richard@gmail.com"
URL = "https://github.com/jcrichard/pyrb"
LICENSE = "Apache License, Version 2.0"

REQUIREMENTS = [
    "pandas>=0.19",
    "quadprog>=0.1.0"
]

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        packages=find_packages(),
        package_data={'docs': ['*']},
        include_package_data=True,
        zip_safe=False,
        install_requires=REQUIREMENTS,
        classifiers=[
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4'
        ]
    )
