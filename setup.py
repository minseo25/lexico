from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name='lexico',
    version='0.1.0',
    description='A package for extreme kv cache compression via sprase coding over universal dictionaries',
    url='https://github.com/krafton-ai/lexico',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)