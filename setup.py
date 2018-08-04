# -*- coding: utf-8 -*-

import codecs

from setuptools import setup, find_packages


setup(
    name="maze-generator",
    version="0.0.0",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    url='https://github.com/alexebaker/python-maze_generator',
    author='Alexander Baker',
    author_email='alexebaker@gmail.com',
    description='Generates a maze.',
    keywords=['maze'],
    long_description=codecs.open('README.md', encoding="utf8").read(),
    entry_points={
        'console_scripts': ['maze-generator=maze_generator.cli:run']},
    install_requires=[
        'numpy',
        'matplotlib'],
    setup_requires=[
        'pytest-runner'],
    tests_require=[
        'pytest',
        'pytest-cov'])
