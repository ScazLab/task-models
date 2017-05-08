import io
import os
from setuptools import setup, find_packages


def readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.rst')
    with io.open(readme_path) as f:
        return f.read()


setup(
    name='task_models',
    version='0.0.1',
    description='Task models for human robot collabortation',
    long_description=readme(),
    url='https://github.com/ScazLab/task-models',
    author='Olivier Mangin',
    author_email='olivier.mangin@yale.edu',
    license='BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    packages=find_packages(exclude=['tests']),
    test_suite="tests.test_suite",
)
