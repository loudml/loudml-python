import io
import os
from setuptools import setup, find_packages


def find_version():
    if 'LOUDML_PYTHON_VERSION' in os.environ:
        return os.environ['LOUDML_PYTHON_VERSION']
    else:
        raise RuntimeError("Unable to find version string.")


with io.open('README.rst', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='loudml-python',
    version=find_version(),
    description='Loud ML Python API Client',
    long_description=long_description,
    author='Loud ML',
    author_email='support@loudml.io',
    license='MIT',
    packages=find_packages(),
    install_requires=['requests'],
    extras_require={'test': ['mock']},
    python_requires='>=2.7, !=3.0.*, !=3.1.*',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    url='https://github.com/loudml/loudml-python',
)
