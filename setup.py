import io
from io import open as io_open
import os
from setuptools import setup, find_packages


def find_version():
    """Get version from loudml_py/_version.py"""
    _locals = locals()
    src_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(src_dir, 'loudml_py', '_version.py')
    with io_open(version_file, mode='r') as fd:
        exec(fd.read())  # __version__ is set in the exec call.
        return _locals['__version__']


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
    install_requires=[
        'aiohttp>=3.6.2',
        'requests>=2.14.0',
        'pyyaml==5.3',
        'tqdm>=4.35.0',
        'pytz>=2019.2',
        'dateutils>=0.6.6',
        'certifi',
    ],
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
    entry_points={
        'console_scripts': [
            'loudml=loudml_py.cli:main',
            'loudml-wave=loudml_py.wave:main',
        ],
    },
)
