**************
Loud ML - Python
**************

|pypi| |build| |coverage| |license|

In this repository, you'll find all information about the Loud ML Python CLI usage.


==============
What is the Loud ML CLI?
==============

Loud ML helps you to:

* Create and train deep learning models without TensorFlow headaches.
* Control data streaming and inference tasks from your databases to Loud ML inference nodes.
* Persist trained model states and persist model predictions
* Use unsupervised anomaly detection with time series data

The `loudml-python` CLI connects to a remote Loud ML host to run commands remotely. You may
run Loud ML instances locally or in your favorite data centers.

============
Installation
============

You can install the Loud ML Python CLI using the following command.

.. code-block::

    pip install loudml-python

For python3, use the following command

.. code-block::
       
    pip3 install loudml-python

Python 3.2 and 3.3 have reached `EOL <https://en.wikipedia.org/wiki/CPython#Version_history>`_ and support will be removed in the near future.

==========
Change Log
==========

Please see `CHANGELOG.md <https://github.com/loudml/loudml-python/blob/master/CHANGELOG.md>`_.

===============
Issue Reporting
===============

If you have found a bug or if you have a feature request, please report them at this repository issues section.
Please do not report security vulnerabilities on the public GitHub issue tracker.

======
Author
======

`Loud ML`_

=======
License
=======

This project is licensed under the MIT license. See the `LICENSE <https://github.com/loudml/loudml-python/blob/master/LICENSE>`_
file for more info.

.. _Loud ML: https://loudml.io

.. |pypi| image:: https://img.shields.io/pypi/v/loudml-python.svg?style=flat-square&label=latest%20version
       :target: https://pypi.org/project/loudml-python/
    :alt: Latest version released on PyPI

.. |build| image:: https://img.shields.io/circleci/project/github/loudml/loudml-python.svg?style=flat-square&label=circleci
       :target: https://circleci.com/gh/loudml/loudml-python
    :alt: Build status

.. |coverage| image:: https://img.shields.io/codecov/c/github/loudml/loudml-python.svg?style=flat-square&label=codecov
       :target: https://codecov.io/gh/loudml/loudml-python
    :alt: Test coverage

.. |license| image:: https://img.shields.io/:license-mit-blue.svg?style=flat-square
       :target: https://opensource.org/licenses/MIT
    :alt: License
