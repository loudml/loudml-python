****************
Loud ML - Python
****************

The Python client interface to the Loud ML model server.

|pypi| |build| |coverage| |license|

============
Installation
============

loudml-python requires a running Loud ML `model server <https://github.com/regel/loudml>`_ . See `Loud ML's quickstart <https://loudml.io/guide>`_ for installation instructions.

loudml-python can be installed using `pip` similar to other Python packages. Do not use `sudo` with `pip`. It is usually good to work in a `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ or `venv <https://docs.python.org/3/library/venv.html>`_ to avoid conflicts with other package managers and Python projects. For a quick introduction see `Python Virtual Environments in Five Minutes <https://bit.ly/py-env>`_.

To install loudml-python, simply:

.. code-block:: bash

    $ pip install loudml-python

or from source:

.. code-block:: bash
       
    $ python setup.py install


Getting Started
---------------

If you’ve installed `loudml-python` locally, the `loudml` command should be available via the command line. Executing loudml will start the CLI and automatically connect to the local Loud ML model server instance (assuming you have already started the server with `systemctl start loudmld` or by running loudmld directly). The output should look like this:

.. code-block:: bash

    $ loudml
    Connected to http://localhost:8077 version 1.6.0
    Loud ML shell
    >

You can get a description of the available commands:

.. code-block:: bash

    > help

Client Classes: Loud and Job
----------------------------

The main helper in the Python client librery is the `Loud` class. You
can create an instance that connects to a remote Loud ML model server
and run queries.

.. code-block:: pycon

    >>> import loudml.api
    >>> loud = loudml.api.Loud(loudml_host='localhost', loudml_port=8077)
    >>> models = loud.get_models(
            model_names=['first_model'],
            fields=['settings', 'state'],
            include_fields=True,
        )
    >>> len(models)
    1
    >>> print(models[0]['state'])
    {'trained': False}

Long running commands return a `Job` class instance that can be used to
track the progress of the job or cancel it. `loudml` uses
`tqdm <https://pypi.org/project/tqdm/>`_ to display progress information.

.. code-block:: pycon

    def cancel_job_handler(*args):
        job.cancel()
        print('Signal received. Canceled job: ', job.id)
        sys.exit()

    signal.signal(signal.SIGINT, cancel_job_handler)

    while not job.done():
        time.sleep(1)
        job.fetch()

Data generator: loudml-wave
---------------------------

The `loudml-wave` tool is included in this package. You can use the
application to output time series data with a given pattern and write
the data to a bucket.

The output should look like this:

.. code-block:: bash

    $ loudml-wave -v -f now-1h -t now --shape sin --tags tag_one:foo,tag_two:bar output_bucket
    INFO:root:generating data from 2019-09-21 07:23:51.350293 to 2019-09-21 08:23:51.350316
    Connected to localhost:8077 version 1.5.0.88.g5ad0216
    INFO:root:writing 131 points
    timestamp                value                    tags                     
    1569043431.35            38369.884                tag_one=foo,tag_two=bar  
    1569043431.85            70881.022                tag_one=foo,tag_two=bar  
    1569043491.35            33949.816                tag_one=foo,tag_two=bar  
    1569043551.35            30892.148                tag_one=foo,tag_two=bar  
    1569043551.6833332       10851.922                tag_one=foo,tag_two=bar 

You can get a description of the available commands:

.. code-block:: bash

    $ loudml-wave -h

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

loudml-python is developed and maintained by Sebastien Leger (@regel).
It can be found here: https://github.com/loudml/loudml-python

Special thanks to:

* Christophe Osuna (@osunac) for all the review and packaging support.

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
