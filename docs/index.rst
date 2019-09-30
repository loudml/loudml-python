Python Loud ML Client
===========================

Official low-level client for Loud ML. Its goal is to provide common
ground for all Loud-related code in Python; because of this it tries
to be opinion-free and very extendable.

Compatibility
-------------

The library is compatible with all Loud ML versions since ``1.6.x`` but you
**have to use a matching major version**:

For **Loud ML 1.6** and later, use the major version 1 (``1.x.y``) of the
library.

The recommended way to set your requirements in your `setup.py` or
`requirements.txt` is::

    # Loud ML 1.x
    loudml-python>=1.6.0,<2.0.0

Installation
------------

Install the ``loudml-python`` package with `pip
<https://pypi.python.org/pypi/loudml-python>`_::

    pip install loudml-python

Example Usage
-------------

::

    from datetime import datetime
    from loudml.client import Loud
    loud = Loud(hosts=['localhost:8077'])
    models = loud.models.get(
        model_names=['first_model'],
        fields=['settings', 'state'],
        include_fields=True,
    )
    print(models[0]['state'])


Features
--------

This client was designed as very thin wrapper around Loud ML's REST API to
allow for maximum flexibility. This means that there are no opinions in this
client; it also means that some of the APIs are a little cumbersome to use from
Python.

Persistent Connections
~~~~~~~~~~~~~~~~~~~~~~

``loudml-python`` uses persistent connections inside of individual connection
pools (one per each configured or sniffed node). Out of the box you can choose
between two ``http`` protocol implementations. See :ref:`transports` for more
information.

The transport layer will create an instance of the selected connection class
per node and keep track of the health of individual nodes - if a node becomes
unresponsive (throwing exceptions while connecting to it) it's put on a timeout
by the :class:`~loudml.ConnectionPool` class and only returned to the
circulation after the timeout is over (or when no live nodes are left). By
default nodes are randomized before being passed into the pool and round-robin
strategy is used for load balancing.

You can customize this behavior by passing parameters to the
:ref:`connection_api` (all keyword arguments to the
:class:`~loudml.Loud` class will be passed through). If what
you want to accomplish is not supported you should be able to create a subclass
of the relevant component and pass it in as a parameter to be used instead of
the default implementation.


Automatic Retries
~~~~~~~~~~~~~~~~~

If a connection to a node fails due to connection issues (raises
:class:`~loudml.ConnectionError`) it is considered in faulty state. It
will be placed on hold for ``dead_timeout`` seconds and the request will be
retried on another node. If a connection fails multiple times in a row the
timeout will get progressively larger to avoid hitting a node that's, by all
indication, down. If no live connection is available, the connection that has
the smallest timeout will be used.

By default retries are not triggered by a timeout
(:class:`~loudml.ConnectionTimeout`), set ``retry_on_timeout`` to
``True`` to also retry on timeouts.

.. _sniffing:

Sniffing
~~~~~~~~

The client can be configured to inspect the cluster state to get a list of
nodes upon startup, periodically and/or on failure. See
:class:`~loudml.Transport` parameters for details.

Some example configurations::

    from loudml.client import Loud

    # by default we don't sniff, ever
    loud = Loud()

    # you can specify to sniff on startup to inspect the cluster and load
    # balance across all nodes
    loud = Loud(["seed1", "seed2"], sniff_on_start=True)

    # you can also sniff periodically and/or after failure:
    loud = Loud(["seed1", "seed2"],
              sniff_on_start=True,
              sniff_on_connection_fail=True,
              sniffer_timeout=60)

Thread safety
~~~~~~~~~~~~~

The client is thread safe and can be used in a multi threaded environment. Best
practice is to create a single global instance of the client and use it
throughout your application. If your application is long-running consider
turning on :ref:`sniffing` to make sure the client is up to date on the cluster
location.

By default we allow ``urllib3`` to open up to 10 connections to each node, if
your application calls for more parallelism, use the ``maxsize`` parameter to
raise the limit::

    # allow up to 25 connections to each node
    loud = Loud(["host1", "host2"], maxsize=25)

.. note::

    Since we use persistent connections throughout the client it means that the
    client doesn't tolerate ``fork`` very well. If your application calls for
    multiple processes make sure you create a fresh client after call to
    ``fork``. Note that Python's ``multiprocessing`` module uses ``fork`` to
    create new processes on POSIX systems.

SSL and Authentication
~~~~~~~~~~~~~~~~~~~~~~

You can configure the client to use ``SSL`` for connecting to your
loudml cluster, including certificate verification and HTTP auth::

    from loudml.client import Loud

    # you can use RFC-1738 to specify the url
    loud = Loud(['https://user:secret@localhost:443'])

    # ... or specify common parameters as kwargs

    loud = Loud(
        ['localhost', 'otherhost'],
        http_auth=('user', 'secret'),
        scheme="https",
        port=443,
    )

    # SSL client authentication using client_cert and client_key

    from ssl import create_default_context

    context = create_default_context(cafile="path/to/cert.pem")
    loud = Loud(
        ['localhost', 'otherhost'],
        http_auth=('user', 'secret'),
        scheme="https",
        port=443,
        ssl_context=context,
    )

..  warning::

    ``loudml-python`` doesn't ship with default set of root certificates. To
    have working SSL certificate validation you need to either specify your own
    as ``cafile`` or ``capath`` or ``cadata``  or install `certifi`_ which will
    be picked up automatically.


See class :class:`~loudml.Urllib3HttpConnection` for detailed
description of the options.

.. _certifi: http://certifiio.readthedocs.io/en/latest/

Logging
~~~~~~~

``loudml-python`` uses the standard `logging library`_ from python to define
two loggers: ``loudml`` and ``loudml.trace``. ``loudml``
is used by the client to log standard activity, depending on the log level.
``loudml.trace`` can be used to log requests to the server in the form
of ``curl`` commands using pretty-printed json that can then be executed from
command line. Because it is designed to be shared (for example to demonstrate
an issue) it also just uses ``localhost:8077`` as the address instead of the
actual address of the host. If the trace logger has not been configured
already it is set to `propagate=False` so it needs to be activated separately.

.. _logging library: http://docs.python.org/3.3/library/logging.html

Environment considerations
--------------------------

When using the client there are several limitations of your environment that
could come into play.

When using an HTTP load balancer you cannot use the :ref:`sniffing`
functionality - the cluster would supply the client with IP addresses to
directly connect to the cluster, circumventing the load balancer. Depending on
your configuration this might be something you don't want or break completely.

In some environments (notably on Google App Engine) your HTTP requests might be
restricted so that ``GET`` requests won't accept body. In that case use the
``send_get_body_as`` parameter of :class:`~loudml.Transport` to send all
bodies via post::

    from loudml.client import Loud
    loud = Loud(send_get_body_as='POST')

Compression
~~~~~~~~~~~
When using capacity-constrained networks (low throughput), it may be handy to enable
compression. This is especially useful when doing bulk loads or inserting large
documents. This will configure compression::

   from loudml.client import Loud
   loud = Loud(hosts, http_compress=True)


Running on AWS with IAM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use this client with IAM based authentication on AWS you can use
the `requests-aws4auth`_ package::

    from loudml import Loud, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth

    host = 'YOURHOST.us-east-1.es.amazonaws.com'
    awsauth = AWS4Auth(YOUR_ACCESS_KEY, YOUR_SECRET_KEY, REGION, 'es')

    loud = Loud(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    print(loud.info())

.. _requests-aws4auth: https://pypi.python.org/pypi/requests-aws4auth

Customization
-------------

Custom serializers
~~~~~~~~~~~~~~~~~~

By default, `JSONSerializer`_ is used to encode all outgoing requests.
However, you can implement your own custom serializer::

   from loudml.serializer import JSONSerializer

   class SetEncoder(JSONSerializer):
       def default(self, obj):
           if isinstance(obj, set):
               return list(obj)
           if isinstance(obj, Something):
               return 'CustomSomethingRepresentation'
           return JSONSerializer.default(self, obj)

   loud = Loud(serializer=SetEncoder())

.. _JSONSerializer: https://github.com/loudml/loudml-python/blob/master/loudml/serializer.py#L24

Contents
--------

.. toctree::
      :maxdepth: 2

   api
   connection
   transports

License
-------

`loudml-python` is a product of collaborative work.
Unless otherwise stated, all authors (see commit logs) retain copyright
for their respective work, and release the work under the MIT licence
(text below).

Copyright (c) 2019 Sebastien Leger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

