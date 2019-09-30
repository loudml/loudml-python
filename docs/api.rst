.. _api:

API Documentation
=================

All the API calls map the raw REST api as closely as possible, including the
distinction between required and optional arguments to the calls. This means
that the code makes distinction between positional and keyword arguments; we,
however, recommend that people **use keyword arguments for all calls for
consistency and safety**.

.. note::

    for compatibility with the Python ecosystem we use ``_from`` instead of
    ``from`` and ``_to`` instead of ``to`` as parameter names.


Global options
--------------

Some parameters are added by the client itself and can be used in all API
calls.

Ignore
~~~~~~

An API call is considered successful (and will return a response) if
loudml returns a 2XX response. Otherwise an instance of
:class:`~loudml.TransportError` (or a more specific subclass) will be
raised. You can see other exception and error states in :ref:`exceptions`. If
you do not wish an exception to be raised you can always pass in an ``ignore``
parameter with either a single status code that should be ignored or a list of
them::

    from loudml import Loud
    loud = Loud()

    # ignore 404 and and 400 when deleting a model
    loud.models.delete(model_name='test-model', ignore=[400, 404])


Timeout
~~~~~~~

Global timeout can be set when constructing the client (see
:class:`~loudml.Connection`'s ``timeout`` parameter) or on a per-request
basis using ``request_timeout`` (float value in seconds) as part of any API
call, this value will get passed to the ``perform_request`` method of the
connection class::

    # only wait for 1 second, regardless of the client's default
    loud.version(request_timeout=1)

.. note::

    Some API calls also accept a ``timeout`` parameter that is passed to
    Loud ML server. This timeout is internal and doesn't guarantee that the
    request will end in the specified time.


.. py:module:: loudml

Response Filtering
~~~~~~~~~~~~~~~~~~

The ``include_fields`` and ``fields`` parameters are used to reduce the response returned by loudml.

For example, to only return ``settings`` do::

    loud.models.get(model_names=['test-model'], fields=['settings'], include_fields=True)


Loud
----

.. autoclass:: Loud
      :members:

.. py:module:: loudml

Models
------

.. autoclass:: ModelsClient
      :members:

Jobs
----

.. autoclass:: JobsClient
      :members:

Scheduled Jobs
--------------

.. autoclass:: ScheduledJobsClient
      :members:

Buckets
-------

.. autoclass:: BucketsClient
      :members:

Templates
---------

.. autoclass:: TemplatesClient
      :members:
