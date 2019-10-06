from urllib.parse import (
    urlencode,
    urlparse,
    quote,
    unquote,
)
from loudml_py.utils import (
    string_types,
    query_params,
)
from loudml_py.misc import format_points
from loudml_py.transport import Transport
from loudml_py.errors import TransportError
from loudml_py.jobs import JobsClient
from loudml_py.scheduled_jobs import ScheduledJobsClient
from loudml_py.models import ModelsClient
from loudml_py.templates import TemplatesClient
from loudml_py.buckets import BucketsClient


def _normalize_hosts(hosts):
    """
    Helper function to transform hosts argument to
    :class:`~loudml.Loud` to a list of dicts.
    """
    # if hosts are empty, just defer to defaults down the line
    if hosts is None:
        return [{}]

    # passed in just one string
    if isinstance(hosts, string_types):
        hosts = [hosts]

    out = []
    # normalize hosts to dicts
    for host in hosts:
        if isinstance(host, string_types):
            if '://' not in host:
                host = "//%s" % host

            parsed_url = urlparse(host)
            h = {"host": parsed_url.hostname}

            if parsed_url.port:
                h["port"] = parsed_url.port

            if parsed_url.scheme == "https":
                h['port'] = parsed_url.port or 443
                h['use_ssl'] = True

            if parsed_url.username or parsed_url.password:
                h['http_auth'] = '%s:%s' % (unquote(parsed_url.username),
                                            unquote(parsed_url.password))

            if parsed_url.path and parsed_url.path != '/':
                h['url_prefix'] = parsed_url.path

            out.append(h)
        else:
            out.append(host)
    return out


class Loud():
    """
    Loud ML low-level client. Provides a straightforward mapping from
    Python to REST endpoints.

    The instance has attributes ``jobs``, ``scheduled_jobs``, ``models``,
    ``templates``, and ``buckets`` that provide access to instances of
    :class:`~loudml.JobsClient`,
    :class:`~loudml.ScheduledJobsClient`,
    :class:`~loudml.ModelsClient`,
    :class:`~loudml.TemplatesClient`, and
    :class:`~loudml.BucketsClient` respectively. This is the
    preferred (and only supported) way to get access to those classes
    and their methods.

    You can specify your own connection class which should be used by providing
    the ``connection_class`` parameter::

        # create connection to localhost using the ThriftConnection
        loud = Loud(connection_class=ThriftConnection)

    If you want to turn on :ref:`sniffing` you have several options (described
    in :class:`~loudml.Transport`)::

        # create connection that will automatically inspect the cluster to get
        # the list of active nodes. Start with nodes running on 'loudnode1' and
        # 'loudnode2'
        loud = Loud(
            ['loudnode1', 'loudnode2'],
            # sniff before doing anything
            sniff_on_start=True,
            # refresh nodes after a node fails to respond
            sniff_on_connection_fail=True,
            # and also every 60 seconds
            sniffer_timeout=60
        )

    Different hosts can have different parameters, use a dictionary per node to
    specify those::

        # connect to localhost directly and another node using SSL on port 443
        # and an url_prefix. Note that ``port`` needs to be an int.
        loud = Loud([
            {'host': 'localhost'},
            {'host': 'othernode', 'port': 443, 'url_prefix': 'loud', 'use_ssl': True},
        ])

    If using SSL, there are several parameters that control how we deal with
    certificates (see :class:`~loudml.Urllib3HttpConnection` for
    detailed description of the options)::

        loud = Loud(
            ['localhost:443', 'other_host:443'],
            # turn on SSL
            use_ssl=True,
            # make sure we verify SSL certificates
            verify_certs=True,
            # provide a path to CA certs on disk
            ca_certs='/path/to/CA_certs'
        )

    SSL client authentication is supported
    (see :class:`~loudml.Urllib3HttpConnection` for
    detailed description of the options)::

        loud = Loud(
            ['localhost:443', 'other_host:443'],
            # turn on SSL
            use_ssl=True,
            # make sure we verify SSL certificates
            verify_certs=True,
            # provide a path to CA certs on disk
            ca_certs='/path/to/CA_certs',
            # PEM formatted SSL client certificate
            client_cert='/path/to/clientcert.pem',
            # PEM formatted SSL client key
            client_key='/path/to/clientkey.pem'
        )

    Alternatively you can use RFC-1738 formatted URLs, as long as they are not
    in conflict with other options::

        loud = Loud(
            [
                'http://user:secret@localhost:9200/',
                'https://user:secret@other_host:443/production'
            ],
            verify_certs=True
        )

    By default, `JSONSerializer
    <https://github.com/loudml/loudml-python/blob/master/loudml/serializer.py#L24>`_
    is used to encode all outgoing requests.
    However, you can implement your own custom serializer::

        from loudml_py.serializer import JSONSerializer

        class SetEncoder(JSONSerializer):
            def default(self, obj):
                if isinstance(obj, set):
                    return list(obj)
                if isinstance(obj, Something):
                    return 'CustomSomethingRepresentation'
                return JSONSerializer.default(self, obj)

        loud = Loud(serializer=SetEncoder())

    """
    def __init__(self, hosts=None, transport_class=Transport, **kwargs):
        """
        :arg hosts: list of nodes we should connect to. Node should be a
            dictionary ({"host": "localhost", "port": 9200}), the entire dictionary
            will be passed to the :class:`~loudml.Connection` class as
            kwargs, or a string in the format of ``host[:port]`` which will be
            translated to a dictionary automatically.  If no value is given the
            :class:`~loudml.Urllib3HttpConnection` class defaults will be used.

        :arg transport_class: :class:`~loudml.Transport` subclass to use.

        :arg kwargs: any additional arguments will be passed on to the
            :class:`~loudml.Transport` class and, subsequently, to the
            :class:`~loudml.Connection` instances.
        """
        self.transport = transport_class(_normalize_hosts(hosts), **kwargs)

        # namespaced clients for compatibility with API names
        self.jobs = JobsClient(self)
        self.scheduled_jobs = ScheduledJobsClient(self)
        self.models = ModelsClient(self)
        self.templates = TemplatesClient(self)
        self.buckets = BucketsClient(self)

    def __repr__(self):
        try:
            # get a list of all connections
            cons = self.transport.hosts
            # truncate to 5 if there are too many
            if len(cons) > 5:
                cons = cons[:5] + ['...']
            return '<{cls}({cons})>'.format(cls=self.__class__.__name__, cons=cons)
        except Exception:
            # probably operating on custom transport and connection_pool, ignore
            return super(Loud, self).__repr__()

    def _format_url(self, query, query_params=None):
        scheme = 'https' if self._enable_ssl else 'http'
        if not query_params:
            return "{}://{}:{}{}".format(
                scheme,
                self._host,
                self._port,
                quote(query),
            )
        else:
            return "{}://{}:{}{}?{}".format(
                scheme,
                self._host,
                self._port,
                quote(query),
                urlencode(query_params),
            )

    @query_params()
    def version(self, params=None):
        info = self.info()
        return info.get('version')

    @query_params()
    def ping(self, params=None):
        """
        Returns True if the cluster is up, False otherwise.
        `<https://loudml.io/guide/>`_
        """
        try:
            return self.transport.perform_request('HEAD', '/', params=params)
        except TransportError:
            return False

    @query_params()
    def info(self, params=None):
        """
        Get the basic info from the current cluster.
        `<https://loudml.io/guide/>`_
        """
        return self.transport.perform_request('GET', '/', params=params)

    def write_points(
        self, bucket_name, points, verbose=False, interval=1
    ):
        if verbose:
            for line in format_points(points):
                print(line)
        for job_name in self.buckets.write(
            bucket_name=bucket_name,
            points=points,
            batch_size=len(points),
        ):
            job = self.jobs.id(job_name)
            job.wait(interval)
