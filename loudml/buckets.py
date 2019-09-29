from loudml.utils import (
    NamespacedClient, query_params, _make_path, SKIP_IN_PATH
)
from loudml.errors import TransportError


class BucketsClient(NamespacedClient):
    def generator(
        self,
        bucket_names=None,
        fields=None,
        include_fields=None,
        sort='name:1',
        per_page=100,
    ):
        page = 0
        while True:
            found = 0
            for bucket in self.get(
                bucket_names=bucket_names,
                fields=fields,
                include_fields=include_fields,
                sort=sort,
                per_page=per_page,
                page=page,
            ):
                yield bucket
                found += 1

            page += 1
            if not found:
                break

    @query_params('fields', 'include_fields', 'page', 'per_page', 'sort')
    def get(
        self, bucket_names=None, params=None
    ):
        return self.transport.perform_request('GET', _make_path(
            'buckets', bucket_names), params=params)

    @query_params()
    def exists(
        self, bucket_name, params=None
    ):
        if bucket_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'bucket_name'.")
        try:
            return self.transport.perform_request('HEAD', _make_path(
                'buckets', bucket_name), params=params)
        except TransportError:
            return False

    @query_params()
    def create(
        self, settings, params=None
    ):
        for param in (settings):
            if param in SKIP_IN_PATH:
                raise ValueError("Empty value passed for a required argument.")
        return self.transport.perform_request(
            'POST', '/buckets', params=params, body=settings)

    @query_params()
    def delete(
        self, bucket_name, params=None
    ):
        if bucket_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'bucket_name'.")
        return self.transport.perform_request('DELETE', _make_path(
            'buckets', bucket_name), params=params)

    @query_params()
    def clear(
        self, bucket_name, params=None
    ):
        if bucket_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'bucket_name'.")
        return self.transport.perform_request('POST', _make_path(
            'buckets', bucket_name, '_clear'), params=params)

    @query_params('from', 'to', 'bucket_interval', 'features')
    def read(
        self, bucket_name, params=None
    ):
        response = self.transport.perform_request('POST', _make_path(
            'buckets', bucket_name, '_read'), params=params)
        return response

    def write(
        self,
        bucket_name,
        points,
        batch_size,
        **kwargs
    ):
        params = kwargs
        for batch in [
            points[i:i+batch_size]
            for i in range(
                0, len(points), batch_size)
        ]:
            response = self.transport.perform_request('POST', _make_path(
                'buckets', bucket_name, '_write'), params=params, body=batch)
            yield response
