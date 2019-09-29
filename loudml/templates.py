from loudml.utils import (
    NamespacedClient, query_params, _make_path, SKIP_IN_PATH
)
from loudml.errors import TransportError


class TemplatesClient(NamespacedClient):
    def generator(
        self,
        template_names=None,
        fields=None,
        include_fields=None,
        sort='name:1',
        per_page=100,
    ):
        page = 0
        while True:
            found = 0
            for template in self.get(
                template_names=template_names,
                fields=fields,
                include_fields=include_fields,
                sort=sort,
                per_page=per_page,
                page=page,
            ):
                yield template
                found += 1

            page += 1
            if not found:
                break

    @query_params('fields', 'include_fields', 'page', 'per_page', 'sort')
    def get(
        self, template_names=None, params=None
    ):
        return self.transport.perform_request('GET', _make_path(
            'templates', template_names), params=params)

    @query_params()
    def exists(
        self, template_name, params=None
    ):
        if template_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'template_name'.")
        try:
            return self.transport.perform_request('HEAD', _make_path(
                'templates', template_name), params=params)
        except TransportError:
            return False

    @query_params('name')
    def create(
        self, settings, params=None
    ):
        for param in (settings):
            if param in SKIP_IN_PATH:
                raise ValueError("Empty value passed for a required argument.")
        return self.transport.perform_request(
            'POST', '/templates', params=params, body=settings)

    @query_params()
    def delete(
        self, template_name, params=None
    ):
        if template_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'template_name'.")
        return self.transport.perform_request('DELETE', _make_path(
            'templates', template_name), params=params)
