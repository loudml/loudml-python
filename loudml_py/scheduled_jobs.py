from loudml_py.utils import (
    NamespacedClient, query_params, _make_path, SKIP_IN_PATH
)
from loudml_py.errors import TransportError


class ScheduledJobsClient(NamespacedClient):
    def generator(
        self,
        scheduled_job_names=None,
        fields=None,
        include_fields=None,
        sort='name:1',
        per_page=100,
    ):
        page = 0
        while True:
            found = 0
            for scheduled_job in self.get(
                scheduled_job_names=scheduled_job_names,
                fields=fields,
                include_fields=include_fields,
                sort=sort,
                per_page=per_page,
                page=page,
            ):
                yield scheduled_job
                found += 1

            page += 1
            if not found:
                break

    @query_params('fields', 'include_fields', 'page', 'per_page', 'sort')
    def get(
        self, scheduled_job_names=None, params=None
    ):
        return self.transport.perform_request('GET', _make_path(
            'scheduled_jobs', scheduled_job_names), params=params)

    @query_params()
    def exists(
        self, scheduled_job_name, params=None
    ):
        if scheduled_job_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'scheduled_job_name'.")
        try:
            return self.transport.perform_request('HEAD', _make_path(
                'scheduled_jobs', scheduled_job_name), params=params)
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
            'POST', '/scheduled_jobs', params=params, body=settings)

    @query_params()
    def delete(
        self, scheduled_job_name, params=None
    ):
        if scheduled_job_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'scheduled_job_name'.")
        return self.transport.perform_request('DELETE', _make_path(
            'scheduled_jobs', scheduled_job_name), params=params)
