from loudml.utils import (
    NamespacedClient, query_params, _make_path, SKIP_IN_PATH
)
from loudml.errors import TransportError
import time


class JobClient(NamespacedClient):
    def __init__(
        self, client, job_id
    ):
        super().__init__(client)
        self.id = str(job_id)
        self._name = str(job_id)
        self._state = None
        self._error = None
        self._progress = None
        self._remaining_time = None
        self._total = None

    @query_params('fields', 'include_fields')
    def get(
        self, params=None
    ):
        return self.transport.perform_request('GET', _make_path(
            'jobs', self.id), params=params)

    @query_params()
    def exists(
        self, params=None
    ):
        try:
            return self.transport.perform_request('HEAD', _make_path(
                'jobs', self.id), params=params)
        except TransportError:
            return False

    @query_params()
    def cancel(
        self, params=None
    ):
        return self.transport.perform_request('POST', _make_path(
            'jobs', self.id, '_cancel'), params=params)

    def fetch_result(self):
        job = self.get(
            include_fields=True,
            fields=['result'],
        )[0]
        return job['result']

    def fetch(self):
        job = self.get(
            include_fields=False,
            fields=['result'],
        )[0]

        self._state = job['state']
        self._error = job.get('error')
        self._progress = job.get('progress')
        self._remaining_time = job.get('remaining_time')
        self.reset_total()
        if 'model' in job:
            self._name = '{}({})'.format(job['type'], job['model'])
        else:
            self._name = '{}({})'.format(job['type'], self.id[:6])

    def success(self):
        return self._state == 'done'

    def done(self):
        return self._state in ['done', 'failed', 'canceled']

    def reset_total(self):
        info = self._progress
        if info:
            self._total = int(info['max_evals'])

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self._state

    @property
    def error(self):
        return self._error

    @property
    def total(self):
        return self._total

    @property
    def step(self):
        info = self._progress
        if not info:
            if self.done():
                return 1
            else:
                return 0
        else:
            return min(self._total, int(info['eval']))

    def wait(self, interval, callback=None):
        while not self.done():
            time.sleep(interval)
            self.fetch()
            if callback:
                callback()


class JobsClient(NamespacedClient):
    def generator(
        self,
        job_names=None,
        fields=None,
        include_fields=None,
        sort='id:1',
        per_page=100,
    ):
        page = 0
        while True:
            found = 0
            for job in self.get(
                job_names=job_names,
                fields=fields,
                include_fields=include_fields,
                sort=sort,
                per_page=per_page,
                page=page,
            ):
                yield job
                found += 1

            page += 1
            if not found:
                break

    def id(self, job_id):
        return JobClient(self, job_id)

    @query_params('fields', 'include_fields', 'page', 'per_page', 'sort')
    def get(
        self, job_names=None, params=None
    ):
        return self.transport.perform_request('GET', _make_path(
            'jobs', job_names), params=params)

    @query_params()
    def exists(
        self, job_name, params=None
    ):
        if job_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'job_name'.")
        try:
            return self.transport.perform_request('HEAD', _make_path(
                'jobs', job_name), params=params)
        except TransportError:
            return False

    @query_params()
    def cancel_job(
        self, job_name, params=None
    ):
        if job_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'job_name'.")
        return self.transport.perform_request('POST', _make_path(
            'jobs', job_name, '_cancel'), params=params)

    @query_params()
    def cancel_jobs(
        self, job_names, params=None
    ):
        if job_names in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'job_names'.")
        return self.transport.perform_request('POST', _make_path(
            'jobs', job_names, '_cancel'), params=params)
