from loudml_py.utils import (
    NamespacedClient, query_params, _make_path, SKIP_IN_PATH
)
from loudml_py.errors import TransportError


class ModelVersionsClient(NamespacedClient):
    @query_params('version')
    def load(
        self, model_name, params=None
    ):
        if model_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'model_name'.")
        return self.transport.perform_request('POST', _make_path(
            'models', model_name, '_restore'), params=params)

    def generator(
        self,
        model_name=None,
        fields=None,
        include_fields=None,
        sort=None,
        per_page=100,
    ):
        page = 0
        while True:
            found = 0
            for model in self.get(
                model_name=model_name,
                fields=fields,
                include_fields=include_fields,
                sort=sort,
                per_page=per_page,
                page=page,
            ):
                yield model
                found += 1

            page += 1
            if not found:
                break

    @query_params('fields', 'include_fields', 'page', 'per_page', 'sort')
    def get(
        self, model_name, params=None
    ):
        if model_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'model_name'.")
        return self.transport.perform_request('GET', _make_path(
            'models', model_name, 'versions'), params=params)

    @query_params()
    def exists(
        self, model_name, version_name, params=None
    ):
        for param in (model_name, version_name):
            if param in SKIP_IN_PATH:
                raise ValueError("Empty value passed for a required argument.")
        try:
            return self.transport.perform_request('HEAD', _make_path(
                'models', model_name, 'versions', version_name), params=params)
        except TransportError:
            return False


class ModelsClient(NamespacedClient):
    def __init__(self, client):
        super().__init__(client)
        self.versions = ModelVersionsClient(client)

    def generator(
        self,
        model_names=None,
        fields=None,
        include_fields=None,
        sort='name:1',
        per_page=100,
    ):
        page = 0
        while True:
            found = 0
            for model in self.get(
                model_names=model_names,
                fields=fields,
                include_fields=include_fields,
                sort=sort,
                per_page=per_page,
                page=page,
            ):
                yield model
                found += 1

            page += 1
            if not found:
                break

    @query_params('fields', 'include_fields', 'page', 'per_page', 'sort')
    def get(
        self, model_names=None, params=None
    ):
        return self.transport.perform_request('GET', _make_path(
            'models', model_names), params=params)

    @query_params()
    def exists(
        self, model_name, params=None
    ):
        if model_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'model_name'.")
        try:
            return self.transport.perform_request('HEAD', _make_path(
                'models', model_name), params=params)
        except TransportError:
            return False

    @query_params('from_template')
    def create(
        self, settings, params=None
    ):
        for param in (settings):
            if param in SKIP_IN_PATH:
                raise ValueError("Empty value passed for a required argument.")
        return self.transport.perform_request(
            'POST', '/models', params=params, body=settings)

    @query_params()
    def delete(
        self, model_name, params=None
    ):
        if model_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'model_name'.")
        return self.transport.perform_request('DELETE', _make_path(
            'models', model_name), params=params)

    @query_params(
        'from', 'to', 'epochs', 'max_evals', 'continue'
    )
    def train(
        self, model_name, params=None
    ):
        if model_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'model_name'.")

        if not params:
            params = {'max_evals': 10}
        elif 'max_evals' not in params:
            params['max_evals'] = 10

        response = self.transport.perform_request('POST', _make_path(
            'models', model_name, '_train'), params=params)
        return response

    @query_params(
        'from', 'to', 'bg', 'input_bucket', 'output_bucket',
        'save_output_data', 'flag_abnormal_data'
    )
    def eval_model(
        self, model_name, params=None
    ):
        if model_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'model_name'.")
        if not params:
            params = {'bg': True}
        elif 'bg' not in params:
            params['bg'] = True
        response = self.transport.perform_request('POST', _make_path(
            'models', model_name, '_eval'), params=params)
        return response

    @query_params(
        'from', 'to', 'bg', 'input_bucket', 'output_bucket',
        'p_val', 'constraint', 'save_output_data'
    )
    def forecast(
        self, model_name, params=None
    ):
        if model_name in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'model_name'.")
        if not params:
            params = {'bg': True}
        elif 'bg' not in params:
            params['bg'] = True
        response = self.transport.perform_request('POST', _make_path(
            'models', model_name, '_forecast'), params=params)
        return response

    @query_params('save_output_data', 'flag_abnormal_data')
    def start_inference(
        self, model_names, params=None
    ):
        if model_names in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'model_names'.")
        return self.transport.perform_request('POST', _make_path(
            'models', model_names, '_start'), params=params)

    @query_params()
    def stop_inference(
        self, model_names, params=None
    ):
        if model_names in SKIP_IN_PATH:
            raise ValueError(
                "Empty value passed for a required argument 'model_names'.")
        return self.transport.perform_request('POST', _make_path(
            'models', model_names, '_stop'), params=params)
