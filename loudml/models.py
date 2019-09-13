import requests
import logging

from loudml.api import (
    Service,
    Job,
    get_job_id,
)


class ModelService(Service):
    def __init__(
        self,
    ):
        super().__init__(prefix='/models')

    def start_one(
        self,
        name,
        save_output_data=None,
        flag_abnormal_data=None,
    ):
        params = {}
        if save_output_data:
            params['save_prediction'] = bool(save_output_data)
        if flag_abnormal_data:
            params['detect_anomalies'] = bool(flag_abnormal_data)

        return self._do_one(name, '_start', params)

    def start_many(
        self,
        names,
        save_output_data=None,
        flag_abnormal_data=None,
    ):
        params = {}
        if save_output_data:
            params['save_prediction'] = bool(save_output_data)
        if flag_abnormal_data:
            params['detect_anomalies'] = bool(flag_abnormal_data)

        return self._do_many(names, '_start', params)

    def stop_one(
        self,
        name,
    ):
        return self._do_one(name, '_stop')

    def stop_many(
        self,
        names,
    ):
        return self._do_many(names, '_stop')

    def train_one(
        self,
        model_name,
        from_date,
        to_date,
        max_evals=10,
        epochs=None,
        resume=None,
        start_when_done=None,
    ):
        params = {
            'from': from_date,
            'to': to_date,
        }
        if max_evals:
            params['max_evals'] = int(max_evals)
        if epochs:
            params['epochs'] = int(epochs)
        if resume:
            params['continue'] = bool(resume)
        if start_when_done:
            params['autostart'] = bool(start_when_done)

        response = requests.post(
            self._loud._format_url(
                '{}/{}/_train'.format(self._prefix, model_name), params)
        )
        response.raise_for_status()
        if response.ok:
            job_id = get_job_id(response)
            return Job(
                job_id,
                self._loud,
                name='training({})'.format(model_name),
                total=max_evals,
            )
        else:
            logging.error(response.text)
            return None

    def predict_one(
        self,
        model_name,
        from_date,
        to_date,
        input_bucket=None,
        output_bucket=None,
        save_output_data=None,
        flag_abnormal_data=None,
    ):
        params = {
            'from': from_date,
            'to': to_date,
            'bg': True,
        }
        if input_bucket:
            params['input_bucket'] = str(input_bucket)
        if output_bucket:
            params['output_bucket'] = str(output_bucket)
        if save_output_data:
            params['save_prediction'] = bool(save_output_data)
        if flag_abnormal_data:
            params['detect_anomalies'] = bool(flag_abnormal_data)

        response = requests.post(
            self._loud._format_url(
                '{}/{}/_predict'.format(self._prefix, model_name), params)
        )
        response.raise_for_status()
        if response.ok:
            job_id = get_job_id(response)
            return Job(
                job_id,
                self._loud,
                name='predict({})'.format(model_name),
                total=1,
            )
        else:
            logging.error(response.text)
            return None

    def forecast_one(
        self,
        model_name,
        from_date,
        to_date,
        input_bucket=None,
        output_bucket=None,
        save_output_data=None,
        p_val=None,
        constraint=None,
    ):
        params = {
            'from': from_date,
            'to': to_date,
            'bg': True,
        }
        if input_bucket:
            params['input_bucket'] = str(input_bucket)
        if output_bucket:
            params['output_bucket'] = str(output_bucket)
        if save_output_data:
            params['save_prediction'] = bool(save_output_data)
        if p_val:
            params['p_val'] = float(p_val)
        if constraint:
            params['constraint'] = "{}:{}:{}".format(
                constraint['feature'],
                constraint['type'],
                constraint['threshold'],
            )

        response = requests.post(
            self._loud._format_url(
                '{}/{}/_forecast'.format(self._prefix, model_name), params)
        )
        response.raise_for_status()
        if response.ok:
            job_id = get_job_id(response)
            return Job(
                job_id,
                self._loud,
                name='forecast({})'.format(model_name),
                total=1,
            )
        else:
            logging.error(response.text)
            return None

    def get_latent_data(
        self,
        model_name,
        from_date,
        to_date,
    ):
        params = {
            'from': from_date,
            'to': to_date,
        }

        response = requests.post(
            self._loud._format_url(
                '{}/{}/_latent'.format(self._prefix, model_name), params)
        )
        response.raise_for_status()
        if response.ok:
            job_id = get_job_id(response)
            return Job(
                job_id,
                self._loud,
                name='latent({})'.format(model_name),
                total=1,
            )
        else:
            logging.error(response.text)
            return None

    def create_version(
        self,
        model_name,
    ):
        response = requests.post(
            self._loud._format_url(
                '{}/{}/versions'.format(self._prefix, model_name))
        )
        response.raise_for_status()
        if not response.ok:
            logging.error(response.text)
            return None

        return response.json()

    def load_version(
        self,
        model_name,
        version,
    ):
        params = {
            'version': str(version),
        }
        response = requests.post(
            self._loud._format_url(
                '{}/{}/_restore'.format(self._prefix, model_name),
                params)
        )
        response.raise_for_status()
        if not response.ok:
            logging.error(response.text)

    def get_versions(
        self,
        model_name,
    ):
        response = requests.get(
            self._loud._format_url(
                '{}/{}/versions'.format(self._prefix, model_name))
        )
        response.raise_for_status()
        if not response.ok:
            logging.error(response.text)
            return None
        return response.json()
