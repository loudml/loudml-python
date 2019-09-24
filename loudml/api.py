import pkg_resources
import logging
import requests
import time

from urllib.parse import (
    urlencode,
    quote,
)
from loudml.misc import (
    format_points,
)


g_pagination_count = 100


def get_job_id(response):
    job_id = response.text.rstrip("\r\n")
    if job_id.startswith('"') and job_id.endswith('"'):
        job_id = job_id[1:-1]

    return job_id


class Factory():
    @classmethod
    def load(cls, service_name):
        for ep in pkg_resources.iter_entry_points('loudml.services'):
            if ep.name == service_name:
                return ep.load()()
        return None


class Job():
    def __init__(
        self,
        job_id,
        loud,
        name=None,
        total=1,
    ):
        self.id = job_id
        self._state = None
        self._error = None
        self._progress = None
        self._total = int(total)
        self._name = name
        self._tqdm = None
        self.jobs = Factory.load('jobs')
        self.jobs.set_loudml_target(loud)

    @classmethod
    def fetch_all(cls, loud, jobs):
        if not len(jobs):
            return
        statuses = loud.jobs.get(
            names=[job.id for job in jobs],
            include_fields=False,
            fields=['result'],
        )
        for status, job in zip(statuses, jobs):
            old_step = job.step
            job._state = status['state']
            job._error = status.get('error')
            job._progress = status.get('progress')
            job._remaining_time = status.get('remaining_time')
            if not job.done():
                job.reset_total()
            new_step = job.step
            if job._tqdm and new_step != old_step:
                job._tqdm.update(new_step - old_step)

    def fetch_result(self):
        jobs = self.jobs.get(
            names=[str(self.id)],
            include_fields=True,
            fields=['result'],
        )
        if not len(jobs):
            raise Exception('FIXME')

        return jobs[0]['result']

    def fetch(self):
        jobs = self.jobs.get(
            names=[str(self.id)],
            include_fields=False,
            fields=['result'],
        )
        if not len(jobs):
            raise Exception('FIXME')

        self._state = jobs[0]['state']
        self._error = jobs[0].get('error')
        self._progress = jobs[0].get('progress')
        self._remaining_time = jobs[0].get('remaining_time')

    def cancel(self):
        self.jobs.cancel_one(
            str(self.id))

    def success(self):
        return self._state == 'done'

    def done(self):
        return self._state in ['done', 'failed', 'canceled']

    def set_tqdm(self, t):
        self._tqdm = t

    def reset_total(self):
        if not self._tqdm:
            return
        info = self._progress
        if not info:
            self._tqdm.reset(total=1)
        else:
            self._total = int(info['max_evals'])
            self._tqdm.reset(total=self._total)

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
    def tqdm(self):
        return self._tqdm

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


class Service():
    def __init__(
        self,
        prefix,
    ):
        self._prefix = prefix

    def set_loudml_target(
        self,
        loud,
    ):
        self._loud = loud

    def get(
        self,
        names=None,
        fields=None,
        include_fields=None,
        page=0,
        per_page=100,
        sort=None,
    ):
        params = {
            'include_fields': bool(include_fields),
            'page': int(page),
            'per_page': int(per_page),
            'sort': sort,
        }
        if fields:
            params['fields'] = ";".join(fields)
        if not names:
            response = requests.get(
                self._loud._format_url(
                    self._prefix, params)
            )
        else:
            response = requests.get(
                self._loud._format_url(
                    '{}/{}'.format(self._prefix, ";".join(names)),
                    params)
            )
        if response.ok:
            return response.json()
        else:
            response.raise_for_status()

    def exists(
        self,
        name,
    ):
        response = requests.head(
            self._loud._format_url(
                '{}/{}'.format(self._prefix, name))
        )
        return response.ok

    def del_one(
        self,
        name,
    ):
        response = requests.delete(
            self._loud._format_url(
                '{}/{}'.format(self._prefix, name))
        )
        if not response.ok:
            logging.error(response.text)
        response.raise_for_status()

    def del_many(
        self,
        names,
    ):
        response = requests.delete(
            self._loud._format_url(
                '{}/{}'.format(self._prefix, ";".join(names)))
        )
        if not response.ok:
            logging.error(response.text)
        response.raise_for_status()

    def create(
        self,
        settings,
        options=None,
    ):
        response = requests.post(
            self._loud._format_url(
                self._prefix, options),
            json=settings,
        )
        if not response.ok:
            logging.error(response.text)
        response.raise_for_status()

    def _do_one(
        self,
        name,
        action,
        params=None,
        content=None,
    ):
        response = requests.post(
            self._loud._format_url(
                '{}/{}/{}'.format(self._prefix, name, action), params),
            json=content,
        )
        response.raise_for_status()
        return response

    def _do_many(
        self,
        names,
        action,
        params=None,
        content=None,
    ):
        response = requests.post(
            self._loud._format_url(
                '{}/{}/{}'.format(self._prefix, ";".join(names), action),
                params),
            json=content,
        )
        response.raise_for_status()
        return response


class Loud():
    def __init__(
        self,
        loudml_host,
        loudml_port,
        enable_ssl=False,
    ):
        self._host = loudml_host
        self._port = loudml_port
        self._enable_ssl = enable_ssl

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

    @property
    def scheduled_jobs(self):
        service = Factory.load('scheduled_jobs')
        service.set_loudml_target(self)
        return service

    @property
    def jobs(self):
        service = Factory.load('jobs')
        service.set_loudml_target(self)
        return service

    @property
    def buckets(self):
        service = Factory.load('buckets')
        service.set_loudml_target(self)
        return service

    @property
    def models(self):
        service = Factory.load('models')
        service.set_loudml_target(self)
        return service

    @property
    def templates(self):
        service = Factory.load('templates')
        service.set_loudml_target(self)
        return service

    def version(self):
        response = requests.get(
            self._format_url('/')
        )
        response.raise_for_status()
        if not response.ok:
            logging.error(response.text)
        else:
            return response.json().get('version')

    def bucket_generator(
        self,
        bucket_names=None,
        fields=None,
        include_fields=None,
        sort='name:1',
    ):
        global g_pagination_count
        page = 0
        while True:
            found = 0
            for bucket in self.get_buckets(
                bucket_names=bucket_names,
                fields=fields,
                include_fields=include_fields,
                page=page,
                per_page=g_pagination_count,
                sort=sort,
            ):
                yield bucket
                found += 1

            page += 1
            if not found:
                break

    def get_buckets(
        self,
        bucket_names=None,
        fields=None,
        include_fields=None,
        page=0,
        per_page=100,
        sort='name:1',
    ):
        return self.buckets.get(
            names=bucket_names,
            fields=fields,
            include_fields=include_fields,
            page=page,
            per_page=per_page,
            sort=sort,
        )

    def bucket_exists(
        self,
        bucket_name,
    ):
        return self.buckets.exists(
            name=bucket_name)

    def delete_bucket(
        self,
        bucket_name,
    ):
        return self.buckets.del_one(
            name=bucket_name)

    def create_bucket(
        self,
        settings,
    ):
        return self.buckets.create(
            settings=settings)

    def clear_bucket(
        self,
        bucket_name,
    ):
        return self.buckets.clear(
            bucket_name)

    def write_bucket(
        self,
        bucket_name,
        points,
        batch_size,
        **kwargs
    ):
        return self.buckets.write(
            bucket_name=bucket_name,
            points=points,
            batch_size=batch_size,
            **kwargs
        )

    def read_bucket(
        self,
        bucket_name,
        from_date,
        to_date,
        bucket_interval,
        features,
    ):
        return self.buckets.read(
            bucket_name=bucket_name,
            from_date=from_date,
            to_date=to_date,
            bucket_interval=bucket_interval,
            features=features,
        )

    def model_generator(
        self,
        model_names=None,
        fields=None,
        include_fields=None,
        sort='name:1',
    ):
        global g_pagination_count
        page = 0
        while True:
            found = 0
            for model in self.get_models(
                model_names=model_names,
                fields=fields,
                include_fields=include_fields,
                page=page,
                per_page=g_pagination_count,
                sort=sort,
            ):
                yield model
                found += 1

            page += 1
            if not found:
                break

    def get_models(
        self,
        model_names=None,
        fields=None,
        include_fields=None,
        page=0,
        per_page=100,
        sort='name:1',
    ):
        return self.models.get(
            names=model_names,
            fields=fields,
            include_fields=include_fields,
            page=page,
            per_page=per_page,
            sort=sort,
        )

    def model_exists(
        self,
        model_name,
    ):
        return self.models.exists(
            name=model_name)

    def delete_model(
        self,
        model_name,
    ):
        return self.models.del_one(
            name=model_name)

    def create_model(
        self,
        settings,
        template_name=None,
    ):
        params = {}
        if template_name:
            params['from_template'] = template_name

        return self.models.create(
            settings=settings,
            options=params)

    def start_inference(
        self,
        model_names,
        save_output_data=None,
        flag_abnormal_data=None,
    ):
        return self.models.start_many(
            names=model_names,
            save_output_data=save_output_data,
            flag_abnormal_data=flag_abnormal_data)

    def stop_inference(
        self,
        model_names,
    ):
        return self.models.stop_many(
            names=model_names)

    def train_model(
        self,
        model_name,
        from_date,
        to_date,
        max_evals=None,
        epochs=None,
        resume=None,
    ):
        return self.models.train_one(
            model_name=model_name,
            from_date=from_date,
            to_date=to_date,
            max_evals=max_evals,
            epochs=epochs,
            resume=resume,
        )

    def predict_model(
        self,
        model_name,
        from_date,
        to_date,
        input_bucket=None,
        output_bucket=None,
        save_output_data=None,
        flag_abnormal_data=None,
    ):
        return self.models.predict_one(
            model_name=model_name,
            from_date=from_date,
            to_date=to_date,
            input_bucket=input_bucket,
            output_bucket=output_bucket,
            save_output_data=save_output_data,
            flag_abnormal_data=flag_abnormal_data,
        )

    def forecast_model(
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
        return self.models.forecast_one(
            model_name=model_name,
            from_date=from_date,
            to_date=to_date,
            input_bucket=input_bucket,
            output_bucket=output_bucket,
            save_output_data=save_output_data,
            p_val=p_val,
            constraint=constraint,
        )

    def get_model_latent_data(
        self,
        model_name,
        from_date,
        to_date,
    ):
        return self.models.get_latent_data(
            model_name=model_name,
            from_date=from_date,
            to_date=to_date,
        )

    def load_model_version(
        self,
        model_name,
        version,
    ):
        return self.models.load_version(
            model_name=model_name,
            version=version,
        )

    def model_versions_generator(
        self,
        model_name=None,
        fields=None,
        include_fields=None,
        sort='name:1',
    ):
        global g_pagination_count
        page = 0
        while True:
            found = 0
            for model in self.get_model_versions(
                model_name=model_name,
                fields=fields,
                include_fields=include_fields,
                page=page,
                per_page=g_pagination_count,
                sort=sort,
            ):
                yield model
                found += 1

            page += 1
            if not found:
                break

    def get_model_versions(
        self,
        model_name,
        fields=None,
        include_fields=None,
        page=0,
        per_page=100,
        sort='name:1',
    ):
        return self.models.get_versions(
            model_name=model_name,
            fields=fields,
            include_fields=include_fields,
            page=page,
            per_page=per_page,
            sort=sort,
        )

    def template_generator(
        self,
        template_names=None,
        fields=None,
        include_fields=None,
        sort='name:1',
    ):
        global g_pagination_count
        page = 0
        while True:
            found = 0
            for template in self.get_templates(
                template_names=template_names,
                fields=fields,
                include_fields=include_fields,
                page=page,
                per_page=g_pagination_count,
                sort=sort,
            ):
                yield template
                found += 1

            page += 1
            if not found:
                break

    def get_templates(
        self,
        template_names=None,
        fields=None,
        include_fields=None,
        page=0,
        per_page=100,
        sort='name:1',
    ):
        return self.templates.get(
            names=template_names,
            fields=fields,
            include_fields=include_fields,
            page=page,
            per_page=per_page,
            sort=sort,
        )

    def template_exists(
        self,
        template_name,
    ):
        return self.templates.exists(
            name=template_name)

    def delete_template(
        self,
        template_name,
    ):
        return self.templates.del_one(
            name=template_name)

    def create_template(
        self,
        settings,
        template_name,
    ):
        params = {
            'name': template_name,
        }
        return self.templates.create(
            settings=settings, options=params)

    def job_generator(
        self,
        job_names=None,
        fields=None,
        include_fields=None,
        sort='id:1',
    ):
        global g_pagination_count
        page = 0
        while True:
            found = 0
            for job in self.get_jobs(
                job_names=job_names,
                fields=fields,
                include_fields=include_fields,
                page=page,
                per_page=g_pagination_count,
                sort=sort,
            ):
                yield job
                found += 1

            page += 1
            if not found:
                break

    def get_jobs(
        self,
        job_names=None,
        fields=None,
        include_fields=None,
        page=0,
        per_page=100,
        sort='id:1',
    ):
        return self.jobs.get(
            names=job_names,
            fields=fields,
            include_fields=include_fields,
            page=page,
            per_page=per_page,
            sort=sort,
        )

    def job_exists(
        self,
        job_name,
    ):
        return self.jobs.exists(
            name=job_name)

    def cancel_job(
        self,
        job_name,
    ):
        return self.jobs.cancel_one(
            name=job_name)

    def cancel_jobs(
        self,
        job_names,
    ):
        return self.jobs.cancel_many(
            names=job_names)

    def scheduled_job_generator(
        self,
        scheduled_job_names=None,
        fields=None,
        include_fields=None,
        sort='name:1',
    ):
        global g_pagination_count
        page = 0
        while True:
            found = 0
            for scheduled_job in self.get_scheduled_jobs(
                scheduled_job_names=scheduled_job_names,
                fields=fields,
                include_fields=include_fields,
                page=page,
                per_page=g_pagination_count,
                sort=sort,
            ):
                yield scheduled_job
                found += 1

            page += 1
            if not found:
                break

    def get_scheduled_jobs(
        self,
        scheduled_job_names=None,
        fields=None,
        include_fields=None,
        page=0,
        per_page=100,
        sort='name:1',
    ):
        return self.scheduled_jobs.get(
            names=scheduled_job_names,
            fields=fields,
            include_fields=include_fields,
            page=page,
            per_page=per_page,
            sort=sort,
        )

    def scheduled_job_exists(
        self,
        scheduled_job_name,
    ):
        return self.scheduled_jobs.exists(
            name=scheduled_job_name)

    def delete_scheduled_job(
        self,
        scheduled_job_name,
    ):
        return self.scheduled_jobs.del_one(
            name=scheduled_job_name)

    def create_scheduled_job(
        self,
        settings,
    ):
        return self.scheduled_jobs.create(
            settings=settings)

    def write_points(
        self, bucket_name, points, verbose=False, interval=1
    ):
        if verbose:
            for line in format_points(points):
                print(line)
        for job in self.write_bucket(
            bucket_name=bucket_name,
            points=points,
            batch_size=len(points),
        ):
            while not job.done():
                time.sleep(interval)
                job.fetch()
