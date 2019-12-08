import loudml_py as loudml
import os
import unittest
from datetime import datetime
import signal
import sys

from loudml_py.randevents import (
    SinEventGenerator,
)
from loudml_py.misc import make_datetime
from loudml_py.wave import (
    generate_data,
    dump_to_bucket,
)
import logging


def poll_job(job):
    def cancel_job_handler(*args):
        job.cancel()
        print('Signal received. Canceled job: ', job.id)
        sys.exit()

    saved_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, cancel_job_handler)
    job.wait(interval=1)
    signal.signal(
        signal.SIGINT, saved_sigint_handler)


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        hosts = os.environ.get(
            'LOUDML_HOSTS', 'localhost:8077').split(',')
        cls.loud = loudml.Loud(hosts=hosts)
        cls.bucket_name = 'nose'
        settings = {
            'name': cls.bucket_name,
            'type': 'influxdb',
            'addr': os.environ.get('INFLUXDB_ADDR', 'localhost:8086'),
            'database': 'nose',
            'retention_policy': 'autogen',
            'measurement': 'loudml'
        }
        cls.loud.buckets.create(
            settings=settings,
        )
        ts_generator = SinEventGenerator(
            base=5,
            amplitude=5,
            trend=0,
            period=180,
            sigma=1,
        )
        from_date = make_datetime("now-1d")
        to_date = make_datetime("now")
        print("generating data from {} to {}".format(from_date, to_date))
        generator = generate_data(
            ts_generator,
            from_date.timestamp(),
            to_date.timestamp(),
            1000,
            0,
            0,
            'value',
        )
        cls.loud.buckets.clear(cls.bucket_name)
        dump_to_bucket(
            cls.loud,
            generator,
            cls.bucket_name,
            tags=None,
            verbose=False,
        )

    def test_bucket_not_found(self):
        with self.assertRaises(loudml.errors.NotFoundError):
            self.loud.buckets.get(
                bucket_names=['no_bucket'],
                fields=['settings'],
                include_fields=True,
            )
            self.loud.buckets.delete(
                bucket_name='no_bucket',
            )

        self.loud.buckets.delete(
            bucket_name='no_bucket',
            ignore=[404],
        )

    def test_add_del_bucket(self):
        bucket_name = 'test_bucket-{}'.format(
            int(datetime.utcnow().timestamp()))
        new_settings = {
            'name': bucket_name,
            'type': 'influxdb',
            'addr': os.environ.get('INFLUXDB_ADDR', 'localhost:8086'),
            'database': 'telegraf',
            'retention_policy': 'autogen',
            'measurement': 'loudml'
        }
        self.loud.buckets.create(
            settings=new_settings,
        )
        buckets = self.loud.buckets.get(
            bucket_names=[bucket_name],
            fields=['name', 'database'],
            include_fields=True,
        )
        self.assertDictEqual(
            buckets[0],
            {
                'name': new_settings['name'],
                'database': new_settings['database'],
            }
        )
        self.loud.buckets.delete(
            bucket_name=bucket_name,
        )
        with self.assertRaises(loudml.errors.NotFoundError):
            self.loud.buckets.get(
                bucket_names=[bucket_name],
            )

    def test_scheduled_job_not_found(self):
        with self.assertRaises(loudml.errors.NotFoundError):
            self.loud.scheduled_jobs.get(
                scheduled_job_names=['no_scheduled_job'],
            )
            self.loud.scheduled_jobs.delete(
                scheduled_job_name='no_scheduled_job',
            )

        self.loud.scheduled_jobs.delete(
            scheduled_job_name='no_scheduled_job',
            ignore=[404],
        )

    def test_add_del_scheduled_job(self):
        scheduled_job_name = 'test_scheduled_job-{}'.format(
            int(datetime.utcnow().timestamp()))
        new_settings = {
            'name': scheduled_job_name,
            'relative_url': '/models',
            'method': 'get',
            'params': {
            },
            'every': {
              'count': 5,
              'unit': 'minutes',
            },
        }
        self.loud.scheduled_jobs.create(
            settings=new_settings,
        )
        scheduled_jobs = self.loud.scheduled_jobs.get(
            scheduled_job_names=[scheduled_job_name],
            fields=['name', 'method'],
            include_fields=True,
        )
        self.assertDictEqual(
            scheduled_jobs[0],
            {
                'name': new_settings['name'],
                'method': new_settings['method'],
            }
        )
        self.loud.scheduled_jobs.delete(
            scheduled_job_name=scheduled_job_name,
        )
        with self.assertRaises(loudml.errors.NotFoundError):
            self.loud.scheduled_jobs.get(
                scheduled_job_names=[scheduled_job_name],
            )

    def test_model_not_found(self):
        with self.assertRaises(loudml.errors.NotFoundError):
            self.loud.models.get(
                model_names=['no_model'],
                fields=['settings', 'state'],
                include_fields=True,
            )
            self.loud.models.delete(
                model_name='no_model',
            )

        self.loud.models.delete(
            model_name='no_model',
            ignore=[404],
        )

    def test_add_del_model(self):
        model_name = 'test_model-{}'.format(
            int(datetime.utcnow().timestamp()))
        new_settings = {
            'name': model_name,
            'type': 'donut',
            'offset': 30,
            'interval': 60,
            'bucket_interval': 60,
            'span': 20,
            'features': [{
                'name': 'count_foo',
                'metric': 'count',
                'field': 'foo',
                'default': 0,
            }]
        }
        self.loud.models.create(
            settings=new_settings,
        )
        models = self.loud.models.get(
            model_names=[model_name],
            fields=['settings', 'state'],
            include_fields=True,
        )
        settings = models[0]['settings']
        state = models[0]['state']
        self.assertEqual(state['trained'], False)
        for opt in [
            'name',
            'type',
            'interval',
            'bucket_interval',
            'offset',
            'span',
        ]:
            self.assertEqual(settings[opt], new_settings[opt])
        self.loud.models.delete(
            model_name=model_name,
        )
        with self.assertRaises(loudml.errors.NotFoundError):
            self.loud.models.get(
                model_names=[model_name],
            )

    def test_no_training_data(self):
        model_name = 'test_model-{}'.format(
            int(datetime.utcnow().timestamp()))
        new_settings = {
            'default_bucket': self.bucket_name,
            'name': model_name,
            'type': 'donut',
            'offset': 30,
            'interval': 60,
            'bucket_interval': 60,
            'span': 2,
            'features': [{
                'name': 'count_foo',
                'metric': 'count',
                'field': 'foo',  # absent in this bucket
                'default': 0,
            }]
        }
        self.loud.models.create(
            settings=new_settings,
        )
        job_name = self.loud.models.train(
            model_name=model_name,
            _from='now-1d',
            _to='now',
        )
        self.assertIsNotNone(job_name)
        job = self.loud.jobs.id(job_name)
        poll_job(job)
        self.assertEqual(job.state, 'failed')
        self.assertIn('no data found for time range', job.error)

    def test_training(self):
        model_name = 'test_model-{}'.format(
            int(datetime.utcnow().timestamp()))
        new_settings = {
            'default_bucket': self.bucket_name,
            'name': model_name,
            'type': 'donut',
            'offset': 30,
            'interval': 60,
            'bucket_interval': 60,
            'span': 2,
            'features': [{
                'name': 'count_value',
                'metric': 'count',
                'field': 'value',
                'default': 0,
            }]
        }
        self.loud.models.create(
            settings=new_settings,
        )
        job_name = self.loud.models.train(
            model_name=model_name,
            _from='now-1d',
            _to='now',
            max_evals=1,
            epochs=10,
            _continue=False,
        )
        self.assertIsNotNone(job_name)
        job = self.loud.jobs.id(job_name)
        poll_job(job)
        self.assertEqual(job.state, 'done')

        versions = self.loud.models.versions.get(
            model_name=model_name,
        )
        self.assertEqual(len(versions), 1)
