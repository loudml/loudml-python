import loudml_py as loudml
import os
import unittest
from datetime import datetime


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hosts = os.environ.get(
            'LOUDML_HOSTS', 'localhost:8077').split(',')
        cls.loud = loudml.Loud(hosts=hosts)

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
