"""
Loud ML command line tool
"""

import argparse
import signal
import sys
import os
import yaml
import json
import logging
import time
import math
import readline  # noqa: must be loaded (shlex requirement)
import shlex
from tqdm import tqdm
import requests.exceptions

from loudml.api import (
    Loud,
)
from loudml.errors import (
   LoudMLException,
)
from loudml.misc import (
    parse_constraint,
    parse_addr,
    format_buckets,
    format_model_versions,
)


def poll_job(job):
    def cancel_job_handler(*args):
        job.cancel()
        print('Signal received. Canceled job: ', job.id)
        sys.exit()

    signal.signal(signal.SIGINT, cancel_job_handler)

    t = tqdm(desc=job.name, total=job.total)
    last_step = 0
    while True:
        time.sleep(1)
        job.fetch()
        new_step = job.step
        t.update(new_step - last_step)
        last_step = new_step
        if job.done():
            return


class Command:
    """
    Command base class
    """
    def __init__(self):
        self._config = None

    @classmethod
    def description(cls):
        return cls.__doc__

    def set_config(
        self,
        addr,
        quiet=False,
    ):
        """
        Set loudml hostname
        """
        res = parse_addr(addr, default_port=8077)
        self._config = {
            'loudml_host': res['host'],
            'loudml_port': res['port'],
        }
        self._quiet = quiet

    @property
    def quiet(self):
        return self._quiet

    @property
    def config(self):
        if self._config is None:
            self._config = {
                'loudml_host': 'localhost',
                'loudml_port': 8077,
            }
        return self._config

    def add_args(self, parser):
        """
        Declare command arguments
        """

    def exec(self, args):
        """
        Execute command
        """
        if getattr(args, 'model_name') == '*':
            return self.exec_all(args)

    def exec_all(self, args):
        """
        Execute command
        """


class LoadVersionCommand(Command):
    """Restore trained model version."""
    @property
    def short_name(self):
        return 'load-model-version'

    def add_args(self, parser):
        parser.add_argument(
            '-v', '--version',
            help="Version name",
            type=str,
        )
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )

    def exec(self, args):
        if not args.version:
            raise LoudMLException(
                "'version' argument is required")

        loud = Loud(**self.config)
        loud.load_model_version(args.model_name, args.version)


class ListVersionsCommand(Command):
    """List trained model versions and state."""
    @property
    def short_name(self):
        return 'list-model-versions'

    def add_args(self, parser):
        parser.add_argument(
            '-a', '--all',
            help="All internal information",
            action='store_true',
            dest='show_all',
        )
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )

    def exec(self, args):
        loud = Loud(**self.config)
        if args.show_all:
            include_fields = None
            fields = None
        else:
            include_fields = True
            fields = ['state', 'version']

        models = loud.get_model_versions(
            model_name=args.model_name,
            fields=fields,
            include_fields=include_fields,
        )
        if not len(models):
            print('Not found:', args.model_name)
            exit(1)
        if args.show_all:
            print(yaml.dump(models, indent=2))
        else:
            for line in format_model_versions(models):
                print(line)


class CreateModelCommand(Command):
    """Create a new model."""
    @property
    def short_name(self):
        return 'create-model'

    def add_args(self, parser):
        parser.add_argument(
            '-t', '--template-name',
            help="Template name",
            type=str,
        )
        parser.add_argument(
            'model_file',
            help="Model file",
            type=str,
        )
        parser.add_argument(
            '-f', '--force',
            help="Overwrite if present (warning: training data will be lost!)",
            action='store_true',
        )

    def _load_model_json(self, path):
        """
        Load model JSON
        """
        with open(path) as model_file:
            return json.load(model_file)

    def _load_model_yaml(self, path):
        """
        Load model JSON
        """
        try:
            with open(path) as model_file:
                return yaml.load(model_file)
        except OSError as exn:
            raise LoudMLException(exn)
        except yaml.YAMLError as exn:
            raise LoudMLException(exn)

    def load_model_file(self, path):
        """
        Load model file
        """

        _, ext = os.path.splitext(path)
        if ext in [".yaml", ".yml"]:
            settings = self._load_model_yaml(path)
        else:
            settings = self._load_model_json(path)

        return settings

    def exec(self, args):
        loud = Loud(**self.config)
        if args.template_name is not None:
            settings = self._load_model_json(args.model_file)
        else:
            settings = self.load_model_file(args.model_file)

        if args.force and loud.model_exists(settings.get('name')):
            loud.delete_model(settings.get('name'))

        loud.create_model(
            settings=settings, template_name=args.template_name)


class CreateTemplateCommand(CreateModelCommand):
    """Create a new model template."""
    @property
    def short_name(self):
        return 'create-model-template'

    def add_args(self, parser):
        parser.add_argument(
            'template_name',
            help="Template name",
            type=str,
        )
        parser.add_argument(
            'model_file',
            help="Model file",
            type=str,
        )
        parser.add_argument(
            '-f', '--force',
            help="Overwrite if present",
            action='store_true',
        )

    def exec(self, args):
        loud = Loud(**self.config)
        settings = self._load_model_json(args.model_file)

        if args.force and loud.template_exists(args.template_name):
            loud.delete_template(args.template_name)

        loud.create_template(
            settings, args.template_name)


class DeleteTemplateCommand(Command):
    """Delete a model template."""
    @property
    def short_name(self):
        return 'delete-model-template'

    def add_args(self, parser):
        parser.add_argument(
            'template_name',
            help="Template model name",
            type=str,
        )

    def exec(self, args):
        loud = Loud(**self.config)
        loud.delete_template(args.template_name)


class ListTemplatesCommand(Command):
    """List configured model templates."""
    @property
    def short_name(self):
        return 'list-model-templates'

    def add_args(self, parser):
        parser.add_argument(
            '-a', '--all',
            help="All internal information",
            action='store_true',
            dest='show_all',
        )

    def exec(self, args):
        loud = Loud(**self.config)
        templates = loud.get_templates()
        if args.show_all:
            print(yaml.dump(templates, indent=2))
        else:
            print('\n'.join(
                [tmpl['name'] for tmpl in templates]))


class ListJobsCommand(Command):
    """List jobs and query their status."""
    @property
    def short_name(self):
        return 'list-jobs'

    def add_args(self, parser):
        parser.add_argument(
            '-i', '--interactive',
            help="Keep polling interactive job status",
            action='store_true',
        )

    def exec(self, args):
        loud = Loud(**self.config)
        jobs = list(
            loud.job_generator(
                fields=['result'],
                include_fields=False,
            )
        )
        print(jobs)


class CancelJobCommand(Command):
    """Cancel a job."""
    @property
    def short_name(self):
        return 'cancel-job'

    def add_args(self, parser):
        parser.add_argument(
            'job_id',
            help="Job identifier",
            type=str,
        )

    def exec(self, args):
        loud = Loud(**self.config)
        loud.cancel_jobs(
            job_names=[args.job_id])


class CreateBucketCommand(Command):
    """Create a new bucket."""
    @property
    def short_name(self):
        return 'create-bucket'

    def add_args(self, parser):
        parser.add_argument(
            'bucket_file',
            help="Bucket file",
            type=str,
        )
        parser.add_argument(
            '-f', '--force',
            help="Overwrite if present",
            action='store_true',
        )

    def _load_bucket_json(self, path):
        """
        Load bucket JSON
        """
        with open(path) as bucket_file:
            return json.load(bucket_file)

    def _load_bucket_yaml(self, path):
        """
        Load bucket JSON
        """
        try:
            with open(path) as bucket_file:
                return yaml.load(bucket_file)
        except OSError as exn:
            raise LoudMLException(exn)
        except yaml.YAMLError as exn:
            raise LoudMLException(exn)

    def load_bucket_file(self, path):
        """
        Load bucket file
        """

        _, ext = os.path.splitext(path)
        if ext in [".yaml", ".yml"]:
            settings = self._load_bucket_yaml(path)
        else:
            settings = self._load_bucket_json(path)

        return settings

    def exec(self, args):
        loud = Loud(**self.config)
        if args.template is not None:
            settings = self._load_bucket_json(args.bucket_file)
        else:
            settings = self.load_bucket_file(args.bucket_file)

        if args.force and loud.bucket_exists(settings.get('name')):
            loud.delete_bucket(settings.get('name'))

        loud.create_bucket(settings)


class ListBucketsCommand(Command):
    """List configured buckets."""
    @property
    def short_name(self):
        return 'list-buckets'

    def add_args(self, parser):
        parser.add_argument(
            '-a', '--all',
            help="All internal information",
            action='store_true',
            dest='show_all',
        )

    def exec(self, args):
        loud = Loud(**self.config)
        if args.show_all:
            buckets = list(
                loud.bucket_generator(
                    fields=None,
                    include_fields=None,
                )
            )
            print(yaml.dump(buckets, indent=2))
        else:
            for bucket in loud.bucket_generator(
                fields=['name'],
                include_fields=True,
            ):
                print(bucket['name'])


class DeleteBucketCommand(Command):
    """Delete a bucket"""
    @property
    def short_name(self):
        return 'delete-bucket'

    def add_args(self, parser):
        parser.add_argument(
            'bucket_name',
            help="Bucket name",
            type=str,
        )

    def exec(self, args):
        loud = Loud(**self.config)
        loud.delete_bucket(args.bucket_name)


class ShowBucketCommand(Command):
    """Show bucket settings"""
    @property
    def short_name(self):
        return 'show-bucket'

    def add_args(self, parser):
        parser.add_argument(
            'bucket_name',
            help="Bucket name",
            type=str,
        )

    def exec(self, args):
        loud = Loud(**self.config)
        buckets = loud.get_buckets(
            bucket_names=[args.bucket_name],
        )
        if not len(buckets):
            print('Not found:', args.bucket_name)
            exit(1)

        print(yaml.dump(buckets, indent=2))


class ReadBucketCommand(Command):
    """Reads data from a bucket"""
    @property
    def short_name(self):
        return 'read-bucket'

    def add_args(self, parser):
        parser.add_argument(
            'bucket_name',
            help="Bucket name",
            type=str,
        )
        parser.add_argument(
            '-f', '--from',
            help="From date",
            type=str,
            dest='from_date',
        )
        parser.add_argument(
            '-t', '--to',
            help="To date",
            type=str,
            dest='to_date',
        )
        parser.add_argument(
            '-i', '--bucket-interval',
            help="Aggregation bucket interval",
            type=str,
            dest='bucket_interval',
        )
        parser.add_argument(
            '-F', '--features',
            help="Aggregations expression",
            type=str,
            dest='features',
        )
        parser.add_argument(
            '--bg',
            help="Run this request in the background",
            default=False,
            action='store_true',
        )

    def exec(self, args):
        if not args.from_date:
            raise LoudMLException(
                "'from' argument is required")
        if not args.to_date:
            raise LoudMLException(
                "'to' argument is required")

        loud = Loud(**self.config)
        job = loud.read_bucket(
            bucket_name=args.bucket_name,
            from_date=args.from_date,
            to_date=args.to_date,
            bucket_interval=args.bucket_interval,
            features=args.features,
        )
        if args.bg:
            print(job.id)
        else:
            poll_job(job)
            if job.success() and not self.quiet:
                out = job.fetch_result()
                for line in format_buckets(out):
                    print(line)


class WriteBucketCommand(Command):
    """Writes data points to a bucket"""
    @property
    def short_name(self):
        return 'write-bucket'

    def add_args(self, parser):
        parser.add_argument(
            'bucket_name',
            help="Bucket name",
            type=str,
        )
        parser.add_argument(
            '-d', '--data',
            help="Read data from file",
            type=str,
        )
        parser.add_argument(
            '-b', '--batch-size',
            help="Write data in batches",
            type=int,
            default=1000,
        )

    def exec(self, args):
        loud = Loud(**self.config)
        if args.data:
            fd = open(args.data)
        else:
            fd = sys.stdin

        points = json.load(fd)
        fd.close()

        for job in tqdm(loud.write_bucket(
            bucket_name=args.bucket_name,
            points=points,
            batch_size=args.batch_size,
        ), total=math.ceil(len(points) / args.batch_size)):

            def cancel_job_handler(*args):
                job.cancel()
                print('Signal received. Canceled job: ', job.id)
                sys.exit()

            signal.signal(signal.SIGINT, cancel_job_handler)
            while not job.done():
                time.sleep(1)
                job.fetch()


class ClearBucketCommand(Command):
    """Deletes all data points in the given bucket."""
    @property
    def short_name(self):
        return 'clear-bucket'

    def add_args(self, parser):
        parser.add_argument(
            'bucket_name',
            help="Bucket name",
            type=str,
        )

    def exec(self, args):
        loud = Loud(**self.config)
        loud.clear_bucket(
            bucket_name=args.bucket_name,
        )


class ListModelsCommand(Command):
    """List configured models."""
    @property
    def short_name(self):
        return 'list-models'

    def add_args(self, parser):
        parser.add_argument(
            '-a', '--all',
            help="All internal information",
            action='store_true',
            dest='show_all',
        )

    def exec(self, args):
        loud = Loud(**self.config)
        if args.show_all:
            models = list(
                loud.model_generator(
                    fields=None,
                    include_fields=None,
                )
            )
            print(yaml.dump(models, indent=2))
        else:
            for model in loud.model_generator(
                fields=['state'],
                include_fields=False,
            ):
                print(model['settings']['name'])


class DeleteModelCommand(Command):
    """Delete a model"""
    @property
    def short_name(self):
        return 'delete-model'

    def add_args(self, parser):
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )

    def exec(self, args):
        loud = Loud(**self.config)
        loud.delete_model(args.model_name)


class ShowModelCommand(Command):
    """Show model settings and internal state."""
    @property
    def short_name(self):
        return 'show-model'

    def add_args(self, parser):
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )
        parser.add_argument(
            '-a', '--all',
            help="All internal information",
            action='store_true',
            dest='show_all',
        )

    def exec(self, args):
        loud = Loud(**self.config)
        if args.show_all:
            include_fields = None
            fields = None
        else:
            include_fields = False
            fields = ['state']

        models = loud.get_models(
            model_names=[args.model_name],
            fields=fields,
            include_fields=include_fields,
        )
        if not len(models):
            print('Not found:', args.model_name)
            exit(1)

        print(yaml.dump(models, indent=2))


class PlotCommand(Command):
    """Plot model latent space dimensions."""
    @property
    def short_name(self):
        return 'plot-model'

    def add_args(self, parser):
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )
        parser.add_argument(
            '-f', '--from',
            help="From date",
            type=str,
            dest='from_date',
        )
        parser.add_argument(
            '-t', '--to',
            help="To date",
            type=str,
            default="now",
            dest='to_date',
        )
        parser.add_argument(
            '-x',
            help="Z dimension to plot on the x axis",
            type=int,
            default=-1,
        )
        parser.add_argument(
            '-y',
            help="Z dimension to plot on the y axis",
            type=int,
            default=-1,
        )
        parser.add_argument(
            '-o', '--output',
            help="Output figure to file",
            type=str,
            default=None,
        )

    def exec(self, args):
        loud = Loud(**self.config)
        _ = loud.get_model_latent_data(
            model_name=args.model_name,
            from_date=args.from_date,
            to_date=args.to_date,
        )


class TrainCommand(Command):
    """Train a model using data points in the given time range."""
    @property
    def short_name(self):
        return 'train-model'

    def add_args(self, parser):
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )
        parser.add_argument(
            '-f', '--from',
            help="From date",
            type=str,
            dest='from_date',
        )
        parser.add_argument(
            '-t', '--to',
            help="To date",
            type=str,
            default="now",
            dest='to_date',
        )
        parser.add_argument(
            '--bg',
            help="Run this request in the background",
            default=False,
            action='store_true',
        )
        parser.add_argument(
            '-m', '--max-evals',
            help="Maximum number of training iterations",
            type=int,
        )
        parser.add_argument(
            '-e', '--epochs',
            help="Limit the number of epochs used for training",
            default=100,
            type=int,
        )
        parser.add_argument(
            '-c', '--continue',
            help="Resume training from the current version",
            action='store_true',
            dest='resume_training',
        )

    def exec_all(self, args):
        loud = Loud(**self.config)
        for model in loud.model_generator(
            fields=['settings'],
            include_fields=True,
        ):
            args.model_name = model['settings']['name']
            self.exec(args)

    def exec(self, args):
        if args.model_name == '*':
            return self.exec_all(args)
        if not args.from_date:
            raise LoudMLException(
                "'from' argument is required",
            )
        if not args.to_date:
            raise LoudMLException(
                "'to' argument is required",
            )
        loud = Loud(**self.config)
        job = loud.train_model(
            model_name=args.model_name,
            from_date=args.from_date,
            to_date=args.to_date,
            max_evals=args.max_evals,
            epochs=args.epochs,
            resume=args.resume_training,
        )
        if not job:
            print("Failed to start job")
            exit(1)
        if args.bg:
            print(job.id)
        else:
            poll_job(job)


class ForecastCommand(Command):
    """Get forecast data points using a trained model."""
    @property
    def short_name(self):
        return 'forecast-model'

    def add_args(self, parser):
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )
        parser.add_argument(
            '-i', '--input',
            dest='input',
            help="name of the bucket to read input data",
        )
        parser.add_argument(
            '-o', '--output',
            dest='output',
            help="name of the bucket where prediction will be saved",
        )
        parser.add_argument(
            '-c', '--constraint',
            help="Test constraint, using format: feature:low|high:threshold",
            type=str,
        )
        parser.add_argument(
            '-f', '--from',
            help="From date",
            type=str,
            dest='from_date',
        )
        parser.add_argument(
            '-t', '--to',
            help="To date",
            type=str,
            dest='to_date',
        )
        parser.add_argument(
            '--bg',
            help="Run this request in the background",
            default=False,
            action='store_true',
        )
        parser.add_argument(
            '-p',
            help="percentage of confidence interval",
            type=float,
            default=0.68,  # = +/-1 STD
            dest='p_val',
        )
        parser.add_argument(
            '-s', '--save',
            action='store_true',
            help="Save predictions to the output bucket",
        )

    def exec(self, args):
        if not args.from_date:
            raise LoudMLException(
                "'from' argument is required")
        if not args.to_date:
            raise LoudMLException(
                "'to' argument is required")

        constraint = parse_constraint(args.constraint) if args.constraint \
            else None

        loud = Loud(**self.config)
        job = loud.forecast_model(
            model_name=args.model_name,
            input_bucket=args.input,
            output_bucket=args.output if args.save else None,
            save_output_data=args.save,
            from_date=args.from_date,
            to_date=args.to_date,
            constraint=constraint,
            p_val=args.p_val,
        )
        if args.bg:
            print(job.id)
        else:
            poll_job(job)
            if job.success() and not self.quiet:
                out = job.fetch_result()
                for line in format_buckets(out):
                    print(line)


class PredictCommand(Command):
    """Get output data points from a trained model."""
    @property
    def short_name(self):
        return 'predict-model'

    def add_args(self, parser):
        parser.add_argument(
            'model_name',
            help="Model name",
            type=str,
        )
        parser.add_argument(
            '-i', '--input',
            dest='input',
            help="name of the bucket to read input data",
        )
        parser.add_argument(
            '-o', '--output',
            dest='output',
            help="name of the bucket where prediction will be saved",
        )
        parser.add_argument(
            '-f', '--from',
            help="From date",
            type=str,
            dest='from_date',
        )
        parser.add_argument(
            '-t', '--to',
            help="To date",
            type=str,
            dest='to_date',
        )
        parser.add_argument(
            '--bg',
            help="Run this request in the background",
            default=False,
            action='store_true',
        )
        parser.add_argument(
            '-a', '--anomalies',
            help="Detect anomalies",
            action='store_true',
        )
        parser.add_argument(
            '-s', '--save',
            action='store_true',
            help="Save predictions to the output bucket",
        )

    def exec(self, args):
        if not args.from_date:
            raise LoudMLException(
                "'from' argument is required")
        if not args.to_date:
            raise LoudMLException(
                "'to' argument is required")

        loud = Loud(**self.config)
        job = loud.predict_model(
            model_name=args.model_name,
            input_bucket=args.input,
            output_bucket=args.output if args.save else None,
            from_date=args.from_date,
            to_date=args.to_date,
            save_output_data=args.save,
            flag_abnormal_data=args.anomalies,
        )
        if args.bg:
            print(job.id)
        else:
            poll_job(job)
            if job.success() and not self.quiet:
                out = job.fetch_result()
                for line in format_buckets(out):
                    print(line)


class ShowVersionCommand(Command):
    """Print Loud ML model server version."""
    @property
    def short_name(self):
        return 'version'

    def exec(self, args):
        loud = Loud(**self.config)
        print(loud.version())


class HelpCommand(Command):
    """Show this help message."""
    @property
    def short_name(self):
        return 'help'

    def exec(self, args):
        global g_commands
        for command in g_commands:
            cmd = command()
            print('   {:<22}: {}'.format(
                cmd.short_name, command.description()))
        print()


class QuitCommand(Command):
    """Quit this shell."""
    @property
    def short_name(self):
        return 'quit'

    def exec(self, args):
        raise SystemExit()


g_commands = [
    CreateBucketCommand,
    DeleteBucketCommand,
    ListBucketsCommand,
    ShowBucketCommand,
    ReadBucketCommand,
    WriteBucketCommand,
    ClearBucketCommand,
    ListVersionsCommand,
    LoadVersionCommand,
    CreateModelCommand,
    DeleteModelCommand,
    ListModelsCommand,
    ListTemplatesCommand,
    CreateTemplateCommand,
    DeleteTemplateCommand,
    ShowModelCommand,
    TrainCommand,
    PredictCommand,
    ForecastCommand,
    PlotCommand,
    ShowVersionCommand,
    HelpCommand,
    QuitCommand,
]


def cmd_gen(args):
    if args.version:
        yield 'version'
    elif args.execute:
        yield args.execute
    else:
        res = parse_addr(args.addr, default_port=8077)
        config = {
            'loudml_host': res['host'],
            'loudml_port': res['port'],
        }
        loud = Loud(**config)
        print('Connected to {} version {}'.format(
            args.addr, loud.version()))
        print('Loud ML shell')
        while True:
            r = input('> ').strip()
            if len(r):
                yield r


def main(argv=None):
    """
    The Python client interface to Loud ML model servers.
    """
    global g_commands

    epilog = """

Examples:

    # Use loudml in a non-interactive mode to output the two days forecast
    # for model "test-model" and pretty print the output:
    $ loudml --execute 'forecast-model -f now -t now+2d test-model'

    # Connect to a specific Loud ML model server:
    $ loudml --addr hostname:8077

"""

    parser = argparse.ArgumentParser(
        description=main.__doc__,
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '-A', '--addr',
        help="Loud ML model server host and port to connect to.",
        type=str,
        default="localhost:8077",
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help="Quiet: no stdout",
    )
    parser.add_argument(
        '--version',
        action='store_true',
        help="Display the version and exit.",
    )
    parser.add_argument(
        '-e', '--execute',
        type=str,
        help="Execute command and quit.",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    args = parser.parse_args(argv)

    shell_parser = argparse.ArgumentParser(
        add_help=False, usage=argparse.SUPPRESS)
    subparsers = shell_parser.add_subparsers(
        title="Commands",
        dest="command",
    )
    for cmd in g_commands:
        command = cmd()
        subparser = subparsers.add_parser(
            command.short_name, add_help=False)
        command.add_args(subparser)
        subparser.set_defaults(set_config=command.set_config)
        subparser.set_defaults(exec=command.exec)

    try:
        for user_input in cmd_gen(args):
            exit_code = 1
            try:
                shell_args = shell_parser.parse_args(
                    shlex.split(user_input))
                shell_args.set_config(
                    addr=args.addr, quiet=args.quiet)
            except SystemExit:  # Unknown commands. Raised by parse_args()
                if user_input == 'quit':
                    sys.exit(0)
                continue

            try:
                shell_args.exec(shell_args)
                exit_code = 0
            except requests.exceptions.ConnectionError:
                logging.error("%s: connect: connection refused", args.addr)
                logging.error("Please check your connection settings and ensure 'loudmld' is running.")  # noqa
                sys.exit(2)
            except requests.exceptions.HTTPError as exn:
                logging.error(str(exn))
            except requests.exceptions.Timeout:
                logging.error("Request timed out")
            except requests.exceptions.TooManyRedirects:
                logging.error("Too many redirects")
            except requests.exceptions.RequestException as exn:
                logging.error(str(exn))

        return exit_code  # All input is consumed
    except LoudMLException as exn:
        logging.error(exn)
    except KeyboardInterrupt:
        logging.error("operation aborted")

    return 1
