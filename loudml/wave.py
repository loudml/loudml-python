import argparse
import logging
import random
import time
import requests
import sys

from loudml.randevents import (
    CamelEventGenerator,
    LoudMLEventGenerator,
    FlatEventGenerator,
    SawEventGenerator,
    SinEventGenerator,
    TriangleEventGenerator,
)
from loudml.errors import (
   LoudMLException,
)
from loudml.client import (
    Loud,
)
from loudml.misc import (
    make_datetime,
)


def generate_data(
    ts_generator,
    from_date,
    to_date,
    step_ms,
    errors,
    burst_ms,
    field,
):
    ano = False
    previous_ts = None
    for ts in ts_generator.generate_ts(
        from_date,
        to_date,
        step_ms=step_ms,
    ):
        if not ano and errors > 0:
            val = random.random()
            if val < errors:
                ano = True
                total_burst_ms = 0
                previous_ts = ts

        if ano and total_burst_ms < burst_ms:
            total_burst_ms += (ts - previous_ts) * 1000.0
            previous_ts = ts
        else:
            ano = False
            yield ts, {
                field: random.lognormvariate(10, 1),
            }


def build_tag_dict(tags=None):
    tag_dict = {}
    if tags:
        for tag in tags.split(','):
            k, v = tag.split(':')
            tag_dict[k] = v
    return tag_dict


def dump_to_bucket(
    loud,
    generator,
    bucket_name,
    tags=None,
    verbose=False,
    batch_size=10000,
):
    points = []
    for ts, point in generator:
        now = time.time()
        if ts > now:
            time.sleep(ts - now)

        point['timestamp'] = str(ts)
        if tags:
            point['tags'] = tags
        points.append(point)
        if len(points) >= batch_size:
            loud.write_points(
                bucket_name, points, verbose)
            points.clear()

    if len(points):
        loud.write_points(bucket_name, points, verbose)


def main():
    """
    Generate and write random data to a TSDB bucket
    """
    parser = argparse.ArgumentParser(
        description=main.__doc__,
    )
    parser.add_argument(
        'bucket_name',
        help="Output bucket to write to.",
        type=str,
    )
    parser.add_argument(
        '-A', '--addr',
        help="Loud ML model server host and port to connect to.",
        type=str,
        default="localhost:8077",
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Verbose: dump data written to the TSDB",
    )

    parser.add_argument(
        '--field',
        help="Use field name (default=value) in generated data.",
        type=str,
        default="value",
    )
    parser.add_argument(
        '-f', '--from',
        help="From date",
        type=str,
        default="now-7d",
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
        '--shape',
        help="Configure the data shape of count(field) aggregation.",
        choices=[
            'flat',
            'saw',
            'sin',
            'triangle',
            'camel',
            'loudml',
        ],
        default='sin',
    )
    parser.add_argument(
        '--amplitude',
        help="Peak amplitude for periodic shapes",
        type=float,
        default=1,
    )
    parser.add_argument(
        '--base',
        help="Base value for number of events",
        type=float,
        default=1,
    )
    parser.add_argument(
        '--trend',
        help="Trend (event increase per hour)",
        type=float,
        default=0,
    )
    parser.add_argument(
        '--period',
        help="Pattern period in seconds.",
        type=float,
        default=24 * 3600,
    )
    parser.add_argument(
        '--sigma',
        help="Noise sigma added to generated data.",
        type=float,
        default=2,
    )
    parser.add_argument(
        '--step-ms',
        help="Milliseconds elapsed in each step for generating samples",
        type=int,
        default=60000,
    )
    parser.add_argument(
        '-e', '--errors',
        help="Output anomalies with the given error rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        '-b', '--burst-ms',
        help="Burst duration, for anomalies",
        type=int,
        default=0,
    )
    parser.add_argument(
        '--clear',
        help="Clear bucket before inserting new data "
             "(risk of data loss! Use with caution!)",
        action='store_true',
    )
    parser.add_argument(
        '--tags',
        help="List of comma separated key:value tag pairs.",
        type=str,
    )

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    loud = Loud(
        hosts=[args.addr],
    )

    if args.shape == 'flat':
        ts_generator = FlatEventGenerator(base=args.base, trend=args.trend)
    elif args.shape == 'loudml':
        ts_generator = LoudMLEventGenerator(base=args.base, trend=args.trend)
    elif args.shape == 'camel':
        ts_generator = CamelEventGenerator(
            base=args.base,
            amplitude=args.amplitude,
            period=args.period,
            trend=args.trend,
            sigma=args.sigma,
        )
    elif args.shape == 'saw':
        ts_generator = SawEventGenerator(
            base=args.base,
            amplitude=args.amplitude,
            period=args.period,
            trend=args.trend,
            sigma=args.sigma,
        )
    elif args.shape == 'triangle':
        ts_generator = TriangleEventGenerator(
            base=args.base,
            amplitude=args.amplitude,
            period=args.period,
            trend=args.trend,
            sigma=args.sigma,
        )
    else:
        ts_generator = SinEventGenerator(
            base=args.base,
            amplitude=args.amplitude,
            trend=args.trend,
            period=args.period,
            sigma=args.sigma,
        )

    from_date = make_datetime(args.from_date)
    to_date = make_datetime(args.to_date)

    logging.info("generating data from %s to %s", from_date, to_date)

    generator = generate_data(
        ts_generator,
        from_date.timestamp(),
        to_date.timestamp(),
        args.step_ms,
        args.errors,
        args.burst_ms,
        args.field,
    )

    try:
        if args.verbose:
            print('Connected to {} version {}'.format(
                args.addr, loud.version()))

        if args.clear:
            loud.buckets.clear(args.bucket_name)

        dump_to_bucket(
            loud,
            generator,
            args.bucket_name,
            tags=build_tag_dict(args.tags),
            verbose=args.verbose,
        )
        return 0
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
    except LoudMLException as exn:
        logging.error(exn)
    except KeyboardInterrupt:
        logging.error("operation aborted")

    return 1
