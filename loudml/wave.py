import argparse
import logging
import random
import time

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
from loudml.api import (
    Loud,
)
from loudml.misc import (
    make_datetime,
    parse_addr,
    format_points,
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
    **kwargs
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
        if len(points) >= 1000:
            logging.info(
                "writing %d points", len(points))
            if verbose:
                for line in format_points(points):
                    print(line)
            loud.write_bucket(
                bucket_name=bucket_name,
                points=points,
                **kwargs
            )
            points.clear()

    if len(points):
        loud.write_bucket(
            bucket_name=bucket_name,
            points=points,
            **kwargs
        )
        if verbose:
            for line in format_points(points):
                print(line)
        points.clear()


def main():
    """
    Generate random data and write to TSDB
    """

    parser = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'bucket_name',
        help="Output bucket name",
        type=str,
    )
    parser.add_argument(
        '-A', '--addr',
        help="Loud ML remote server address",
        type=str,
        default="127.0.0.1:8077",
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Verbose: dump data written to the TSDB",
    )

    parser.add_argument(
        '-m', '--measurement',
        help="Measurement",
        type=str,
        default='wave',
    )
    parser.add_argument(
        '--doc-type',
        help="Document type",
        type=str,
    )
    parser.add_argument(
        '--field',
        help="Field",
        type=str,
        default="value",
    )
    parser.add_argument(
        '--from',
        help="From date",
        type=str,
        default="now-7d",
        dest='from_date',
    )
    parser.add_argument(
        '--to',
        help="To date",
        type=str,
        default="now",
        dest='to_date',
    )
    parser.add_argument(
        '--shape',
        help="Data shape",
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
        help="Period in seconds",
        type=float,
        default=24 * 3600,
    )
    parser.add_argument(
        '--sigma',
        help="Sigma",
        type=float,
        default=2,
    )
    parser.add_argument(
        '--step-ms',
        help="Milliseconds elapsed in each step fo generating samples",
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
        help="Tags",
        type=str,
    )

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    addr = parse_addr(args.addr, default_port=8077)
    loud = Loud(
        loudml_host=addr['host'],
        loudml_port=addr['port'],
    )

    tags = build_tag_dict(args.tags)

    try:
        if args.clear:
            loud.clear_bucket(args.bucket_name)
    except LoudMLException as exn:
        logging.error(exn)
        return 1

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

    kwargs = {}
    if args.measurement:
        kwargs['measurement'] = args.measurement
    if args.doc_type:
        kwargs['doc_type'] = args.doc_type

    try:
        dump_to_bucket(
            loud,
            generator,
            args.bucket_name,
            tags=tags,
            verbose=args.verbose,
            **kwargs
        )
    except LoudMLException as exn:
        logging.error(exn)
