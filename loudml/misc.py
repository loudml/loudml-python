"""
Miscelaneous Loud ML helpers
"""
from itertools import chain
from loudml import (
    errors,
)
import datetime
import dateutil.parser


def parse_addr(addr, default_port=None):
    addr = addr.split(':')
    return {
        'host': 'localhost' if len(addr[0]) == 0 else addr[0],
        'port': default_port if len(addr) == 1 else int(addr[1]),
    }


def parse_constraint(constraint):
    try:
        feature, _type, threshold = constraint.split(':')
    except ValueError:
        raise errors.Invalid("invalid format for 'constraint' parameter")

    if _type not in ('low', 'high'):
        raise errors.Invalid(
            "invalid threshold type for 'constraint' parameter")

    try:
        threshold = float(threshold)
    except ValueError:
        raise errors.Invalid("invalid threshold for 'constraint' parameter")

    return {
        'feature': feature,
        'type': _type,
        'threshold': threshold,
    }


def parse_timedelta(
    delta,
    min=None,
    max=None,
    min_included=True,
    max_included=True,
):
    """
    Parse time delta
    """

    if isinstance(delta, str) and len(delta) > 0:
        unit = delta[-1]

        if unit in '0123456789':
            unit = 's'
            value = delta
        else:
            value = delta[:-1]
    else:
        unit = 's'
        value = delta

    try:
        value = float(value)
    except ValueError:
        raise errors.Invalid("invalid time delta value")

    if unit == 'M':
        value *= 30
        unit = 'd'
    elif unit == 'y':
        value *= 365
        unit = 'd'

    unit = {
        's': 'seconds',
        'm': 'minutes',
        'h': 'hours',
        'd': 'days',
        'w': 'weeks',
    }.get(unit)

    if unit is None:
        raise errors.Invalid("invalid time delta unit")

    message = "time delta must be {} {} seconds"

    if min is not None:
        if min_included:
            if value < min:
                raise errors.Invalid(message.format(">=", min))
        else:
            if value <= min:
                raise errors.Invalid(message.format(">", min))

    if max is not None:
        if max_included:
            if value > max:
                raise errors.Invalid(message.format("<=", max))
        else:
            if value >= max:
                raise errors.Invalid(message.format("<", max))

    return datetime.timedelta(**{unit: value})


def ts_to_datetime(ts):
    """
    Convert timestamp to datetime
    """
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)


def ts_to_str(ts):
    """
    Convert timestamp to string
    """
    return datetime_to_str(ts_to_datetime(ts))


def str_to_datetime(string):
    """
    Convert string (ISO or relative) to timestamp
    """
    if string.startswith("now"):
        now = datetime.datetime.now()
        if len(string) == 3:
            return now
        return now + parse_timedelta(string[3:])
    else:
        return dateutil.parser.parse(string)


def str_to_ts(string):
    """
    Convert string to timestamp
    """
    return str_to_datetime(string).timestamp()


def datetime_to_str(dt):
    """
    Convert datetime to string
    """
    return "%s.%03dZ" % (
        dt.strftime("%Y-%m-%dT%H:%M:%S"), dt.microsecond / 1000)


def make_datetime(mixed):
    """
    Build a datetime object from a mixed input (second timestamp or string)
    """

    try:
        return ts_to_datetime(float(mixed))
    except ValueError as exn:
        if isinstance(mixed, str):
            return str_to_datetime(mixed)
        else:
            raise exn


def make_ts(mixed):
    """
    Build a timestamp from a mixed input
        (second timestamp or ISO string or relative time)
    """

    try:
        return float(mixed)
    except ValueError:
        return str_to_ts(mixed)


def _format_float(s):
    try:
        _ = float(s)
        return '{:.3f}'.format(s).rstrip('0')
    except ValueError:
        return str(s)


def _format_observation(data, i, feature):
    if not data['observed'][feature][i]:  # None
        return 'None'
    return '{:.3f}'.format(
        data['observed'][feature][i]).rstrip('0')


def _format_prediction(data, i, feature):
    if data.get('stats'):
        val = '{:.3f}'.format(
            data['predicted'][feature][i]).rstrip('0')
        score = float(data['stats'][i]['score'])
        if bool(data['stats'][i]['anomaly']):
            flag = '*'
        else:
            flag = ''
        return '{} [{} {:2.1f}]'.format(val, flag, score)
    else:
        return '{:.3f}'.format(
            data['predicted'][feature][i]).rstrip('0')


def format_buckets(data):
    features = sorted(data['observed'].keys())
    features_first_row = [
        '@{}'.format(feature)
        for feature in features
    ]
    loudml = [
        'loudml.{}'.format(feature)
        for feature in features
    ] if data.get('predicted') else []
    first_row = list(chain(
        ['timestamp'], features_first_row, loudml))

    rows = [first_row]
    for i, timestamp in enumerate(data['timestamps']):
        actual = [
            _format_observation(
                data, i, feature)
            for feature in features
        ]
        loudml = [
            _format_prediction(data, i, feature)
            for feature in features
        ] if data.get('predicted') else []
        row = list(chain(
            [str(timestamp)], actual, loudml))
        rows.append(row)

    col_width = max(
        len(word) for row in rows for word in row) + 2  # padding
    for row in rows:
        yield "".join(
            word.ljust(col_width) for word in row)


def format_points(points):
    features = sorted(
        set([
            field for point in points for field in point.keys()
        ]) - set(['timestamp', 'tags']))

    first_row = list(chain(
        ['timestamp'], features, ['tags']))
    rows = [first_row]
    for point in points:
        fields = [
            _format_float(point.get(feature))
            for feature in features
        ]
        tags = ",".join([
            '{}={}'.format(tag, val)
            for tag, val in point.get('tags', {}).items()
        ])
        row = list(chain(
            [str(point['timestamp'])], fields, [tags]))
        rows.append(row)

    col_width = max(
        len(word) for row in rows for word in row) + 2  # padding
    for row in rows:
        yield "".join(
            word.ljust(col_width) for word in row)


def format_model_versions(versions):
    features = sorted(
        set([
            key for ver in versions for key in ver['version'].keys()
        ]) - set(['name']))
    states = sorted(
        set([
            key for ver in versions for key in ver['state'].keys()
        ]))

    first_row = list(chain(
        ['version'], features, states))
    rows = [first_row]
    for ver in versions:
        fields = [
            _format_float(ver['version'].get(feature))
            for feature in features
        ]
        extras = [
            _format_float(ver['state'].get(state))
            for state in states
        ]

        row = list(chain(
            [ver['version']['name']], fields, extras))
        rows.append(row)

    col_width = max(
        len(word) for row in rows for word in row) + 2  # padding
    for row in rows:
        yield "".join(
            word.ljust(col_width) for word in row)


def format_jobs(jobs):
    features = [
        'id', 'name', 'state', 'x/N', 'time_left', 'duration']
    first_row = features
    rows = [first_row]
    for job in jobs:
        job_name = '{}({})'.format(
            job.get('type'), job.get('model'))
        job_state = job.get('state')
        if 'progress' in job:
            eval_count = int(job['progress']['eval'])
            eval_total = int(job['progress']['max_evals'])
            if eval_count > eval_total:
                eval_count = eval_total
            job_progress = '{}/{}'.format(
                eval_count, eval_total)
        else:
            job_progress = ''
        if 'remaining_time' in job:
            time_left = _format_float(job['remaining_time'])
        else:
            time_left = ''
        row = [
            job['id'],
            job_name,
            job_state,
            job_progress,
            time_left,
            _format_float(job.get('duration', '')),
        ]
        rows.append(row)

    col_width = [
        max(len(row[i]) for row in rows) + 2
        for i, _ in enumerate(first_row)
    ]
    for row in rows:
        yield "".join(
            word.ljust(col_width[i]) for i, word in enumerate(row))
