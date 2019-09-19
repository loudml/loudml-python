"""
Miscelaneous Loud ML helpers
"""
from itertools import chain
from loudml import (
    errors,
)


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


def _format_float(s):
    try:
        _ = float(s)
        return '{:.3f}'.format(s).rstrip('0')
    except ValueError:
        return str(s)


def _format_observation(data, i, feature):
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
    loudml = [
        'loudml.{}'.format(feature)
        for feature in features
    ] if data.get('predicted') else []
    first_row = list(chain(
        ['timestamp'], features, loudml))

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
            point.keys() for point in points
        ]) - set(['timestamp', 'tags']))

    first_row = list(chain(
        ['timestamp'], features, ['tags']))
    rows = [first_row]
    for point in points:
        fields = [
            point.get(feature)
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
