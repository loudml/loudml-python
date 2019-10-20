import re


def test_version():
    """Test version string"""
    from loudml_py import __versionstr__
    version_parts = re.split('[.-]', __versionstr__)
    assert 3 <= len(version_parts)  # must have at least Major.minor.patch
    try:
        map(int, version_parts[:3])
    except ValueError:
        raise TypeError('Version Major.minor.patch must be 3 integers')
