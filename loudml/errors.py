"""
Loud ML errors
"""


class LoudMLException(Exception):
    """Loud ML exception"""
    code = 500

    def __init__(self, msg=None):
        super().__init__(msg or self.__doc__)


class Invalid(LoudMLException):
    """Data is invalid"""
    code = 400

    def __init__(self, error, name=None, path=None, hint=None):
        self.error = error
        self.name = name
        self.path = path
        self.hint = hint

    def __str__(self):
        hint = "" if self.hint is None else " ({})".format(self.hint)

        if self.path is None or len(self.path) == 0:
            return "{} is invalid: {}{}".format(
                self.name or "data",
                self.error,
                hint,
            )
        else:
            path = '.'.join([str(key) for key in self.path])
            return "invalid field {}: {}{}".format(path, self.error, hint)


class NotFound(LoudMLException):
    """Not found"""
    code = 404


class MethodNotSupported(LoudMLException):
    """Not found"""
    code = 500
