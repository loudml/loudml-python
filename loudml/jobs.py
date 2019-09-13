from loudml.api import (
    Service,
)


class JobService(Service):
    def __init__(
        self,
    ):
        super().__init__(prefix='/jobs')

    def cancel_one(
        self,
        name,
    ):
        return self._do_one(name, '_cancel')

    def cancel_many(
        self,
        names,
    ):
        return self._do_many(names, '_cancel')

    def create(
        self,
        settings,
        extra_params,
    ):
        raise NotImplementedError('`create` not supported')

    def del_one(
        self,
        name,
    ):
        raise NotImplementedError('`del_one` not supported')

    def del_many(
        self,
        names,
    ):
        raise NotImplementedError('`del_many` not supported')
