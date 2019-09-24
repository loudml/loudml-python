from loudml.api import (
    Service,
)


class ScheduledJobService(Service):
    def __init__(
        self,
    ):
        super().__init__(prefix='/scheduled_jobs')
