from loudml.api import (
    Service,
)


class TemplateService(Service):
    def __init__(
        self,
        loud,
    ):
        super().__init__(prefix='/templates')
