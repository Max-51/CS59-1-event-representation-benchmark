from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation


@register_representation("get")
class GETRepresentation(BaseRepresentation):
    def __init__(self, config):
        super().__init__(config)

    def build(self, events):
        raise NotImplementedError("GET not implemented yet")
