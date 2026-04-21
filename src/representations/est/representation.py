from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation


@register_representation("est")
class ESTRepresentation(BaseRepresentation):
    def __init__(self, config):
        super().__init__(config)

    def build(self, events):
        raise NotImplementedError("EST not implemented yet")
