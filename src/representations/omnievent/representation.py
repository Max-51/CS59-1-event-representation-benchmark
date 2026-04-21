from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation


@register_representation("omnievent")
class OmniEventRepresentation(BaseRepresentation):
    def __init__(self, config):
        super().__init__(config)

    def build(self, events):
        raise NotImplementedError("OmniEvent not implemented yet")
