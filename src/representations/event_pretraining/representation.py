from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation


@register_representation("event_pretraining")
class EventPretrainingRepresentation(BaseRepresentation):
    def __init__(self, config):
        super().__init__(config)

    def build(self, events):
        raise NotImplementedError("Event Pretraining not implemented yet")
