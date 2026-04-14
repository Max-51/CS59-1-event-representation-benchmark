from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation

@register_representation("evrepsl")
class EvRepSLRepresentation(BaseRepresentation):
    def build(self, events):
        print("Building EvRepSL representation")
        return None