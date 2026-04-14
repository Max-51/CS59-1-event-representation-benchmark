from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation

@register_representation("ergo")
class ERGORepresentation(BaseRepresentation):
    def build(self, events):
        print("Building ERGO representation")
        return None