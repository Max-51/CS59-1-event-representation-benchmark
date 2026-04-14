from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation

@register_representation("est")
class ESTRepresentation(BaseRepresentation):
    def build(self, events):
        print("Building EST representation")
        return None