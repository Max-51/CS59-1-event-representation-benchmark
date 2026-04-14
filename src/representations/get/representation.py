from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation

@register_representation("get")
class GETRepresentation(BaseRepresentation):
    def build(self, events):
        print("Building GET representation")
        return None