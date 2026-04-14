from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation

@register_representation("matrix_lstm")
class MatrixLSTMRepresentation(BaseRepresentation):
    def build(self, events):
        print("Building Matrix-LSTM representation")
        return None