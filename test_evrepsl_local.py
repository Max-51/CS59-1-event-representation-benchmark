import numpy as np
from src.representations.evrepsl.representation import EvRepSLRepresentation

# Create a tiny fake event stream: [x, y, t, p]
events = np.array([
    [10, 20, 0.01, 1],
    [11, 21, 0.02, 0],
    [12, 22, 0.03, 1],
    [13, 23, 0.04, 0],
    [14, 24, 0.05, 1],
], dtype=np.float32)

config = {
    "width": 320,
    "height": 240,
    "device": "cpu",          # Use cpu first for safer testing
    "return_numpy": True
}

rep = EvRepSLRepresentation(config)
output = rep.build(events)

print("Type of output:", type(output))
print("Shape of output:", output.shape)
print("Dtype of output:", output.dtype)
print("Min value:", output.min())
print("Max value:", output.max())