from src.representations.registry import REPRESENTATION_REGISTRY

from src.representations.est import ESTRepresentation
from src.representations.ergo import ERGORepresentation
from src.representations.get import GETRepresentation
from src.representations.matrix_lstm import MatrixLSTMRepresentation
from src.representations.evrepsl import EvRepSLRepresentation
from src.representations.omnievent import OmniEventRepresentation
from src.representations.event_pretraining import EventPretrainingRepresentation

import numpy as np

print("Registered methods:")
print(REPRESENTATION_REGISTRY)

# Minimal fake events: Nx4 -> (x, y, t, p)
events = np.array([
    [10, 20, 0.001, 1],
    [15, 25, 0.002, -1],
    [30, 40, 0.003, 1],
], dtype=np.float32)

for name, cls in REPRESENTATION_REGISTRY.items():
    print(f"\nTesting method: {name}")

    config = {}

    # EvRepSL needs extra config
    if name == "evrepsl":
        config = {
            "width": 320,
            "height": 240,
            "device": "cpu",
            "return_numpy": True,
        }

    rep = cls(config)

    try:
        output = rep.build(events=events)
        print(f"{name} build succeeded")
        if output is not None:
            print(f"Output type: {type(output)}")
    except NotImplementedError as e:
        print(f"{name} placeholder OK: {e}")
    except Exception as e:
        print(f"{name} failed: {type(e).__name__}: {e}")