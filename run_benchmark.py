import argparse
import numpy as np

from src.representations.registry import REPRESENTATION_REGISTRY

# Import all representations to trigger registration
from src.representations.est.representation import ESTRepresentation
from src.representations.ergo.representation import ERGORepresentation
from src.representations.get.representation import GETRepresentation
from src.representations.matrix_lstm.representation import MatrixLSTMRepresentation
from src.representations.evrepsl.representation import EvRepSLRepresentation


def generate_dummy_events(num_events=5000, width=320, height=240):
    """
    Generate a small dummy event stream for quick benchmark testing.

    Returns:
        numpy.ndarray of shape (N, 4)
        columns are [x, y, t, p]
    """
    event_xs = np.random.randint(0, width, size=num_events)
    event_ys = np.random.randint(0, height, size=num_events)
    event_timestamps = np.sort(np.random.rand(num_events).astype(np.float32))
    event_polarities = np.random.randint(0, 2, size=num_events)

    events = np.stack(
        [event_xs, event_ys, event_timestamps, event_polarities],
        axis=1
    ).astype(np.float32)

    return events


def main():
    parser = argparse.ArgumentParser(description="Run event representation benchmark")
    parser.add_argument("--method", type=str, required=True, help="Representation method name")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu or cuda")
    parser.add_argument("--width", type=int, default=320, help="Sensor width")
    parser.add_argument("--height", type=int, default=240, help="Sensor height")
    parser.add_argument("--num_events", type=int, default=5000, help="Number of dummy events")

    args = parser.parse_args()

    method_name = args.method.lower()

    if method_name not in REPRESENTATION_REGISTRY:
        print(f"Error: method '{method_name}' is not registered.")
        print("Available methods:", list(REPRESENTATION_REGISTRY.keys()))
        return

    rep_class = REPRESENTATION_REGISTRY[method_name]

    config = {
        "width": args.width,
        "height": args.height,
        "device": args.device,
        "return_numpy": True,
    }

    rep = rep_class(config=config)

    print(f"Running benchmark for method: {method_name}")

    events = generate_dummy_events(
        num_events=args.num_events,
        width=args.width,
        height=args.height
    )

    output = rep.build(events)

    print("Benchmark run finished.")
    print("Output type:", type(output))

    if hasattr(output, "shape"):
        print("Output shape:", output.shape)

    if isinstance(output, np.ndarray):
        print("Output dtype:", output.dtype)
        print("Output min:", output.min())
        print("Output max:", output.max())


if __name__ == "__main__":
    main()