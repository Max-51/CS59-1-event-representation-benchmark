import argparse

from src.representations.registry import REPRESENTATION_REGISTRY

# 导入所有方法，触发注册
from src.representations.est.representation import ESTRepresentation
from src.representations.ergo.representation import ERGORepresentation
from src.representations.get.representation import GETRepresentation
from src.representations.matrix_lstm.representation import MatrixLSTMRepresentation
from src.representations.evrepsl.representation import EvRepSLRepresentation


def main():
    parser = argparse.ArgumentParser(description="Run event representation benchmark")
    parser.add_argument("--method", type=str, required=True, help="Representation method name")
    args = parser.parse_args()

    method_name = args.method.lower()

    if method_name not in REPRESENTATION_REGISTRY:
        print(f"Error: method '{method_name}' is not registered.")
        print("Available methods:", list(REPRESENTATION_REGISTRY.keys()))
        return

    rep_class = REPRESENTATION_REGISTRY[method_name]
    rep = rep_class(config={})

    print(f"Running benchmark for method: {method_name}")
    rep.build(events=None)


if __name__ == "__main__":
    main()