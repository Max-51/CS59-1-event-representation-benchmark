from src.representations.registry import REPRESENTATION_REGISTRY

# 导入所有方法，触发注册
from src.representations.est.representation import ESTRepresentation
from src.representations.ergo.representation import ERGORepresentation
from src.representations.get.representation import GETRepresentation
from src.representations.matrix_lstm.representation import MatrixLSTMRepresentation
from src.representations.evrepsl.representation import EvRepSLRepresentation

print("Registered methods:")
print(REPRESENTATION_REGISTRY)

for name in REPRESENTATION_REGISTRY:
    rep_class = REPRESENTATION_REGISTRY[name]
    rep = rep_class(config={})
    rep.build(events=None)