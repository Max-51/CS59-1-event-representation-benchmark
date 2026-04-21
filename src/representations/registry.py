REPRESENTATION_REGISTRY = {}

def register_representation(name):
    def wrapper(cls):
        REPRESENTATION_REGISTRY[name] = cls
        return cls
    return wrapper

def get_representation(name):
    if name not in REPRESENTATION_REGISTRY:
        raise ValueError(f"Unknown representation: {name}. Available: {list(REPRESENTATION_REGISTRY.keys())}")
    return REPRESENTATION_REGISTRY[name]
