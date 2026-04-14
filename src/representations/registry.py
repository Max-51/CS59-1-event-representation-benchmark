REPRESENTATION_REGISTRY = {}

def register_representation(name):
    def wrapper(cls):
        REPRESENTATION_REGISTRY[name] = cls
        return cls
    return wrapper
