class BaseRepresentation:
    def __init__(self, config):
        self.config = config

    def build(self, events):
        """
        events: Nx4 (x, y, t, p)
        return: representation tensor / structure
        """
        raise NotImplementedError
