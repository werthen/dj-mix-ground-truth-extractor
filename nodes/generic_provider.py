class GenericProvider:
    needs = []

    def __init__(self, **kwargs):
        self.provides = kwargs.keys()
        self.dict = kwargs

    def process(self, state):
        return {**state, **self.dict}
