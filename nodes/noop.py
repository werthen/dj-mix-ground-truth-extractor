class Noop:
    provides = []
    needs = []

    def process(self, state):
        return state
