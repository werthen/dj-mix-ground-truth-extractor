class AnnotationEvent:
    event_type = 'event'

    def __init__(self, time):
        self.time = time

    def to_json(self):
        return {
            **self.__dict__,
            'type': self.event_type
        }
