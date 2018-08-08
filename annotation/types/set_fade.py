from annotation.types.event import AnnotationEvent


class SetFade(AnnotationEvent):
    event_type = 'set_value'

    def __init__(self, time, track, value):
        super().__init__(time)
        self.track = track
        self.value = value
