from annotation.types.event import AnnotationEvent


class StartTrack(AnnotationEvent):
    event_type = 'start_track'

    def __init__(self, time, track):
        super().__init__(time)
        self.track = track
