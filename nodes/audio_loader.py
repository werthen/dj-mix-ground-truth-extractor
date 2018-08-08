import librosa


class AudioLoader:
    needs = []

    def __init__(self, audio_id, path):
        self.provides = [audio_id]
        self.path = path
        self.audio_id = audio_id

    def process(self, state):
        return {**state, self.audio_id: librosa.load(self.path)}
