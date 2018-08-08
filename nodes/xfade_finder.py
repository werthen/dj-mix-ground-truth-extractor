import librosa
import numpy as np


def get_crossfades(matches):
    pairs = set()
    for m in matches:
        others = [mm for mm in matches if mm != m]
        close_ind = np.argmin(
            np.abs(np.array([e.bounds[1] for e in others]) - m.bounds[0])
        )
        other_track = others[close_ind]

        pair = None
        if m.ident < other_track.ident:
            pair = (m, other_track)
        else:
            pair = (other_track, m)

        pairs.add(pair)

    xfades = []
    for p in pairs:
        x = abs(p[0].bounds[1] - p[1].bounds[0])
        y = abs(p[1].bounds[1] - p[0].bounds[0])
        if x < y:
            fst = p[0]
            snd = p[1]
        else:
            fst = p[1]
            snd = p[0]

        begin, end = max(0, snd.bounds[0] - 30), snd.bounds[0] + 30

        xfades.append({
            'mix': (begin, end),
            'src_tracks': (fst, snd),
            'src_times': (begin + fst.offset, begin + snd.offset)
        })

    return xfades


def generate_xfade(m, mx, msr):
    print(f"Loading {m.path}")
    x, sr = librosa.load(m.path)
    xx = librosa.effects.time_stretch(x, 1 / m.slope)
    begin = int(-m.offset * msr)
    librosa.output.write_wav(f"align_test-{m.ident}.wav", np.array((mx[begin:begin + 60 * msr], xx[:60 * sr])), sr)
    print(f"align_test-{m.ident}.wav")


def generate_xfades(matches, mx, msr):
    for m in matches:
        generate_xfades(m, mx, msr)


class XFadeFinder:
    provides = ['xfades']
    needs = ['segments']

    def process(self, state):
        return {**state, 'xfades': get_crossfades(state['segments'])}
