import numpy as np
import pandas as pd
from scipy import stats


class MatchInfo:
    def __init__(self, path, ident, slope, offset, s):
        self.path = path
        self.ident = ident
        self.bounds = (s[0], s[-1])
        self.slope = slope
        self.offset = offset

    def __str__(self):
        return f"{self.path} - {self.ident} - {self.offset}"


def parse_data(filename):
    return pd.read_csv(filename, sep=';')


def calculate_segments(initial_dataframe, identifier):
    # skip first element, panako makes this unreliable
    initial_dataframe = initial_dataframe[1:]

    flatten = lambda l: [item for sublist in l for item in sublist]

    df = initial_dataframe[initial_dataframe[' Match Identifier'] == identifier][[' Match start (s)']]
    df.columns = ['match_start']

    if len(df) < 20:
        return None

    # Extract the general slope
    a_bounds = (0.8, 1.2)
    n_bins = 10000
    bins = np.linspace(*a_bounds, n_bins)
    hist = pd.Series(np.zeros(n_bins), index=bins)
    series = df['match_start']
    for i in series.index:
        for j in range(2, 300):
            if i + j not in series.index:
                continue
            a_ij = (series[i + j] - series[i])/float(j)
            if a_bounds[0] < a_ij < a_bounds[1]:
                hist[bins[np.digitize(a_ij, bins)]] += 1

    hist[np.abs(hist.index - 1) < 0.0005] = 0
    a = hist.idxmax()
    if abs(a - 1) < 0.005:
        a = 1

    # Extract offset
    med = (df['match_start'] - a * df.index).median()
    estimate = stats.gaussian_kde(df['match_start'] - a * df.index)
    rnge = np.linspace(med - 750, med + 750, 50000)
    ests = estimate(rnge)
    amx = ests.argmax()
    mx = ests.max()

    al = np.argmax(ests[:amx] >= mx / 2)
    ar = 25000 + np.argmax(ests[amx:] <= mx / 2)
    bounds = (rnge[np.clip(al, 0, 50000 - 1)], rnge[np.clip(ar, 0, 50000 - 1)])

    points = (df['match_start'] - a * df.index)
    lb, rb = bounds
    points_in_bounds = points[(points > lb) & (points < rb)]

    segs = [[]]
    for i, x in enumerate(points_in_bounds.index):
        if i + 1 >= len(points_in_bounds.index):
            break

        idx = points_in_bounds.index
        di = idx[i + 1] - idx[i]

        if di > 10:
            segs.append([])

        segs[-1].append(idx[i])

    segs = sorted(segs, key=len)
    segs = [s for s in segs if len(s) > 20]
    segs = [s for s in segs if s[-1] - s[0] >= 5]

    # TODO: Remove this hardcoded stuff
    path = "src_tracks/" + initial_dataframe[initial_dataframe[' Match Identifier'] ==
                                             identifier].iloc[0]['Match description']

    offset = (amx/50000)*1500 - 750 + med
    if len(segs) > 0:
        return MatchInfo(path, identifier, a, offset, flatten(sorted(segs, key=lambda x: x[0])))
    else:
        return None


class SegmentFinder:
    provides = ['segments']
    needs = ['fingerprints']

    def process(self, state):
        df = state['fingerprints']

        identifiers = df[' Match Identifier'].unique()
        matches = [calculate_segments(df, ident) for ident in identifiers]
        matches = [m for m in matches if m]

        return {**state, 'segments': matches}
