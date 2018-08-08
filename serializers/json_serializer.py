def obj_segment(s):
    return {
        'path': s.path,
        'bounds': list(int(x) for x in s.bounds),
        'stretch': s.slope,
        'offset': s.offset,
    }


class JSONSerializer(object):

    def output(self, result):
        return {
            'segments': [obj_segment(s) for s in result['segments']],
            'xfades': [
                {
                    'mix_bounds': list(int(x) for x in x['mix']),
                    'src_tracks': [obj_segment(s) for s in x['src_tracks']],
                    'faders': [list(fads[0]), list(fads[1])],
                }
                for x, fads in zip(result['xfades'], result['faders'])
            ],
        }
