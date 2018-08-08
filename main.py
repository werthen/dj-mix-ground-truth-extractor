import logging
import json

import argparse

from serializers import JSONSerializer


def default_annotator(mix_path, src_track_paths):
    from nodes import Unmixer, GenericProvider, SegmentFinder, XFadeFinder, Fingerprinter
    from annotator import Annotator
    return Annotator([
        GenericProvider(
            src_track_paths=src_track_paths,
            mix_path=mix_path
        ),
        Fingerprinter(lazy=False),
        Unmixer(),
        XFadeFinder(),
        SegmentFinder()
    ])


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Analyse a DJ mix and reverse engineer it')

    parser.add_argument(
        'mix_path',
        metavar='<mix_path>',
        action='store',
        nargs=1,
        help='Path to a mix which needs to be analysed')

    parser.add_argument(
        'source_tracks',
        metavar='<source_track_directory_path>',
        action='store',
        nargs=1,
        help='Path to all source tracks')

    parser.add_argument(
        'output_file',
        metavar='<output_file>',
        action='store',
        nargs=1,
        help='Path to the output JSON file')

    args = parser.parse_args()

    result = default_annotator(args.mix_path[0], args.source_tracks[0]).start()

    serializer = JSONSerializer()

    with open(args.output_file[0], "w") as f:
        f.write(json.dumps(serializer.output(result)))


if __name__ == '__main__':
    main()
