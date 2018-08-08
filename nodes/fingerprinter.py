import subprocess
import logging
import re
import pandas as pd
from io import StringIO
import os
from .segment_finder import parse_data


class Fingerprinter:
    needs = ['mix_path']
    provides = ['fingerprints']

    def __init__(self, lazy=True):
        self.lazy = lazy

    def process(self, state):
        mix_path = state['mix_path']

        cache_filename = mix_path + '.fpcache'
        if self.lazy:
            logging.info('looking for file ' + cache_filename)

            if os.path.isfile(cache_filename):
                logging.info('found file ' + cache_filename)
                return {
                    **state,
                    'fingerprints': parse_data(cache_filename)
                }

        logging.info('Storing tracks into Panako database')
        subprocess.run(
            [f"panako store {state['src_track_paths']}/*.wav"],
            shell=True
        )

        completed = subprocess.run(
            ['panako', 'monitor', state['mix_path']],
            stdout=subprocess.PIPE,
            encoding='utf-8'
        )

        fingerprints = pd.read_csv(
            StringIO(re.sub(',', '.', completed.stdout)),
            sep=';'
        )

        if self.lazy:
            with open(cache_filename, 'w') as f:
                f.write(re.sub(',', '.', completed.stdout))

        return {
            **state,
            'fingerprints': fingerprints
        }
