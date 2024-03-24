import re
from typing import List, Optional, Union
from subprocess import check_call

import numpy as np
# import nmslib

FILE_PATTERN = re.compile('(?P<uri>[a-zA-Z][-a-zA-Z0-9+.]*)://.*/(?P<file>.*)$')


def move_files(source: str, destination: str):
    check_call(['gsutil', '-m', 'cp', '-r', source, destination])


class AnnBase:
    def __init__(self, mapping: Optional[List[str]] = None):
        self.mapping = mapping

    def build_index(self):
        raise NotImplementedError

    def search_vec_top_n(self, vector, n: int):
        raise NotImplementedError

    def _save_file(self, path: str):
        raise NotImplementedError

    def _load_file(self, path: str, **kwargs):
        raise NotImplementedError

    def load(self, path: str, **kwargs):
        match_result = FILE_PATTERN.match(path.strip())
        if match_result is None:
            self._load_file(path)
        else:
            uri = match_result.groupdict()['uri']
            file = match_result.groupdict()['file']
            if uri == 'gs':
                move_files(path+'*', '.')
                self._load_file(file, **kwargs)

    def save(self, path: str):
        match_result = FILE_PATTERN.match(path.strip())
        if match_result is None:
            self._save_file(path)
        else:
            uri = match_result.groupdict()['uri']
            file = match_result.groupdict()['file']
            if uri == 'gs':
                self._save_file(file)
                move_files(file+'*', path[:-len(file)])
