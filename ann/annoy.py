import re
from typing import List, Optional, Union
from subprocess import check_call

import numpy as np
# import nmslib
from annoy import AnnoyIndex

from ann.base_ann import AnnBase


class Annoy(AnnBase):
    def __init__(self, vector_len: int, metric: str = 'angular', **kwargs):
        super().__init__(**kwargs)
        self.index = AnnoyIndex(vector_len, metric)

    def build_index(self, num_trees: int = 30):
        self.index.build(num_trees)

    def add_data(self, data: np.ndarray):
        for i, embed in enumerate(data):
            self.index.add_item(i, embed)

    def search_vec_top_n(self, vector, n: int = 5):
        neighbours, distance = self.index.get_nns_by_vector(vector, n, include_distances=True)
        items = neighbours
        if self.mapping:
            items = []
            for idx in neighbours:
                items.append(self.mapping[idx])
        return items, distance

    def calc_distance(self, a, b):
        distance = self.index.get_distance(int(a), int(b))
        return distance

    def _load_file(self, path: str, **kwargs):
        self.index.load(path)

    def _save_file(self, path: str):
        self.index.save(path)


# class NMSLib(AnnBase):
#     def __init__(self, method: str = 'hnsw',
#                  metric: str = 'cosinesimil', **kwargs):
#         super().__init__(**kwargs)
#         self.index = nmslib.init(method=method, space=metric)

#     def build_index(self,
#                     data: Union[List[np.ndarray], np.ndarray],
#                     ids: Optional[np.ndarray] = None,
#                     m: int = 20, ef: int = 300, post: int = 2):
#         if isinstance(data, list):
#             data = np.vstack(data)
#         self.index.addDataPointBatch(data, ids)
#         index_params = {
#             'M': m,
#             'efConstruction': ef,
#             'post': post}
#         self.index.createIndex(index_params)

#     def search_vec_top_n(self, vector, n: int = 5):
#         neighbours, distance = self.index.knnQuery(vector, k=n)
#         items = neighbours.tolist()
#         if self.mapping:
#             items = []
#             for idx in neighbours:
#                 items.append(self.mapping[idx])
#         return items, distance.tolist()

#     def calc_distance(self, a, b):
#         distance = self.index.getDistance(a, b)
#         return distance

#     def _load_file(self, path: str, ef: Optional[int] = None):
#         self.index.loadIndex(path, load_data=True)
#         if ef is not None:
#             self.index.setQueryTimeParams({'efSearch': ef})

#     def _save_file(self, path: str):
#         self.index.saveIndex(path, save_data=True)