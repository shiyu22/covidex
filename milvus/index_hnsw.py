import os

# import hnswlib
import numpy as np

import helper
from milvus import Milvus, IndexType, MetricType, Status

_HOST = '127.0.0.1'
_PORT = '19690'
milvus = Milvus(_HOST, _PORT)

class Indexer:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def get_path(self, path) -> str:
        return os.path.join(self.folder_path, path)

    def load_data(self) -> None:
        self.metadata_path = self.get_path('metadata.csv')
        self.metadata = helper.load_metadata(self.metadata_path)

        self.specter_path = self.get_path('specter.csv')
        self.embedding, self.dim = helper.load_specter_embeddings(
            self.specter_path)
        self.num_elements = len(self.embedding)

        print(f'Number of embeddings: {self.num_elements}')
        print(f'Embedding dimension: {self.dim}')
        assert len(self.metadata) == len(self.embedding), "Data size mismatch"

    # def initialize_hnsw_index(self) -> None:
    #     # Declaring index
    #     # possible options are l2, cosine or ip
    #     milvus = Milvus(_HOST, _PORT)

    #     # Create collection demo_collection if it dosen't exist.
    #     collection_name = 'example_collection_'

    #     status, ok = milvus.has_collection(collection_name)
    #     if not ok:
    #         param = {
    #             'collection_name': collection_name,
    #             'dimension': _DIM,
    #             'index_file_size': _INDEX_FILE_SIZE,  # optional
    #             'metric_type': MetricType.L2  # optional
    #         }

    #         milvus.create_collection(param)
    #     # create index of vectors, search more rapidly
    #     index_param = {M: 16, efConstruction:500}

    #     # Create ivflat index in demo_collection
    #     # You can search vectors without creating index. however, Creating index help to
    #     # search faster
    #     print("Creating index: {}".format(index_param))
    #     status = milvus.create_index(collection_name, IndexType.HNSW, index_param)


    def index_and_save(self) -> None:
        print('[HNSW] Starting to index...')
        data = np.empty((0, self.dim))
        data_labels = []
        index_to_uid = []

        for index, uid in enumerate(self.embedding):
            if index % 1000 == 0:
                print(
                    f'[HNSW] Indexed {index}/{self.num_elements}')

            if index % 200 == 0 and len(data_labels) > 0:
                # save progress
                self._add_to_index(data, data_labels,  index)
                # reset
                data = np.empty((0, self.dim))
                data_labels = []

            vector = self.embedding[uid]
            assert len(vector) == self.dim, "Vector dimension mismatch"
            data = np.concatenate((data, [vector]))
            data_labels.append(index)
            index_to_uid.append(uid)

        if len(data_labels) > 0:
            self._add_to_index(data, data_labels, index)
            self._save_index(data, data_labels, index_to_uid, index)

        print('[HNSW] Finished indexing')

    def _add_to_index(self, data, data_labels, index):
        collection_name = 'example_collection_'

        status, ok = milvus.has_collection(collection_name)
        if not ok:
            param = {
                'collection_name': collection_name,
                'dimension': _DIM,
                'index_file_size': _INDEX_FILE_SIZE,  # optional
                'metric_type': MetricType.L2  # optional
            }

            milvus.create_collection(param)

        print("--------data--------", data)
        print("--------index--------", index)
        status, ids = milvus.insert(collection_name=collection_name, records=data, ids=index)

        # create index of vectors, search more rapidly
        index_param = {M: 16, efConstruction:500}

        # Create ivflat index in demo_collection
        # You can search vectors without creating index. however, Creating index help to
        # search faster
        print("Creating index: {}".format(index_param))
        status = milvus.create_index(collection_name, IndexType.HNSW, index_param)

    def _save_index(self, data, data_labels, index_to_uid, index):
        print('[HNSW] Saving index', index)

        file_name = 'cord19-hnsw-milvus'
        # output_path = self.get_path(f'{file_name}.bin')
        # helper.remove_if_exist(output_path)
        # self.hnsw.save_index(output_path)

        # Save index to uid file
        helper.save_index_to_uid_file(
            index_to_uid,
            index,
            self.get_path(f'{file_name}.txt'))


if __name__ == '__main__':
    indexer = Indexer("./api/index/cord19-hnsw-index-milvus")
    indexer.load_data()
    # indexer.initialize_hnsw_index()
    indexer.index_and_save()
