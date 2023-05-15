import numpy as np
import pickle

from graphnet.data.sqlite_dataset import SQLiteDataset


with open('/home/iwsatlas1/timg/remote/icedata/export/rde/feature_map.pkl', 'rb') as f:
    feature_map = pickle.load(f)


def map_feature(features: tuple):
    _, dom_x, dom_y, dom_z, *_ = features
    features_additional = feature_map[dom_x, dom_y, dom_z]
    return *features, *features_additional


with open('/home/iwsatlas1/timg/remote/icedata/export/rde/feature_map_transformed_robust.pkl', 'rb') as f:
    feature_map_transformed = pickle.load(f)


def map_feature_transformed(features: tuple):
    _, dom_x, dom_y, dom_z, *_ = features
    features_additional = feature_map_transformed[dom_x, dom_y, dom_z]
    return *features, *features_additional


class SQLiteDataset_RDE(SQLiteDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Modify _featues for _create_graph() but not _features_str for _query_database()
        self._features = self._features + ['rde', 'type']

    def __getitem__(self, i):
        self.establish_connection(i)
        features, truth, node_truth = self._query_database(i)

        # Add ice data columns to features
        # print('before', features)
        features = np.apply_along_axis(map_feature, 1, features)
        # print('after ', features)

        graph = self._create_graph(features, truth, node_truth)
        return graph


class SQLiteDataset_RDE_Transformed(SQLiteDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Modify _featues for _create_graph() but not _features_str for _query_database()
        self._features = self._features + ['rde', 'type']

    def __getitem__(self, i):
        self.establish_connection(i)
        features, truth, node_truth = self._query_database(i)

        # Add ice data columns to features
        # print('before', features)
        features = np.apply_along_axis(map_feature_transformed, 1, features)
        # print('after ', features)

        graph = self._create_graph(features, truth, node_truth)
        return graph
