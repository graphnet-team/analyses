import numpy as np
import pickle

from graphnet.data.sqlite_dataset import SQLiteDataset


with open('/home/iwsatlas1/timg/remote/icedata/export/feature_map.pkl', 'rb') as f:
    feature_map = pickle.load(f)


def map_feature(feature: tuple):
    _, dom_x, dom_y, dom_z, *_ = feature
    icedata = feature_map[dom_x, dom_y, dom_z]
    return *feature, *icedata[[0, 1, 2, 5]]
    # return *feature, *icedata[[0, 1]]


with open('/home/iwsatlas1/timg/remote/icedata/export/feature_map_transformed.pkl', 'rb') as f:
    feature_map_transformed = pickle.load(f)


def map_feature_transformed(feature: tuple):
    _, dom_x, dom_y, dom_z, *_ = feature
    icedata = feature_map_transformed[dom_x, dom_y, dom_z]
    return *feature, *icedata[[0, 1, 2, 5]]
    # return *feature, *icedata[[0, 1]]


class SQLiteDataset_IDON_Tilt(SQLiteDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Modify _featues for _create_graph() but not _features_str for _query_database()
        self._features = self._features + ['A', 'B', 'C', 'F']
        # self._features = self._features + ['A', 'B']

    def __getitem__(self, i):
        self.establish_connection(i)
        features, truth, node_truth = self._query_database(i)

        # Add ice data columns to features
        # print('before', features)
        features = np.apply_along_axis(map_feature, 1, features)
        # print('after ', features)

        graph = self._create_graph(features, truth, node_truth)
        return graph


class SQLiteDataset_IDON_Tilt_Transformed(SQLiteDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Modify _featues for _create_graph() but not _features_str for _query_database()
        self._features = self._features + ['A', 'B', 'C', 'F']
        # self._features = self._features + ['A', 'B']

    def __getitem__(self, i):
        self.establish_connection(i)
        features, truth, node_truth = self._query_database(i)

        # Add ice data columns to features
        # print('before', features)
        features = np.apply_along_axis(map_feature_transformed, 1, features)
        # print('after ', features)

        graph = self._create_graph(features, truth, node_truth)
        return graph


if __name__ == '__main__':
    from graphnet.data.constants import FEATURES, TRUTH

    _features = FEATURES.DEEPCORE
    _truth = TRUTH.DEEPCORE
    del _truth[_truth.index('interaction_time')]

    database = '/remote/ceph/user/t/timg/dev_lvl7_robustness_muon_neutrino_0000.db'
    # selection, _ = get_even_track_cascade_indicies(database)
    selection = [44516992, 37881223, 80121006, 37948011, 80050979, 75630085, 64468308, 24470418, 22342000, 55703454, 4466568, 46733838, 75633164, 46815642, 29014936, 35556964, 42320583, 11284724, 64455600, 93379700, 4549955, 77891946, 11322109, 73377424, 24647, 22241198, 28913, 93352520, 89037020, 75582610, 53387900, 44539423, 51238046, 57805718, 26765062, 11200363, 57876275, 49065189, 60055789, 66671238, 93470873, 22363977, 80185504, 60056900, 44606772, 62305843, 82262109, 48920539, 28964839, 11124913,
                 86871212, 46834046, 53433347, 80123755, 91190690, 24516429, 84591227, 11115480, 35662416, 28906425, 11113405, 82263808, 11239335, 57882283, 97822741, 77855909, 55600465, 22224630, 57812568, 82349145, 84627178, 17894775, 86708035, 97908321, 86851591, 73467289, 91265015, 24502826, 91240322, 168252, 11323440, 62389111, 24552456, 64525355, 73478253, 53347994, 28967065, 53443053, 40088495, 10241, 44603004, 97883173, 49022815, 24462232, 24555037, 93455364, 22412591, 77883608, 42333533, 31255169]
    dataset = SQLiteDataset_IDON_Tilt_Transformed(
        database=database,
        pulsemaps='SRTTWOfflinePulsesDC',
        features=_features,
        truth=_truth,
        selection=selection,
        node_truth=None,
        node_truth_table=None,
        string_selection=None,
    )

    for i in range(10):
        graph = dataset[i]
        print(i, graph.features)
        print(i, graph.x)
