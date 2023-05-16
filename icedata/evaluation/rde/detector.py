
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.data.constants import FEATURES


class IceCubeDeepCore_RDE(IceCubeDeepCore):
    features = FEATURES.ICECUBE86 + ['rde', 'type']
