
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.data.constants import FEATURES


class IceCubeDeepCore_IDON(IceCubeDeepCore):
    features = FEATURES.ICECUBE86 + ['A', 'B']
