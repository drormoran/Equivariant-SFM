from enum import Enum


class Phases(Enum):
    OPTIMIZATION = 1
    TRAINING = 2
    VALIDATION = 3
    TEST = 4
    FINE_TUNE = 5
    SHORT_OPTIMIZATION = 6