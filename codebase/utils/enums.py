import enum
from enum import Enum


class AdvNormMethods(enum.IntEnum):
    not_normalize = 0
    normalize = 1
    range_scale = 2


class AssitiveActors(enum.IntEnum):
    mountaincar_basic = 0
    mountaincar_joystick = 1