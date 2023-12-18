import enum
from enum import Enum


class AdvNormMethods(enum.IntEnum):
    not_normalize = 0
    normalize = 1
    range_scale = 2

class AssitiveActors(enum.IntEnum):
    mountaincar_basic = 0
    mountaincar_joystick = 1
    mountaincar_pretrained = 2
    lunar_lander_joystick = 3

class NoiseUpdateFreq(enum.IntEnum):
    every_action = 0
    every_episode = 1
    every_rollout = 2

class InjectionTypes(enum.IntEnum):
    decremental = 0
    scheduled = 1