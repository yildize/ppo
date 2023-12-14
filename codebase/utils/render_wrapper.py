import gym
from gym.wrappers import NormalizeObservation


class RenderWrapper:
    """ This method will help me add new functionalities to env without destructing its interface.
    But be very careful during usage. Since there are two environments, changes applied to one won't be
    applied to others. Do this active/deactive just before the reset otherwise, problems might occur."""

    def __init__(self, env_name:str):
        self._env = NormalizeObservation(gym.make(env_name))# Original env.
        self._env_render = NormalizeObservation(gym.make(env_name, render_mode="human"))
        self._active_env = self._env # by default active env is the no render one.

    def __getattr__(self, name):
        # This method is called when an attribute is not found in the object
        # It will pass the call to the wrapped environment object
        return getattr(self._active_env, name)

    def activate_rendering(self):
        """ Be very careful any change on previous env won't be applied to new one."""
        self._active_env = self._env_render

    def inactive_rendering(self):
        """ Be very careful any change on previous env won't be applied to new one."""
        self._active_env = self._env