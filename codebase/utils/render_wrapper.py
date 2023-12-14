import gym
from gym.wrappers import NormalizeObservation
import pickle


class RenderWrapper:
    """ This method will help me add new functionalities to env without destructing its interface.
    But be very careful during usage. Since there are two environments, changes applied to one won't be
    applied to others. Do this active/deactive just before the reset otherwise, problems might occur."""

    def __init__(self, env_name:str, normalize_obs=False):
        self.normalize_obs = normalize_obs
        self._env = NormalizeObservation(gym.make(env_name)) if normalize_obs else gym.make(env_name) # Original env.
        self._env_render = NormalizeObservation(gym.make(env_name, render_mode="human")) if normalize_obs else gym.make(env_name)
        if normalize_obs: self._env_render.obs_rms = self._env.obs_rms  # Let them use the exact same obs_rms to provide convenience
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

    def save_obs_rms(self, filename='obs_rms.pkl'):
        """This method saves the normalization parameters, to later use for demo purposes. """
        # Check if the active environment is an instance of NormalizeObservation
        if isinstance(self._active_env, NormalizeObservation):
            # Save obs_rms using pickle
            with open(filename, 'wb') as f:
                pickle.dump(self._active_env.obs_rms, f)
        else:
            print("You've tried to save obs_rms but normalization is not enabled in this environment.")
            return
