import gym
from gym.wrappers import NormalizeObservation
import pickle
import numpy as np

from utils.normalize_obs import NormalizeObsAlternative


class RenderWrapper:
    """ This method will help me add new functionalities to env without destructing its interface.
    But be very careful during usage. Since there are two environments, changes applied to one won't be
    applied to others. Do this active/deactive just before the reset otherwise, problems might occur."""

    def __init__(self, env_name:str, normalize_obs=False):
        self.env_name = env_name
        self.normalize_obs = normalize_obs
        self._env = NormalizeObsAlternative(gym.make(env_name)) if normalize_obs else gym.make(env_name) # Original env.
        self._env_render = NormalizeObsAlternative(gym.make(env_name, render_mode="human")) if normalize_obs else gym.make(env_name, render_mode="human")
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

    def set_obs_rms(self, obs_rms):
        """ Sets the obs_rms of the inner envs."""
        if self.normalize_obs:
            self._env.obs_rms = obs_rms
            self._env_render.obs_rms = obs_rms
        else:
            print("You are trying to set obs_rms of an environment that is not normalized!")

    def freeze_obs_rmss(self):
        """ Freezes the obs_rms update by method injection."""
        if not self.normalize_obs: raise Exception("You are trying to freeze obs_rmss but normalize_obs=False!")

        def new_normalize_to_inject(normalize_obs_instance, obs):
            """This method is just there to be injected to NormalizeObservation wrapper to prevent mean and variance updates.
            It is not a good practice, but python allows dynamic method injection."""
            return (obs - normalize_obs_instance.obs_rms.mean) / np.sqrt(normalize_obs_instance.obs_rms.var + normalize_obs_instance.epsilon)

        # Assuming 'obj' is an instance of NormalizeObservation
        self._env.normalize = new_normalize_to_inject.__get__(self._env, NormalizeObservation)
        self._env_render.normalize = new_normalize_to_inject.__get__(self._env_render, NormalizeObservation)