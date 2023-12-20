import time
from abc import ABC, abstractmethod
import torch
import numpy as np

from injection.joystick import Joystick


class LunarLanderAssistiveActor(ABC):
    """ This is the base class to define assitive actors for lunarlander environment.
    Assitive actors are basically rule based or more advanced actors that assists the
    currently training agent."""

    def get_action(self, state):
        """ All assistive actors will accept a state and return the action in return."""
        self.state = state
        return self._action_strategy()

    @abstractmethod
    def _action_strategy(self):
        ...


class JoystickLunarLanderAssistiveActor(LunarLanderAssistiveActor):
    """ This assitive actor uses the axis0 and axis1 input from the joystick to provide actions. Also an additional stabilization
    method is utilized to control the lunar lander easier, without this rule based stabilization method, it is almost impossible
    to delicately control the lunar lander through joystick."""

    def __init__(self):
        super().__init__()
        self.joystick = Joystick()

    def _action_strategy(self):
        time.sleep(0.05)  # slow down time to make the task easier for human operator.

        # Get joystick inputs
        axis_thrust = -self.joystick.axis_1
        axis_steering = self.joystick.axis_0
        if axis_thrust < 0.03: axis_thrust = 0  # to prevent small residual commands on joystick

        # Apply stabilization with consideration to manual control
        axis_steering = -self._stabilize_lander(self.state, -axis_steering, axis_steering)

        # Return the action
        return torch.tensor([axis_thrust, axis_steering], dtype=torch.float32)

    @staticmethod
    def _stabilize_lander(observation, axis_steering, manual_control_intensity):
        """
        Stabilize the Lunar Lander based on its current angle and angular velocity.
        Less aggressive stabilization when there's manual control input.
        """
        angle, angular_velocity = observation[4], observation[5]

        # Adjust these thresholds and factors based on trial and error
        ANGLE_THRESHOLD = 0.05
        ANGULAR_VELOCITY_THRESHOLD = 0.1
        ANGLE_STABILIZATION_FACTOR = 1.0  # How aggressively to counteract the angle
        ANGULAR_VELOCITY_STABILIZATION_FACTOR = 1.0  # How aggressively to counteract the angular velocity

        stabilization = 0
        if abs(angle) > ANGLE_THRESHOLD:
            stabilization -= angle * ANGLE_STABILIZATION_FACTOR
        if abs(angular_velocity) > ANGULAR_VELOCITY_THRESHOLD:
            stabilization -= angular_velocity * ANGULAR_VELOCITY_STABILIZATION_FACTOR

        # Reduce stabilization intensity when there's manual control
        stabilization_intensity = 1 - abs(manual_control_intensity)
        stabilization *= stabilization_intensity

        # Combine user input with stabilization
        return np.clip(axis_steering + stabilization, -1, 1)