from abc import ABC, abstractmethod
import torch

from injection.joystick import Joystick


class BaseMountainCarAssistiveActor(ABC):
    """ This is the base class to define assitive actors for mountain car.
    Assitive actors are basically rule based or more advanced actors that assists the
    currently training agent."""

    def __init__(self):
        self.x = None
        self.v = None

    def get_action(self, state):
        self.x, self.v = state[0], state[1]
        return self._action_strategy()

    @abstractmethod
    def _action_strategy(self):
        ...


class DummyMountainCarAssitiveActor(BaseMountainCarAssistiveActor):
    """ This is the simplest mountain car actor that only reacts according to current
    velocity. It basically always tries to increase the abs(velocity) so that it can swing
    This simple strategy achieves episode rewards of around 92."""
    def __init__(self):
        super().__init__()

    def _action_strategy(self):
        if self.v <= 0: rec_action = torch.tensor([-1], dtype=torch.float32)
        else:rec_action=torch.tensor([1], dtype=torch.float32)
        return rec_action


class JoystickAssistiveActor(BaseMountainCarAssistiveActor):
    """ This assitive actor uses the axis0 input from the joystick to provide actions"""
    def __init__(self):
        super().__init__()
        self.joystick = Joystick()

    def _action_strategy(self):
        return torch.tensor([self.joystick.axis_0], dtype=torch.float32)