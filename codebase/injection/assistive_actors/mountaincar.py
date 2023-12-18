import json
from abc import ABC, abstractmethod
import torch

from core.networks import MLP
from injection.joystick import Joystick
from utils.utils import load_with_pickle, find_first_file_in_directory


class BaseMountainCarAssistiveActor(ABC):
    """ This is the base class to define assitive actors for mountain car.
    Assitive actors are basically rule based or more advanced actors that assists the
    currently training agent."""

    def __init__(self):
        self.x = None
        self.v = None

    def get_action(self, state):
        self.state = state
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

class PreTrainedMountainCarAssistiveActor(BaseMountainCarAssistiveActor):

    def __init__(self):
        super().__init__()
        self.hyperparams_dict = self.__load_hyperparams_dict()
        self.actor = self.__create_empty_policy_network()
        self.__load_demo_actor()

    def _action_strategy(self):
        with (torch.no_grad()):
            return self.actor(self.state)

    def __load_hyperparams_dict(self):
        # First read the hyperparams from the demo folder to reconstruct the actor:
        hyperparams_path = find_first_file_in_directory(directory_path=f"./injection/assistive_actors/pretrained/mountain_car", containing="hyperparams")
        if hyperparams_path is None: raise FileNotFoundError(f"Please place a hyperparams file inside the ./injection/assitive_actors/pretrained/mountain_car so that, assistive actor "
                                                             f"can reconstruct the same actor.")
        with open(hyperparams_path, 'r') as f:
            hyperparams_dict = json.load(f)
        return hyperparams_dict

    def __create_empty_policy_network(self, ) -> torch.nn.Module:
        # Construct the policy/actor network same with the saved model so that we can successfully load the weights.
        policy:torch.nn.Module = MLP(input_dim=2, output_dim=1,
                                     hidden_dim=self.hyperparams_dict["hidden_dim"], num_hidden_layers=self.hyperparams_dict["num_hidden_layers_actor"],
                                     learn_std=self.hyperparams_dict["learn_std"], tanh_acts=self.hyperparams_dict["tanh_acts"])
        return policy

    def __load_demo_actor(self):
        model_path = find_first_file_in_directory(directory_path=f"./injection/assistive_actors/pretrained/mountain_car", containing="actor")


        # Load in the actor model saved by the PPO algorithm
        self.actor.load_state_dict(torch.load(model_path))



class JoystickMountainCarAssistiveActor(BaseMountainCarAssistiveActor):
    """ This assitive actor uses the axis0 input from the joystick to provide actions"""
    def __init__(self):
        super().__init__()
        self.joystick = Joystick()

    def _action_strategy(self):
        return torch.tensor([self.joystick.axis_0], dtype=torch.float32)


class SB3AssitiveActor(BaseMountainCarAssistiveActor):
    """ Assitive actors can be other trained actors as well, rather than being rule based dummy logics.
    But pay attention to normalization compatability in this case."""
    def __init__(self):
        super().__init__()
        self.actor = ...

    def _action_strategy(self):
        ...
