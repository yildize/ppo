from injection.assistive_actors.lunarlander import JoystickLunarLanderAssistiveActor
from injection.assistive_actors.mountaincar import DummyMountainCarAssitiveActor, JoystickMountainCarAssistiveActor, \
    PreTrainedMountainCarAssistiveActor
from utils.enums import AssitiveActors


class AssitiveActorFactory:
    """ This is a utility factor class that is used to instantiate required assistive actor objects"""
    @staticmethod
    def create(assitive_actor:AssitiveActors):
        if assitive_actor is AssitiveActors.mountaincar_basic:
            return DummyMountainCarAssitiveActor()
        elif assitive_actor is AssitiveActors.mountaincar_joystick:
            return JoystickMountainCarAssistiveActor()
        elif assitive_actor is AssitiveActors.mountaincar_pretrained:
            return PreTrainedMountainCarAssistiveActor()
        elif assitive_actor is AssitiveActors.lunar_lander_joystick:
            return JoystickLunarLanderAssistiveActor()