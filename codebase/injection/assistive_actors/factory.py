from injection.assistive_actors.mountaincar import DummyMountainCarAssitiveActor, JoystickMountainCarAssistiveActor
from utils.enums import AssitiveActors


class AssitiveActorFactory:
    @staticmethod
    def create(assitive_actor:AssitiveActors):
        if assitive_actor is AssitiveActors.mountaincar_basic:
            return DummyMountainCarAssitiveActor()
        elif assitive_actor is AssitiveActors.mountaincar_joystick:
            return JoystickMountainCarAssistiveActor()