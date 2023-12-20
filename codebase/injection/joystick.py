import time
import pygame


class Joystick:
    """ Utility module to seamlessly integrate joystick. Simply this module can allow easy access to
    various joystick inputs and provide utility methods to interact with the joystick."""

    def __init__(self, yes_button:int=0, no_button:int=1):
        # Initialize Pygame and the joystick
        pygame.init()
        pygame.display.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("Error, no joystick is found.")
        else:
            # Use joystick #0 and initialize it
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

        self.yes_button = yes_button
        self.no_button = no_button

    @property
    def axis_0(self):
        pygame.event.pump()
        return self.joystick.get_axis(0)

    @property
    def axis_1(self):
        pygame.event.pump()
        return self.joystick.get_axis(1)

    @property
    def axis_2(self):
        pygame.event.pump()
        return self.joystick.get_axis(2)

    @property
    def activate_injection(self):
        pygame.event.pump()
        return self.joystick.get_button(self.yes_button)

    def wait_for_button_press(self, info="Please press any joystick button.", max_wait_time_s=5):
        st = time.time()
        print(info)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"Button {event.button} pressed.")
                    return event.button
                if time.time() - st > max_wait_time_s: return None


if __name__ == "__main__":
    joystick = Joystick()

    while True:
        button = joystick.wait_for_button_press()
        print(button)