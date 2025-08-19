from pynput import keyboard
import numpy as np

class KeyboardInput:
    def __init__(self):
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.gripper_position = 0
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            key_char = key.char.lower()
            if key_char in ['w', 'a', 's', 'd', 'q', 'e']:
                linear_mapping = {
                    'w': [5, 0, 0],
                    's': [-5, 0, 0],
                    'a': [0, 5, 0],
                    'd': [0, -5, 0],
                    'q': [0, 0, 5],
                    'e': [0, 0, -5],
                }
                self.linear_velocity = np.array(linear_mapping[key_char])
            elif key_char in ['i', 'j', 'k', 'l', 'u', 'o']:
                angular_mapping = {
                    'i': [1.25, 0, 0],
                    'k': [-1.25, 0, 0],
                    'j': [0, 1.25, 0],
                    'l': [0, -1.25, 0],
                    'u': [0, 0, 1.25],
                    'o': [0, 0, -1.25],
                }
                self.angular_velocity = np.array(angular_mapping[key_char])

        except AttributeError:
            if key == keyboard.Key.space:
                # Toggle gripper position between 0 and 255
                self.gripper_position = 255 if self.gripper_position == 0 else 0

    def on_release(self, key):
        try:
            key_char = key.char.lower()
            if key_char in ['w', 'a', 's', 'd', 'q', 'e']:
                self.linear_velocity = np.zeros(3)
            elif key_char in ['i', 'j', 'k', 'l', 'u', 'o']:
                self.angular_velocity = np.zeros(3)
        except AttributeError:
            pass

    def get_action(self):
        action = np.concatenate((self.linear_velocity, self.angular_velocity, np.array([self.gripper_position])))
        return action