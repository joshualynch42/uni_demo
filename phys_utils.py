from gym import Env
from cri.robot import SyncRobot
from cri.controller import MagicianController as Controller
from vsp.video_stream import CvImageOutputFileSeq, CvVideoDisplay, CvPreprocVideoCamera
from vsp.processor import CameraStreamProcessor, AsyncProcessor
import matplotlib.pylab as plt # needs to be after initialising controller (strange bug)
import pandas as pd
import string
import time
import numpy as np
from msvcrt import getch
import random, os
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

alphabet_arr = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D',
        'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'SPACE']
arrow_arr = ['UP', 'DOWN', 'LEFT', 'RIGHT']
key_coords = pd.read_csv(r"D:\Users\Josh\github\uni_demo\key_coords.csv")

def make_sensor(): # amcap: reset all settings; autoexposure off; saturdation max
    camera = CvPreprocVideoCamera(source=1,  # might need changing for webcam
                crop=[320-128-10, 240-128+10, 320+128-10, 240+128+10],
                size=[128, 128],
                threshold=[61, -5],
                exposure=-6)
    for _ in range(5): camera.read() # Hack - camera transient

    return AsyncProcessor(CameraStreamProcessor(camera=camera,
                display=CvVideoDisplay(name='sensor'),
                writer=CvImageOutputFileSeq()))

sensor = make_sensor()
robot = SyncRobot(Controller())
robot.linear_speed = 60
robot.coord_frame = [0, 0, 0, 0, 0, 0] # careful

def translate_coord(coords):
    """
    Input: A 1x3 coordinate vector in the form [x, y, z]

    This function converts from imaginary to real coordinates and vice versa

    Returns: A translated 1x3 coordinate vector

    """
    x, y, z = coords[0], coords[1], coords[2]
    if x > 3:
        row = key_coords.loc[key_coords['X'] == x]
        row = row.loc[row['Y'] == y]
        if len(row) > 0:
            new_x = row['IM_X'].to_numpy()[0]
            new_y = row['IM_Y'].to_numpy()[0]
        else:
            print('Error: no matching coordinates found')
            exit()
    else:
        row = key_coords.loc[key_coords['IM_X'] == x]
        row = row.loc[row['IM_Y'] == y]
        if len(row) > 0:
            new_x = row['X'].to_numpy()[0]
            new_y = row['Y'].to_numpy()[0]
        else:
            print('Error: no matching coordinates found')
            exit()
    return [new_x, new_y, z]

def coords_to_letter(coords):
    x, y, z = coords[0], coords[1], coords[2]
    row = key_coords.loc[key_coords['X'] == x]
    row = row.loc[row['Y'] == y]
    if len(row) > 0:
        letter = row['Key'].to_numpy()[0]
    else:
        print('Error: no matching coordinates found')
        exit()

    return letter

def letter_to_coords(letter):
    row = key_coords.loc[key_coords['Key'] == letter]
    if len(row) > 0:
        x = row['X'].to_numpy()[0]
        y = row['Y'].to_numpy()[0]
        z = row['Z'].to_numpy()[0]
    else:
        print('Error: no matching coordinates found')
        exit()

    return [x, y, z]

def get_key():
    input_key = getch()
    input_key = input_key.decode("utf-8")
    input_key = input_key.upper()
    return input_key

def create_state(image, goal_letter):
    one_hot_arr = [0]*(len(key_coords))
    num = key_coords.index[key_coords['Key'] == goal_letter].tolist()[0]
    one_hot_arr[num] = 1
    state = {'img': image, 'vec': np.array(one_hot_arr)}
    return state  # current image and goal letter

def create_one_hot(letter):
    one_hot_arr = [0]*(len(key_coords))
    num = key_coords.index[key_coords['Key'] == letter].tolist()[0]
    one_hot_arr[num] = 1
    return np.array(one_hot_arr)

def get_image(coords):
    outfile = r"D:\Users\Josh\github\individual_project\physical\temp_photo.png"
    robot.move_linear([coords[0], coords[1], -37, 0, 0, 0])
    frames_sync = sensor.process(num_frames=1, start_frame=1, outfile=outfile)
    robot.move_linear([coords[0], coords[1], coords[2], 0, 0, 0])
    dir = "D:/Users/Josh/github/individual_project/physical/temp_photo_0.png"
    img = load_img(dir, color_mode='grayscale', target_size=(64,64))
    img = img_to_array(img).astype('float32') / 255
    return img

def move_phys_dobot(coords):
    robot.move_linear([coords[0], coords[1], coords[2], 0, 0, 0])

def press_phys_dobot(coords):
    robot.move_linear([coords[0], coords[1], -38, 0, 0, 0])
    robot.move_linear([coords[0], coords[1], coords[2], 0, 0, 0])

class phys_discrete_arrow_env(Env):
    def __init__(self):
        self.current_step = 0
        self.starting_letter = random.choice(arrow_arr)
        self.current_coords = letter_to_coords(self.starting_letter)
        move_phys_dobot(self.current_coords)
        self.goal_letter = random.choice(arrow_arr)
        self.action_array = np.array(['up', 'left', 'right', 'down', 'pressdown'])
        self.state = create_state(get_image(self.current_coords), self.goal_letter)
        self.img_shape = np.shape(self.state['img'])
        self.max_ep_len = 25
        self.action_space_size = len(self.action_array)

    def reset(self):
        self.current_step = 0
        self.starting_letter = random.choice(arrow_arr)
        self.current_coords = letter_to_coords(self.starting_letter)
        move_phys_dobot(self.current_coords)
        self.goal_letter = random.choice(arrow_arr)
        self.state = create_state(get_image(self.current_coords), self.goal_letter)

        return self.state

    def move_dobot(self, action):
        self.current_coords = translate_coord(self.current_coords) #turn coords into imaginary coords

        if action == 'up':  # all values become up key
            if self.current_coords[0] != 2 or self.current_coords[1] != 43:
                self.current_coords = np.array([2, 43, -25])

        elif action == 'right':
            if self.current_coords[1] == 43:  # up and down become right
                self.current_coords = np.array([3, 46, -25])
            elif self.current_coords[1] == 40:
                self.current_coords = np.array([3, 43, -25])

        elif action == 'left':
            if self.current_coords[1] == 43:  # up and down become right
                self.current_coords = np.array([3, 40, -25])
            elif self.current_coords[1] == 46:
                self.current_coords = np.array([3, 43, -25])

        elif action == 'down':
            if self.current_coords[0] == 2 and self.current_coords[1] == 43:
                self.current_coords = np.array([3, 43, -25])

        else:
            print('Error: Not a valid action')
            exit()
        self.current_coords = translate_coord(self.current_coords) #turn coords into real
        move_phys_dobot(self.current_coords) #move robot to new coords
        current_img = get_image(self.current_coords) #get new image of state
        self.state = create_state(current_img, self.goal_letter)

    def step(self, action, steps):
        done = False
        info = []
        if steps > self.max_ep_len -1:
            reward = 0
        else:
            action = self.action_array[action]
            if action == 'pressdown':
                done = True
                # press_phys_dobot(self.current_coords) # move dobot to press key
                # actual_let = get_key() # get key press from console
                # print(actual_let)
                cur_let = coords_to_letter(self.current_coords)
                goal_let = self.goal_letter
                if cur_let == goal_let:
                    reward = 1
                else:
                    reward = 0
            else:
                reward = 0
                prev_state = self.current_coords
                self.move_dobot(action)
                if np.array_equal(prev_state, self.current_coords, equal_nan=False):
                    reward = 0  # potential punishment for not leaving the key

        return self.state, reward, done, info

    def render(self):
        pass

class phys_discrete_alphabet_env(Env):
    def __init__(self):
        self.current_step = 0
        self.starting_letter = random.choice(alphabet_arr)
        self.goal_letter = random.choice(alphabet_arr)
        self.current_coords = letter_to_coords(self.starting_letter)
        move_phys_dobot(self.current_coords)
        self.action_array = np.array(['upleft', 'upright', 'left', 'right', 'downleft', 'downright', 'pressdown'])
        self.state = create_state(get_image(self.current_coords), self.goal_letter)
        self.img_shape = np.shape(get_image(self.current_coords))
        self.max_ep_len = 16
        self.action_space_size = len(self.action_array)

    def reset(self):
        self.current_step = 0
        self.starting_letter = random.choice(alphabet_arr)
        self.goal_letter = random.choice(alphabet_arr)
        self.current_coords = letter_to_coords(self.starting_letter)
        move_phys_dobot(self.current_coords)
        self.state = create_state(get_image(self.current_coords), self.goal_letter)

        return self.state

    def move_dobot(self, action):
        self.current_coords = translate_coord(self.current_coords) #turn coords into imaginary coords

        if action == 'upleft':
            if self.current_coords[0] == 3 and self.current_coords[1] == 14:
                self.current_coords = np.array([2, 11, -25])
            elif self.current_coords[0] > 0 and self.current_coords[1] > 0:
                self.current_coords = self.current_coords + np.array([-1, -1, 0])

        elif action == 'upright':
            if self.current_coords[0] == 3 and self.current_coords[1] == 14:
                self.current_coords = np.array([2, 14, -25])
            elif self.current_coords[0] > 0:
                self.current_coords = self.current_coords + np.array([-1, 2, 0])

        elif action == 'left':
            if self.current_coords[0] < 3 and self.current_coords[1] > 2:
                self.current_coords = self.current_coords + np.array([0, -3, 0])

        elif action == 'right':
            if self.current_coords[0] < 2 and self.current_coords[1] < 25:
                self.current_coords = self.current_coords + np.array([0, 3, 0])
            elif self.current_coords[0] < 3 and self.current_coords[1] < 18:
                self.current_coords = self.current_coords + np.array([0, 3, 0])

        elif action == 'downleft':
            if self.current_coords[0] == 1 and self.current_coords[1] > 24:
                pass
            elif self.current_coords[0] < 2 and self.current_coords[1] > 2:
                self.current_coords = self.current_coords + np.array([1, -2, 0])
            elif self.current_coords[0] == 2 and self.current_coords[1] > 5: # bottom row not z or x
                self.current_coords = np.array([3, 14, -25])

        elif action == 'downright':
            if self.current_coords[0] == 1 and self.current_coords[1] > 19:
                pass
            elif self.current_coords[0] < 2 and self.current_coords[1] < 25:
                self.current_coords = self.current_coords + np.array([1, 1, 0])
            elif self.current_coords[0] == 2 and self.current_coords[1] > 2: # bottom row not z
                self.current_coords = np.array([3, 14, -25])

        else:
            print('Error: Not a valid action')
            exit()
        self.current_coords = translate_coord(self.current_coords) #turn coords into real
        move_phys_dobot(self.current_coords) #move robot to new coords
        current_img = get_image(self.current_coords) #get new image of state
        self.state = create_state(current_img, self.goal_letter)

    def step(self, action, steps):
        done = False
        info = []
        if steps > self.max_ep_len -1:
            reward = 0
        else:
            action = self.action_array[action]
            if action == 'pressdown':
                done = True
                # press_phys_dobot(self.current_coords) # move dobot to press key
                # actual_let = get_key() # get key press from console
                # print('actual letter', actual_let)
                cur_let = coords_to_letter(self.current_coords)
                goal_let = self.goal_letter
                if cur_let == goal_let:
                    reward = 1
                else:
                    reward = 0
            else:
                reward = 0
                prev_state = self.current_coords
                self.move_dobot(action)
                if np.array_equal(prev_state, self.current_coords, equal_nan=False):
                    reward = 0  # potential punishment for not leaving the key

        return self.state, reward, done, info

    def render(self):
        pass


class phys_discrete_arrow_env_pong(Env):
    def __init__(self, current_letter, goal_letter):
        self.current_step = 0
        self.starting_letter = current_letter
        self.goal_letter = goal_letter
        self.current_coords = letter_to_coords(self.starting_letter)
        move_phys_dobot(self.current_coords)
        self.action_array = np.array(['up', 'left', 'right', 'down', 'pressdown'])
        self.state = create_state(get_image(self.current_coords), self.goal_letter)
        self.img_shape = np.shape(self.state['img'])
        self.max_ep_len = 5
        self.action_space_size = len(self.action_array)

    def reset(self, current_letter, goal_letter):
        self.current_step = 0
        self.starting_letter = current_letter
        self.goal_letter = goal_letter
        self.current_coords = letter_to_coords(self.starting_letter)
        move_phys_dobot(self.current_coords)
        self.state = create_state(get_image(self.current_coords), self.goal_letter)

        return self.state

    def move_dobot(self, action):
        self.current_coords = translate_coord(self.current_coords) #turn coords into imaginary coords

        if action == 'up':  # all values become up key
            if self.current_coords[0] != 2 or self.current_coords[1] != 43:
                self.current_coords = np.array([2, 43, -25])

        elif action == 'right':
            if self.current_coords[1] == 43:  # up and down become right
                self.current_coords = np.array([3, 46, -25])
            elif self.current_coords[1] == 40:
                self.current_coords = np.array([3, 43, -25])

        elif action == 'left':
            if self.current_coords[1] == 43:  # up and down become right
                self.current_coords = np.array([3, 40, -25])
            elif self.current_coords[1] == 46:
                self.current_coords = np.array([3, 43, -25])

        elif action == 'down':
            if self.current_coords[0] == 2 and self.current_coords[1] == 43:
                self.current_coords = np.array([3, 43, -25])

        else:
            print('Error: Not a valid action')
            exit()
        self.current_coords = translate_coord(self.current_coords) #turn coords into real
        move_phys_dobot(self.current_coords) #move robot to new coords
        # current_img = get_image(self.current_coords) #get new image of state
        # self.state = create_state(current_img, self.goal_letter)

    def step(self, action, steps):
        done = False
        info = []
        if steps > self.max_ep_len -1:
            reward = 0
            done = True
        else:
            action = self.action_array[action]
            if action == 'pressdown':
                done = True
                # press_phys_dobot(self.current_coords) # move dobot to press key
                # actual_let = get_key() # get key press from console
                # print(actual_let)
                cur_let = coords_to_letter(self.current_coords)
                goal_let = self.goal_letter
                if cur_let == goal_let:
                    reward = 1
                else:
                    reward = 0
            else:
                reward = 0
                prev_state = self.current_coords
                self.move_dobot(action)
                if np.array_equal(prev_state, self.current_coords, equal_nan=False):
                    reward = 0  # potential punishment for not leaving the key

        return self.state, reward, done, info

    def render(self):
        pass

class phys_discrete_alphabet_env_pong(Env):
    def __init__(self, current_letter, goal_letter):
        self.current_step = 0
        self.starting_letter = current_letter
        self.goal_letter = goal_letter
        self.current_coords = letter_to_coords(self.starting_letter)
        move_phys_dobot(self.current_coords)
        self.action_array = np.array(['upleft', 'upright', 'left', 'right', 'downleft', 'downright', 'pressdown'])
        self.state = create_state(get_image(self.current_coords), self.goal_letter)
        self.img_shape = np.shape(get_image(self.current_coords))
        self.max_ep_len = 16
        self.action_space_size = len(self.action_array)

    def reset(self, current_letter, goal_letter):
        self.current_step = 0
        self.starting_letter = current_letter
        self.goal_letter = goal_letter
        self.current_coords = letter_to_coords(self.starting_letter)
        move_phys_dobot(self.current_coords)
        self.state = create_state(get_image(self.current_coords), self.goal_letter)

        return self.state

    def move_dobot(self, action):
        self.current_coords = translate_coord(self.current_coords) #turn coords into imaginary coords

        if action == 'upleft':
            if self.current_coords[0] == 3 and self.current_coords[1] == 14:
                self.current_coords = np.array([2, 11, -25])
            elif self.current_coords[0] > 0 and self.current_coords[1] > 0:
                self.current_coords = self.current_coords + np.array([-1, -1, 0])

        elif action == 'upright':
            if self.current_coords[0] == 3 and self.current_coords[1] == 14:
                self.current_coords = np.array([2, 14, -25])
            elif self.current_coords[0] > 0:
                self.current_coords = self.current_coords + np.array([-1, 2, 0])

        elif action == 'left':
            if self.current_coords[0] < 3 and self.current_coords[1] > 2:
                self.current_coords = self.current_coords + np.array([0, -3, 0])

        elif action == 'right':
            if self.current_coords[0] < 2 and self.current_coords[1] < 25:
                self.current_coords = self.current_coords + np.array([0, 3, 0])
            elif self.current_coords[0] < 3 and self.current_coords[1] < 18:
                self.current_coords = self.current_coords + np.array([0, 3, 0])

        elif action == 'downleft':
            if self.current_coords[0] == 1 and self.current_coords[1] > 24:
                pass
            elif self.current_coords[0] < 2 and self.current_coords[1] > 2:
                self.current_coords = self.current_coords + np.array([1, -2, 0])
            elif self.current_coords[0] == 2 and self.current_coords[1] > 5: # bottom row not z or x
                self.current_coords = np.array([3, 14, -25])

        elif action == 'downright':
            if self.current_coords[0] == 1 and self.current_coords[1] > 19:
                pass
            elif self.current_coords[0] < 2 and self.current_coords[1] < 25:
                self.current_coords = self.current_coords + np.array([1, 1, 0])
            elif self.current_coords[0] == 2 and self.current_coords[1] > 2: # bottom row not z
                self.current_coords = np.array([3, 14, -25])

        else:
            print('Error: Not a valid action')
            exit()
        self.current_coords = translate_coord(self.current_coords) #turn coords into real
        move_phys_dobot(self.current_coords) #move robot to new coords
        current_img = get_image(self.current_coords) #get new image of state
        self.state = create_state(current_img, self.goal_letter)

    def step(self, action, steps):
        done = False
        info = []
        if steps > self.max_ep_len -1:
            reward = 0
            done = True
        else:
            action = self.action_array[action]
            if action == 'pressdown':
                done = True
                press_phys_dobot(self.current_coords) # move dobot to press key
                # actual_let = get_key() # get key press from console
                # print('actual letter', actual_let)
                cur_let = coords_to_letter(self.current_coords)
                goal_let = self.goal_letter
                if cur_let == goal_let:
                    reward = 1
                else:
                    reward = 0
            else:
                reward = 0
                prev_state = self.current_coords
                self.move_dobot(action)
                if np.array_equal(prev_state, self.current_coords, equal_nan=False):
                    reward = 0  # potential punishment for not leaving the key

        return self.state, reward, done, info

    def render(self):
        pass

class her():
    def __init__(self):
        self.her_buffer = []

    def update_her_buffer(self, transition):
        self.her_buffer.append(transition)

    def sample(self, index):
        return self.her_buffer[index]

    def update_transition(self, index, new_goal, max_steps):
        current_state, action, reward, new_state, done = self.sample(index)
        temp_current_state = current_state.copy()
        temp_new_state = new_state.copy()
        temp_current_state['vec'] = new_goal
        temp_new_state['vec'] = new_goal

        if index == max_steps-1:
            new_reward = 1
        else:
            new_reward = 0

        return (temp_current_state, action, new_reward, temp_new_state, done)

class phys_discrete_alphabet_env_new(Env):
    def __init__(self, current_letter, goal_letter):
        self.current_step = 0
        self.starting_letter = current_letter
        self.goal_letter = goal_letter
        self.current_coords = letter_to_coords(self.starting_letter)
        move_phys_dobot(self.current_coords)
        self.action_array = np.array(['upleft', 'upright', 'left', 'right', 'downleft', 'downright', 'pressdown'])
        self.state = create_state(get_image(self.current_coords), self.goal_letter)
        self.img_shape = np.shape(get_image(self.current_coords))
        self.max_ep_len = 16
        self.action_space_size = len(self.action_array)

    def reset(self, current_letter, goal_letter):
        self.current_step = 0
        self.starting_letter = current_letter
        self.goal_letter = goal_letter
        self.current_coords = letter_to_coords(self.starting_letter)
        move_phys_dobot(self.current_coords)
        self.state = create_state(get_image(self.current_coords), self.goal_letter)

        return self.state

    def move_dobot(self, action):
        self.current_coords = translate_coord(self.current_coords) #turn coords into imaginary coords

        if action == 'upleft':
            if self.current_coords[0] == 3 and self.current_coords[1] == 14:
                self.current_coords = np.array([2, 11, -25])
            elif self.current_coords[0] > 0 and self.current_coords[1] > 0:
                self.current_coords = self.current_coords + np.array([-1, -1, 0])

        elif action == 'upright':
            if self.current_coords[0] == 3 and self.current_coords[1] == 14:
                self.current_coords = np.array([2, 14, -25])
            elif self.current_coords[0] > 0:
                self.current_coords = self.current_coords + np.array([-1, 2, 0])

        elif action == 'left':
            if self.current_coords[0] < 3 and self.current_coords[1] > 2:
                self.current_coords = self.current_coords + np.array([0, -3, 0])

        elif action == 'right':
            if self.current_coords[0] < 2 and self.current_coords[1] < 25:
                self.current_coords = self.current_coords + np.array([0, 3, 0])
            elif self.current_coords[0] < 3 and self.current_coords[1] < 18:
                self.current_coords = self.current_coords + np.array([0, 3, 0])

        elif action == 'downleft':
            if self.current_coords[0] == 1 and self.current_coords[1] > 24:
                pass
            elif self.current_coords[0] < 2 and self.current_coords[1] > 2:
                self.current_coords = self.current_coords + np.array([1, -2, 0])
            elif self.current_coords[0] == 2 and self.current_coords[1] > 5: # bottom row not z or x
                self.current_coords = np.array([3, 14, -25])

        elif action == 'downright':
            if self.current_coords[0] == 1 and self.current_coords[1] > 19:
                pass
            elif self.current_coords[0] < 2 and self.current_coords[1] < 25:
                self.current_coords = self.current_coords + np.array([1, 1, 0])
            elif self.current_coords[0] == 2 and self.current_coords[1] > 2: # bottom row not z
                self.current_coords = np.array([3, 14, -25])

        else:
            print('Error: Not a valid action')
            exit()
        self.current_coords = translate_coord(self.current_coords) #turn coords into real
        move_phys_dobot(self.current_coords) #move robot to new coords
        current_img = get_image(self.current_coords) #get new image of state
        self.state = create_state(current_img, self.goal_letter)

    def step(self, action, steps):
        done = False
        info = []
        if steps > self.max_ep_len -1:
            reward = 0
        else:
            action = self.action_array[action]
            if action == 'pressdown':
                done = True
                # press_phys_dobot(self.current_coords) # move dobot to press key
                # actual_let = get_key() # get key press from console
                # print('actual letter', actual_let)
                cur_let = coords_to_letter(self.current_coords)
                goal_let = self.goal_letter
                if cur_let == goal_let:
                    reward = 1
                else:
                    reward = 0
            else:
                reward = 0
                prev_state = self.current_coords
                self.move_dobot(action)
                if np.array_equal(prev_state, self.current_coords, equal_nan=False):
                    reward = 0  # potential punishment for not leaving the key

        return self.state, reward, done, info

    def render(self):
        pass
