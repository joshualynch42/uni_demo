import keyboard
import pygame
import time
import sys
from dueling_ddqn_per import *
from phys_utils import *

def add_text(str, x_coord, y_coord):
    text = font.render(str, True, green, blue)
    textRect = text.get_rect()
    textRect.center = (x_coord, y_coord)
    display_surface.blit(text, textRect)

# PYGAME SETUP #
pygame.init()
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
X = 800
Y = 400
display_surface = pygame.display.set_mode((X, Y))
pygame.display.set_caption('Show Text')
font = pygame.font.Font('freesansbold.ttf', 32)

# Robot agent #
rl_params = {
'replay_memory_size': 100,
'minibatch_size': 16,
'epsilon_decay': 0, # for alphabet
'discount': 0,
'min_replay_memory_size': 0,
'min_epsilon': 0,
'epsilon': 0,
'update_target_every': 1,
'episodes': 1
}
env = phys_discrete_alphabet_env_new('Q', 'Q')
model_dir = "D:/Users/Josh/github/uni_demo/alphabet_transfer_learning_phys_Dueling Double Per_new.h5"
agent = Dueling_Per_DDQNAgent(env, rl_params)
agent.load_model(model_dir)

# Robot to press key #
def robot_press(env, agent, goal_letter):
    current_state = env.reset(coords_to_letter(env.current_coords), goal_letter)
    done = False
    steps = 0
    while not done and steps < env.max_ep_len:
        steps += 1
        done = move_robot_coords(env, goal_letter)
    return coords_to_letter(env.current_coords)

def move_robot_coords(env, goal_letter):
    current_imaginary = translate_coord(env.current_coords)
    goal_imaginary = translate_coord(letter_to_coords(goal_letter))
    done = False
    if current_imaginary[1] < goal_imaginary[1] - 2:
        if current_imaginary[1] == 20:
            current_imaginary[0] -= 1
            current_imaginary[1] += 5
        else:
            current_imaginary[1] += 3
    elif current_imaginary[1] > goal_imaginary[1] + 2:
        current_imaginary[1] -= 3
    elif current_imaginary[0] < goal_imaginary[0]:
        if current_imaginary[1] == 25:
            current_imaginary[0] += 1
            current_imaginary[1] -= 5
        elif current_imaginary[1] == 22:
            current_imaginary[0] += 1
            current_imaginary[1] -= 2
        elif current_imaginary[1] == 27:
            current_imaginary[0] += 1
            current_imaginary[1] -= 2
        else:
            current_imaginary[0] += 1
            current_imaginary[1] += 1
    elif current_imaginary[0] > goal_imaginary[0]:
        current_imaginary[0] -= 1
        current_imaginary[1] -= 1
    else:
        done = True
    env.current_coords = translate_coord(current_imaginary) #turn coords into real
    move_phys_dobot(env.current_coords) #move robot to new coords
    if done == False:
        press_phys_dobot(env.current_coords)

    return done

# DISPLAY INITIAL QUESTION #
display_surface.fill(white)
add_text('What name would you like to use?', X // 2, Y // 4)
pygame.display.update()

## --- FIRST LOOP ------ ##
# USER ENTERS NAME TO TYPE #
type_str = ''
event_counter = 0
done = False
while not done:
    event_counter += 1
    # event is the key pressed by the robot
    event = keyboard.read_key()
    # each key press produces 2 events. counter used to stop this
    if event_counter % 2 == 0:
        # If the users enters the name
        if event == 'enter':
            done = True
            break
        # swap word space for ' '
        if event == 'space':
            event = ' '
        # reset event counter
        event_counter = 0
        # add new letter (event) to string for display
        type_str = type_str + event

    ### DISPLAY STRING ###
    display_surface.fill(white)
    add_text('What name would you like to use?', X // 2, Y // 4)
    add_text(type_str, X // 2, Y // 2)
    add_text('Press ENTER to when done', X // 2, Y // 4 * 3)
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

## --- THE USER HAS PRESSED ENTER --- ##
display_surface.fill(white)
add_text('The Dobot is now typing out the name', X // 2, Y // 4)
pygame.display.update()

## --- SECOND LOOP --- ##
# THE ROBOT IS TYPING OUT THE NAME #
final_str = ''
event_counter = 0
len_str = 0
while len_str < len(type_str):
    # act robot #
    goal_letter = [char for char in type_str][len_str]
    robot_press(env, agent, goal_letter.upper())

    event = coords_to_letter(env.current_coords)

    if event == 'SPACE':
        event = ' '
    len_str += 1
    final_str = final_str + event

    ### DISPLAY STRING ###
    display_surface.fill(white)
    add_text('The Dobot is now typing out the name', X // 2, Y // 4)
    add_text(final_str, X // 2, Y // 2)
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

### DISPLAY STRING ###
display_surface.fill(white)
add_text('The Dobot has finished', X // 2, Y // 4)
add_text(final_str, X // 2, Y // 2)
add_text('Press ENTER to exit', X // 2, Y // 4 * 3)
pygame.display.update()

while True:
    ### DISPLAY STRING ###
    display_surface.fill(white)
    add_text('The Dobot has finished', X // 2, Y // 4)
    add_text(final_str, X // 2, Y // 2)
    add_text('Press ENTER to exit', X // 2, Y // 4 * 3)
    pygame.display.update()

    event = keyboard.read_key()
    if event == 'enter':
        pygame.quit()
        quit()
