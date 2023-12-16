#________________________________________RECORDING HUMAN PLAY____________________________________________________________________
import cv2
import numpy as np
#____________________________________________________________PREPROCESS_________________________________________________________________________________
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os
import time
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode='human', apply_api_compatibility=True)

env = JoypadSpace(env, [["right"], ["right", "A"],['NOOP']])

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

#do we need this ?
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)

#_____________________________________LET HUMAN PLAY , LOAD AGENT AND STORE HUMAN ACTIONS, AGENT ACTIONS AND ACTION VALUES OF STATES _______________________________________________________________

class MarioNet(nn.Module):
    """mini CNN structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
#___________________________________________________________
device = "cuda" if torch.cuda.is_available() else "cpu"
state_dim = (4, 84, 84) 
action_dim = 2
net = MarioNet(state_dim, action_dim).float()
net = net.to(device=device)
#___________________________________________________________
state_dict = torch.load(r".\Checkpoint\mario_net_14.chkpt")
net.load_state_dict(state_dict['model'])
#__________________________________________________________

def act(model,state) :    
    state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
    state = torch.tensor(state, device=device).unsqueeze(0)
    action_values = net(state, model="online")
    action_idx = torch.argmax(action_values, axis=1).item()
    
    return action_idx

#_________________________________________________________
# Create an empty list to store action values
action_values_list = []
action_agent_list=[]
#_________________________________________________________
# Create a flag - restart or not
done = True

import keyboard


def get_action():
    
    if keyboard.is_pressed("6"):
        if keyboard.is_pressed("8"):
            return 1
        return 0
    elif keyboard.is_pressed("8"):
        return 1
    else:
        return 2
Human_act=[]
human_state=[]
initial=[]
# Loop through each frame in the game
# Loop through each episode
episode=False
state = env.reset()
pos=[]
while not episode:
    
    next_state, reward, done, trunc, info = env.step(get_action())
    
    if get_action() != 2:
        Human_act.append(get_action())
        human_state.append(state)
        #__________________________________________________
        stateout = state
        stateout = stateout[0].__array__() if isinstance(stateout, tuple) else stateout.__array__()
        stateout = torch.tensor(stateout, device=device).unsqueeze(0)
        action_values = net(stateout, model="online")
        values = action_values.tolist()
        action_values_list.append(values)
        action_idx = torch.argmax(action_values[-1]).item()
        action_agent_list.append(action_idx)
        #print(values[-1])
        pos.append(info["x_pos"])
    else:
        initial.append(get_action())
        #_______________________________________________
    # Show the game on the screen
    state = next_state # 
    env.render()

    #____________________________________________________________________________________
    # Calculate delay based on the environment's frame rate
    frame_rate = env.metadata['video.frames_per_second']
    delay = 1.0 / frame_rate if frame_rate > 0 else 0.01
    
    # Sleep to control the frame rate
    time.sleep(delay)
    #___________________________________________________   
    # Check if the episode is done
    if info["life"]<2:
        print(info["life"])
        episode=True

# Close the game
env.close()

#_________________________________Analysing VARIABLES_________________________________________
# important states_______________________________________________________
threshold=1
imp_states=[]
mistakes=[]
well=[]
for i in range(len(action_values_list)) :
    if abs(action_values_list[i][0][0]-action_values_list[i][0][1])> threshold :
        imp_states.append(action_values_list[i][0])
        
        if action_agent_list[i] == Human_act[i] :
            
            print("in states:", i, "The human played well")
            well.append(i)
            
        else:
            
            print("in states", i, "The human has made a mistake")
            mistakes.append(i)
            
    else:
        imp_states.append(0)

#_______________________________________ CALCULATING LOSS______________________________
count=[]
for i in imp_states:
    if not( i==0 ):
           count.append(i)     
well_loss= (len(well)/len(count))*100
print("The human playing well acuracy is ", well_loss )
mistakes_loss= (len(mistakes)/len(count))*100
print("The human making mistake acuracy is  ", mistakes_loss)

print("Mario has gone this far: ", np.max(pos))