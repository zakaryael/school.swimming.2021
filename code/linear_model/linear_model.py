
from pandas.core.frame import DataFrame
from numpy import linalg
import swimmer_actions_states_cfg as cfg
import os
import argparse
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
import csv
import json

from  tiles3 import *

parser = argparse.ArgumentParser(description='Compute the .cfg, .json, .geo and preconditioner files for different values of height of the swimmer simulations.')
parser.add_argument("--Dim",help="The dimension",type=int,default=2)
args = parser.parse_args()

#GENERATE THE FLUID SIMULATION

def generate_simulation(state, action) :
    Wall_position = 20
    os.makedirs('sw/q-learning/three_sphere_swimmer', exist_ok=True)    
    cfg.write_cfg('sw/q-learning/three_sphere_swimmer',args.Dim)
    cfg.write_json(action,'sw/q-learning/three_sphere_swimmer',args.Dim)
    cfg.write_geo(state,'sw/q-learning/three_sphere_swimmer',args.Dim)
    cfg.write_preconditioner('sw/q-learning/three_sphere_swimmer',args.Dim)
    os.system('/home/lberti/feelpp_Ecole/feelpp/build-Clang12-cpp17-toolboxes-nomor-release-nosws-nodoxygen-nomatplot/toolboxes/fluid/feelpp_toolbox_fluid --config-file sw/q-learning/three_sphere_swimmer/three_sphere_swimmer.cfg ')
    data = pd.read_csv("/home/lberti/feel/toolboxes/fluid/moving_body/q-learning/three_sphere_swimmer/np_1/fluid.measures.csv")
    data.columns = data.columns.str.strip()
    l1 = list(data['Quantities_body_CircleCenter.mass_center_0'])
    l2 = list(data['Quantities_body_CircleCenter.mass_center_1'])

    lL1 = list(data['Quantities_body_CircleLeft.mass_center_0'])
    lL2 = list(data['Quantities_body_CircleLeft.mass_center_1'])
    
    lR1 = list(data['Quantities_body_CircleRight.mass_center_0'])
    lR2 = list(data['Quantities_body_CircleRight.mass_center_1'])
    
    reward = l1[len(l1)-1]
    x_cm2 = l1[len(l1)-1]
    y_cm = Wall_position - l2[len(l2)-1] # distance from the wall

    if action=="retract_left_arm":
        Length1 = 6
        Length2 = state[3]
    if action=="retract_right_arm": 
        Length2 = 6
        Length1 = state[2]
    if action=="extend_left_arm":
        Length1 = 10
        Length2 = state[3]
    if action=="extend_right_arm": 
        Length2 = 10
        Length1 = state[2]

    # orientation with respect to the x direction
    theta = np.arctan2((l2[len(l2)-1]-lL2[len(lL2)-1]),(l1[len(l1)-1]-lL1[len(lL1)-1]))
    #MODIF
    #theta =theta if theta>0 else theta+2*np.pi


    new_state = [y_cm,theta,Length1,Length2]
    return reward,x_cm2,new_state

def get_key(val, dict):
    for key, value in dict.items():
         if val == value:
             return key


def get_index_of_possible_actions(state):
    indexes = []
    for action in possible_actions(state):
        indexes.append(get_key(action, actions_space))
    return indexes

# state_space [-10,10]
# action space [-10,10]

# Names of actions

action1 = "retract_left_arm"
action2 = "extend_left_arm"
action3 = "retract_right_arm"
action4 = "extend_right_arm"

# Dictionary of actions taking as keys numbers which are indexes of these actions
# that will be used for Q-table columns

actions_space = {0 : action1, 1 : action2, 2 : action3, 3 : action4}

def possible_actions(state):
    action_space = []
    if np.round(state[2])==10:
        action_space.append("retract_left_arm")
    if np.round(state[3])==10:
        action_space.append("retract_right_arm")
    if np.round(state[2])==6:
        action_space.append("extend_left_arm")
    if np.round(state[3])==6:
        action_space.append("extend_right_arm") 
    return action_space

# def possible_actions(state):
#     if state[0] and state[1]:
#         actions = ["retract_left_arm", "retract_right_arm"]
#     elif state[0] and not state[1]:
#         actions = ["retract_left_arm", "extend_right_arm"]
#     elif not state[0] and state[1]:
#         actions = ["extend_left_arm", "retract_right_arm"]
#     elif not state[0] and not state[1]:
#         actions = ["extend_left_arm", "extend_right_arm"]
#     return actions
# Parameters

N_max = 10 #number of episodes
T_swimming_period = 4
N_tilings = 1
alpha = 0.8#0.1/N_tilings #MODIF
N_swimming_cycles = 4

D_max = 10 #max distance from wall

iht = IHT(16)
iht_size = iht.size
weights = np.zeros(iht_size*len(actions_space))

np.savetxt('weights.csv', weights, delimiter=',')

def q_function(state,action,w):
    return np.sum(w[my_tiles(state)+action*iht_size])

# Function adjusting the tiling bounds for feature computation

def my_tiles(state):
    scale_factor_y2=10/D_max
    scale_factor_theta=10/np.pi
    scale_factor_length=10/D_max
    return np.array(tiles(iht,N_tilings,list([state[0]*scale_factor_y2,state[1]*scale_factor_theta]),list([state[2]*scale_factor_length,state[3]*scale_factor_length])))



# State (y_2,\theta,L1,L2)
W =np.zeros((T_swimming_period*N_max,iht_size*len(actions_space)))
columns = ['Episode','Timestep','State','Action','Reward']
Data = pd.DataFrame(columns=columns)
for episode in range(0, N_max):
    print('-------Episode',episode,'/',N_max,'------------')
    y2 =20# random.uniform(0,D_max)
    theta = 0#random.uniform(0,2*np.pi)
    L1 = random.choice([6,10])
    L2 = random.choice([6,10])
    state=[y2,theta,L1,L2]
    action = random.choice(possible_actions(state))
    action_index = get_key(action, actions_space)
    x_cm_init = 0
    cumulative_xcm2=0
    for iteration in range(T_swimming_period+1):
        W[iteration]=weights
        if iteration > 1:
            print('--------- Weights W norm ---',np.linalg.norm(np.array(W)[iteration-1]))
        print('--------- Weights W norm ---',np.linalg.norm(np.array(W)[iteration]))
        print('--------- Weights norm ---',np.linalg.norm(np.array(weights)))
        # If the state is terminal
        new_data = pd.DataFrame([[episode,iteration,state,action,cumulative_xcm2]],columns=columns)
        Data = pd.concat([Data, new_data], ignore_index = True)
        print(state)
        reward,x_cm2,new_state = generate_simulation(state,action)
        print('------Results simulation ------',reward,x_cm2,new_state)
        cumulative_xcm2 += x_cm2 # x_cm2 is the ouput when starting from 0 
        print('---------Cumulative displacement --------', cumulative_xcm2)
        if (iteration==T_swimming_period): # The swimmer must move to the right
            for tile in my_tiles(state):
                weights[tile+iht_size*action_index] += alpha*(reward-q_function(state,action_index,weights))
            new_data = pd.DataFrame([[episode,iteration,new_state,"terminal_state",cumulative_xcm2]],columns=columns)
            Data = pd.concat([Data, new_data], ignore_index = True)
            break
        # Choose policy epsilon-greedily
        if random.uniform(0,1) < 0.3 : #0.1 MODIF
            new_action = random.choice(possible_actions(new_state)) # Here we choose a random action from the possible actions!
            print('------Results simulation with random action------',reward,x_cm2,new_state,new_action)
            new_action_index = get_key(new_action, actions_space)   # Get the index of the action to be used to update the Q-function approximation
        else :                              # Here, the action that maximizes the Q-function will be chosen
            possible_action_indexes = get_index_of_possible_actions(new_state)   # Get indexes of possible actions so we can access their Q-table values to choose the maximum
            print('-------possible actions------',possible_action_indexes)
            max_q = -100 # q_function(new_state,0,weights)
            for i in get_index_of_possible_actions(new_state):
                q_value = q_function(new_state,i,weights)
                print('----',q_value,'----',actions_space[i])
                if q_value >=max_q:
                    max_q = q_value
                    new_action_index = i
            new_action = actions_space[new_action_index] 
            print('------Results simulation with new action------',reward,x_cm2,new_state,new_action)
        print('----------Error------------ ',reward+q_function(new_state,new_action_index,weights)-q_function(state,action_index,weights))
        for tile in my_tiles(state):
            print('------Weight------',weights[tile+iht_size*action_index])
            weights[tile+iht_size*action_index] += alpha*(reward+q_function(new_state,new_action_index,weights)-q_function(state,action_index,weights))
        state=new_state
        action=new_action
        action_index = new_action_index        
        # new_data = pd.DataFrame([[episode,iteration,state,action,cumulative_xcm2]],columns=columns)
        # Data = pd.concat([Data, new_data], ignore_index = True)
    Data.to_csv('Simulation_data.csv')
    np.savetxt('weights.csv', np.array(W), delimiter=',')

W = np.array(W)     
Data.to_csv('Simulation_data.csv')
#Weights
print('Save the weights')
np.savetxt('weights.csv', W, delimiter=',')
#Save the results of the model
#Output an example of optimal policy by maximizing the Q-function approximation
print('Print the optimal policy for horizontal swimmer far from walls')
columns = ['State','New action index','Action','Reward']
Data_policy = pd.DataFrame(columns=columns)
state =[20,0,10,10]
new_action_index=0
results=[state,0,actions_space[new_action_index],0]
for j in range(15):
    max_q = -100 # q_function(state,new_action_index,weights)
    for i in get_index_of_possible_actions(state):
        q_value = q_function(state,i,weights)
        print('----',q_value,'----',actions_space[i])
        if q_value >=max_q:
            max_q = q_value
            new_action_index = i
    action = actions_space[new_action_index]
    reward,x_cm2,new_state = generate_simulation(state,action)
    new_data_policy = pd.DataFrame([[state,new_action_index,actions_space[new_action_index],reward]],columns=columns)
    Data_policy = pd.concat([Data_policy, new_data_policy], ignore_index = True)
    state=new_state

print(Data_policy)

Data_policy.to_csv('optimal_policy.csv')

for i in range(4):
    print('---- Q function at [0,0,10,10] ---- Action ',actions_space[i])
    print(q_function([20,0,10,10],i,weights))
    print('---- Q function at [0,0,10,6] ---- Action ',actions_space[i])
    print(q_function([20,0,10,6],i,weights))
    print('---- Q function at [0,0,6,10] ---- Action ',actions_space[i])
    print(q_function([20,0,6,10],i,weights))
    print('---- Q function at [0,0,6,6] ---- Action ',actions_space[i])
    print(q_function([20,0,6,6],i,weights))

