import sys
import feelpp as feelpp
import feelpp.quality as q
import feelpp.toolboxes.core as tb
import feelpp.interpolation as I
from feelpp.toolboxes.fluid import *

def remesh_toolbox(f, hclose, hfar, parent_mesh):
    # 3 spheres
    required_facets=["CircleLeft","CircleCenter","CircleRight"]
    required_elts=["CirLeft","CirCenter","CirRight"]
    Xh = feelpp.functionSpace(mesh=f.mesh())
    n_required_elts_before=feelpp.nelements(feelpp.markedelements(f.mesh(),required_elts))
    n_required_facets_before=feelpp.nfaces(feelpp.markedfaces(f.mesh(),required_facets))
    print(" . [before remesh]   n required elts: {}".format(n_required_elts_before))
    print(" . [before remesh] n required facets: {}".format(n_required_facets_before))
#    metric=feelpp.
    new_mesh,cpt = feelpp.remesh(f.mesh(), "gradedls({},{})".format(hclose, hfar), required_elts, required_facets, None )
    print(" . [after remesh]  n remeshes: {}".format(cpt))
    n_required_elts_after=feelpp.nelements(feelpp.markedelements(new_mesh,required_elts))
    n_required_facets_after=feelpp.nfaces(feelpp.markedfaces(new_mesh,required_facets))
    print(" . [after remesh]  n required elts: {}".format(n_required_elts_after))
    print(" . [after remesh] n required facets: {}".format(n_required_facets_after))
    f.applyRemesh(new_mesh)

def solve_fluid_problem():

    sys.argv = ['fluid-remesh']
    e = feelpp.Environment(
        sys.argv, opts=tb.toolboxes_options("fluid"),
        config=feelpp.globalRepository("fluid-qlearning"))
    
    #feelpp.Environment.setConfigFile('cases/swimmers/3-sphere/2d/three_sphere_2D.cfg')
    feelpp.Environment.setConfigFile('sw/q-learning/three_sphere_swimmer/three_sphere_swimmer.cfg')

    f = fluid(dim=2, orderVelocity=2, orderPressure=1)
    f.init()


    # 3 spheres
    hfar = 1
    hclose = 0.002


    parent_mesh=f.mesh()
    #remesh_toolbox(f, hclose, hfar, None )
    #remesh_toolbox(f, hclose, hfar, None)
    #f.exportResults()
    f.startTimeStep()
    while not f.timeStepBase().isFinished():
        min_etaq = q.etaQ(f.mesh()).min()
        if min_etaq < 0.1:
    #    if f.timeStepBase().iteration() % 10 == 0:
            remesh_toolbox(f, hclose, hfar, None)
        if feelpp.Environment.isMasterRank():
            print("============================================================\n")
            print("time simulation: {}s iteration : {}\n".format(f.time(), f.timeStepBase().iteration()))
            print("  -- mesh quality: {}s\n".format(min_etaq))
            print("============================================================\n")
        for i in range(4):
            f.solve()
        f.exportResults()
        
        f.updateTimeStep()
    
    return true

# ----------------------------------------------------------
# ----------------------------------------------------------
# ----------------------------------------------------------
# ----------------------------------------------------------
# ----------------------------------------------------------

# Q-learning

##³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³ Importing libraries ³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³##

import swimmer_actions_states_cfg as cfg
import os
import argparse
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
import csv
import json

parser = argparse.ArgumentParser(description='Compute the .cfg, .json, .geo and preconditioner files for different values of height of the swimmer simulations.')
parser.add_argument("--Dim",help="The dimension",type=int,default=2)
args = parser.parse_args()


##³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³ Defining Functions ³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³##

# This function generates simulations for a given action and state, reads the csv file
# and returns the reward. Note that the swimmer is located at the center (0,0) for each
# simulation, so the reward is the center of mass 0 of the center sphere.

def generate_simulation(state, action) :
    os.makedirs('sw/q-learning/three_sphere_swimmer', exist_ok=True)    
    cfg.write_cfg('sw/q-learning/three_sphere_swimmer',args.Dim)
    cfg.write_json(action,'sw/q-learning/three_sphere_swimmer',args.Dim)
    cfg.write_geo(state,'sw/q-learning/three_sphere_swimmer',args.Dim)
    cfg.write_preconditioner('../sw/q-learning/three_sphere_swimmer',args.Dim)
    # ------- WORKING LINE
    os.system('feelpp_toolbox_fluid --config-file sw/q-learning/three_sphere_swimmer/three_sphere_swimmer.cfg ')
    # ------- THIS STOPS THE Q-LEARNING AFTER ONE RUN OF THE FLUID TOOLBOX ONLY
    #state = solve_fluid_problem()

    #data = pd.read_csv("/home/lberti/fluid-qlearning/np_1/fluid.measures.csv")
    data = pd.read_csv("/home/lberti/feel/toolboxes/fluid/moving_body/q-learning/three_sphere_swimmer/np_1/fluid.measures.csv")
    data.columns = data.columns.str.strip()
    l1 = list(data['Quantities_body_CircleCenter.mass_center_0'])
    reward = l1[len(l1)-1]
    #New_State = new_state(state, action)
    return reward#, New_State

# This function takes for argument a state and an action, and returns the new state
# of the swimmer after taking the action in that state.

def new_state(state, action):
    if state[0] and action == "retract_left_arm" :
        s = [False, state[1]]
    elif not state[0] and action == "extend_left_arm" :
        s = [True, state[1]]
    elif state[1] and action == "retract_right_arm" :
        s = [state[0], False]
    elif not state[1] and action == "extend_right_arm" :
        s = [state[0], True]
    #else :
    #    s = "This action can not be taken at this state"
    return s

# This function takes for argument a state and return the possible action the swimmer
# can perform. 

def possible_actions(state):
    if state[0] and state[1]:
        actions = ["retract_left_arm", "retract_right_arm"]
    elif state[0] and not state[1]:
        actions = ["retract_left_arm", "extend_right_arm"]
    elif not state[0] and state[1]:
        actions = ["extend_left_arm", "retract_right_arm"]
    elif not state[0] and not state[1]:
        actions = ["extend_left_arm", "extend_right_arm"]
    return actions

# This function takes as arguments a dictionary and a value from it and returns the
# key of this value. Actually we are working with dictionaries for the actions and
# the states

def get_key(val, dict):
    for key, value in dict.items():
         if val == value:
             return key

# This function will be needed to access the keys of the possible actions. It is
# defined here to make the part of the Q-learning algorithm short and abvious.

def get_index_of_possible_actions(state):
    indexes = []
    for action in possible_actions(state):
        indexes.append(get_key(action, actions_space))
    return indexes

# This function will be used to determine the argmax of the Q-table. The Q-table 
# has 4 lines and 4 columns but the argmax should be over the possible actions and
# not all the actions. Hence, the numpy.argmax can't be used here since we take the
# argmax of specific values (possible actions). At each state, there are only two
# possible actions so we put their indexes here with the Q_table (to access the
# Q-table values) and this function returns the index of the action maximizing the
# Q-table.

def get_max_Q_table_index(indexes, state_index, Q_table):
    if Q_table[state_index, indexes[0]] >= Q_table[state_index, indexes[1]] :
        index_max = indexes[0]
    else :
        index_max = indexes[1]
    return index_max


##³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³ Defining states and actions ³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³##


# Names of actions

action1 = "retract_left_arm"
action2 = "extend_left_arm"
action3 = "retract_right_arm"
action4 = "extend_right_arm"

# Dictionary of actions taking as keys numbers which are indexes of these actions
# that will be used for Q-table columns

actions_space = {0 : action1, 1 : action2, 2 : action3, 3 : action4}

# Names of states
# Here, there are 4 states, each state is a list of two components, each component
# is a boolien (True or Fals) describing the state of an arm. the first component is
# for the left arm and the second component is for the right arm, with the CONVENTION
# "False" means the arm is short (already retrated) and "True" means the arm is long.

state1 = [True, True]
state2 = [True, False]
state3 = [False, True]
state4 = [False, False]

# Dictionary of states taking as keys numbers which are indexes of these states
# that will be used for Q-table lines

states_space = {0 : state1, 1 : state2, 2 : state3, 3 : state4}


##³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³ Q-learning algorithm ³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³##


n = m = 4                   # The Q-table dimensions (4 actions and 4 states)
alpha = 1                 # The learning rate for Q-learning algorithm
gamma = 0.95                 # Discount factor for Q-learning algorithm
eps   = 0.05                 # Epsilon-greedy scheme constant
N_max = 3000                 # The maximum number of learning steps

state = state2              # The initial state of the swimmer will be state1=[True, True]
state = random.choice([states_space[i] for i in range(0, 4)])
cumul_reward = 0            # Cumulative reward initialisation
cumul_reward_list = []      # List that will contain cumulative reward of each learning step to be plotted at the end
actions_list = []           # List that will contain the actions chosen over the learning process
states_list = [state]       # List that will contain the states chosen over the learning process, starting from "state1" we have chosen
Q_table = np.zeros((n, m))  # Q-table initialisation by zeros
stored_results = np.zeros((n, m))   # Matrix of dimensions 4*4 to stores results of simulations to avoid repeating them!
# As this matrix is initialised with zeros, the reward of a simulation (state, action) will be put in the state_indes
# line and the action index column.

#stored_results = np.array([[-1.40, 0, 1.40, 0], [-1.53, 0, 0, -1.40], [0, 1.40, 1.53, 0], [0,  1.53, 0, -1.53]])
# This matrix contains the simulations results to avoid running them

for i in range(0, N_max):
    #if i < 10:
    #    eps = 0.5
    #else :
    #    eps = 0.05
    state_index = get_key(state, states_space)     # Get the index of the state to be used to access Q-table
    # Epsilon-greedy scheme
    if random.uniform(0,1) < eps : 
        action = random.choice(possible_actions(state)) # Here we choose a random action from the possible actions!
        action_index = get_key(action, actions_space)   # Get the index of the action to be used to access Q-table
    else :                              # Here, the action that maximizes the Q-table will be chosen
        possible_action_indexes = get_index_of_possible_actions(state)   # Get indexes of possible actions so we can access their Q-table values to choose the maximum
        action_index = get_max_Q_table_index(possible_action_indexes, state_index, Q_table) # the function returns the index of the action maximizing the Q-table 
        action = actions_space[action_index]   # here, we return the action's name (it will be used to generate the simulation as the function (generating simulations) takes for argument the name and not the index)
    print("For i =", i, ":")
    print("state = ", state)
    print("action = ", action)
    if stored_results[state_index, action_index] == 0:     # if it is <0, it means we don't have results for this (state, action) situation. Hence, we need to perform the simulation
        reward = generate_simulation(state, action)        # Generate the simulation and compute the reward
        New_state = new_state(state, action)               # The new state the swimmer moves to, after taking the action in the old state
        stored_results[state_index, action_index] = reward # Since we performed a new simulation, we store the reward so we don't need to perform it again
    else :                                                 # Here (if =0) means the simulation for this case (state, action) has been performed and the reward has been stored in the matrix!
        reward = stored_results[state_index, action_index] # The reward has already been stored in the matrix
        New_state = new_state(state, action)               # The new state 
    #print("New_state = ", New_state)
    print("reward = ", reward)
    New_state_index = get_key(New_state, states_space)     # get the index of the state to be used to access Q-table
    New_indexes = get_index_of_possible_actions(New_state)
    max_value = max(Q_table[New_state_index, New_indexes[0]], Q_table[New_state_index, New_indexes[1]])
    # The Q-learning formula
    Q_table[state_index, action_index] = Q_table[state_index, action_index] + alpha*(reward + gamma*max_value - Q_table[state_index, action_index])
    print(Q_table[New_state_index, :])
    print("max_Q_table = ", max_value)
    cumul_reward += reward                  # Compute the cumulative reward
    print("cumul reward = ", cumul_reward)
    state = New_state                       # The state of the swimmer changes to the new one
    cumul_reward_list.append(cumul_reward)  # Add the cumulative reward to the list of cumulative rewards
    actions_list.append(action)             # Add the action to the list of actions
    states_list.append(New_state)           # Add the state to the list of states

##³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³ Export the results in a csv file for cumulative reward ³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³##

with open(os.path.join(os.path.dirname(__file__), 'cumulative_reward_qlearning.csv'), 'w', newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Iteration','Cumulative reward'])
    for i in range(0, N_max):
        writer.writerow([i,cumul_reward_list[i]])

##³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³ Export the results in a txt file for swimming strategy ³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³##

results_file= open(os.path.join(os.path.dirname(__file__), 'Results_of_learning.txt'), 'w')
results_file.write("-------States-------\n")
results_file.writelines(json.dumps(states_space))
results_file.write("\n-------Actions-------\n")
results_file.writelines(json.dumps(actions_space))
results_file.write("\n-------Q_matrix-------\n")
for line in Q_table:
    np.savetxt(results_file, line, fmt='%.2f')
results_file.write("\n------- Swimming strategy from maximization of Q_matrix - long version -------\n")
optimal_policy=[]
state=random.choice(states_list)
for i in range(n*m):
    state_index = get_key(state, states_space) 
    possible_action_indexes = get_index_of_possible_actions(state) 
    action_index = get_max_Q_table_index(possible_action_indexes, state_index, Q_table)
    action = actions_space[action_index]
    state_new=new_state(state, action)
    optimal_policy.append(state_new)
    state=state_new
for element in optimal_policy:
    for elements in element:
        results_file.write(str(elements)+" ")
    results_file.write("\n")

# results_file.write("\n------- Swimming strategy from maximization of Q_matrix - shorter version - max cycle length -------\n")
# shorter_policy=[]
# state_key=[]
# sequence=[]
# for state in optimal_policy:
#     state_key.append(get_key(state, states_space))
# sequence = find_max_seq(state_key)
# print(sequence)
# shorter_policy = list(set(tuple(l) for l in sequence))
# print (shorter_policy)

results_file.close()
##³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³ Display results ³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³##


print(len(np.arange(0, N_max))) 
print(len(cumul_reward_list))   
fig =plt.figure()
plt.plot(list(np.arange(0, N_max)), cumul_reward_list)
#plt.title("Q-learning")
#plt.xlabel("Iterations")
#plt.ylabel("Cumulative reward")
#plt.grid()
fig.savefig(os.path.join(os.path.dirname(__file__),'Cumulative_reward.png'))
plt.show()


