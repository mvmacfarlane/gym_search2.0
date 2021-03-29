import gym
from gym import error, spaces, utils
from gym.utils import seeding
from copy import deepcopy
import numpy as np
import torch
from networkx.classes.function import all_neighbors
import time
import os
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import dgl
import dgl.function as fn


class SearchEnv(gym.Env):
  metadata = {'render.modes': ['human']}


  def __init__(self,environment,search_budget,reward_shape,MCTS=False,test=False):

    #long term variables
    self.done_overall = False
    self.initial_state = environment.reset(render_mode='raw')
    self.sub_env_storage = deepcopy(environment)
    self.reward_shape = reward_shape
    self.test = test
    self.file_location = '/home/mvmacfarlane/Documents/PhD/PhD/graphs/'
    self.search_budget = search_budget

    #I want this notion to be change to like stop added at leaf
    self.MCTS = MCTS

    #short term variables
    self.reached_leaf = False

    #state space variables
    self.max_score = -10000000

    self.Tree = dgl.graph()    #we need to switch this to a 

    self.nodes_added = 0
    self.search_location = 0
    self.depth = 0
    self.plays=0
    self.sub_env = deepcopy(environment)
    self.cumulative_sub_episode_reward = 0


    #initilising the first step of the tree
    self.initial_reward = 0

    #shouldn't action really be null, need to enforce this at some point
    self.add_node_to_tree(state = self.initial_state,reward = self.initial_reward,action = 0)
    self.nodes_added += 1
    self.expand_location(0)

    self.fixed_time = time.time()





  def step(self, action):

    

    

    
    

    _,reward_sub,done,self.search_location = self.move(self.search_location,action)

    

    



    #rename cumulative as it is a dumb name
    self.cumulative_sub_episode_reward += reward_sub


    leaf_node = len([x for x in self.Tree.predecessors(self.search_location)]) == 0

    


    #think want we want in all the other cases and check it is handled appropriately
    if self.MCTS and leaf_node and not done:

      sub_reward = self.rollout()

      done = True

      self.cumulative_sub_episode_reward += sub_reward

      self.expand_location(self.search_location)

    

    



  
    if done:

      if self.reward_shape:
        reward = max(self.cumulative_sub_episode_reward - self.max_score,0)

      else:
        reward = 0


      self.max_score = max(self.max_score,self.cumulative_sub_episode_reward)

      print("Attempt:"+str(self.plays),"score:"+str(reward),"max score:"+str(self.max_score))
      print(time.time()-self.fixed_time)
      self.fixed_time = time.time()

      # propogating reward through the tree for Monte Carlo Tree Search
      self.propogate_reward(tree = self.Tree,location = self.search_location,reward = self.cumulative_sub_episode_reward)


      self.reset_sub_game_variables()
      

      if self.plays == self.search_budget:
        print("finished")
        self.done_overall = True

        if not self.reward_shape:
          reward = self.max_score


    
    else:
      reward = 0

      if leaf_node :
        self.expand_location(self.search_location)

    

    self.children  = nx.get_node_attributes(self.Tree, "action_node_nums")[self.search_location]



    #Would be nice to give this the proper name based on test number and graph number
    if self.test and self.done_overall:
      self.save_image_tree(self.file_location, self.Tree,self.time,self.nodes_added)

    
    

    return [self.Tree,self.search_location,self.search_budget-self.plays,self.max_score,self.children,self.depth],reward,self.done_overall ,None


  

  def reset(self):

    #full episode variables
    self.plays=0
    self.done_overall = False
    self.initial_state = self.sub_env_storage.reset(render_mode='raw')

    #this is the game that we will be repeating
    self.sub_env_storage = deepcopy(self.sub_env_storage)

    #creating a place to store the search tree graphs
    if self.test:
      self.time = time.time()
      os.mkdir(self.file_location+str(self.time))

    #state space variables
    self.max_score = -10000000
    self.Tree = nx.DiGraph()
    self.nodes_added = 0
    self.search_location = 0
    self.depth = 0

    #sub episode variables
    self.sub_env = deepcopy(self.sub_env_storage)
    self.cumulative_sub_episode_reward = 0

    #initilising the first step of the tree
    self.add_node_to_tree(self.initial_state,0,0)
    self.nodes_added += 1
    self.expand_location(0)

    self.fixed_time = time.time()

    #self.children = [x for x in self.Tree.predecessors(self.search_location)]
    self.children  = nx.get_node_attributes(self.Tree, "action_node_nums")[self.search_location]


    return [self.Tree,self.search_location,self.search_budget-self.plays,self.max_score,self.children,self.depth]





  def render(self, mode='human'):
    return None


  def close(self):
    return None


  def save_image_tree(self,location,tree,time,nodes_added):
    plt.figure(figsize=(30,40))
    pos=graphviz_layout(tree, prog='dot')
    nx.draw_networkx(tree, pos, with_labels=False, arrows=True,arrowsize=20)
    plt.gca().invert_yaxis()
    plt.savefig(location+str(time)+'/nx_test_'+str(nodes_added)+'.png')
    plt.close()

  def reset_sub_game_variables(self):
    self.plays = self.plays + 1
    self.search_location = 0
    self.depth = 0
    self.cumulative_sub_episode_reward = 0
    self.sub_env = deepcopy(self.sub_env_storage)


  #just finishes off the game without adding it to the search tree and tells you want the reward was (might need to specify alternatives)
  def rollout(self):
    
    # self.fixed_time = time.time()

    done = False
    total_reward = 0

    total = 0
    while not done:

      action = torch.randint(low=0,high=self.sub_env.action_space.n, size = (1,))
      
      state,reward_sub,done, _  = self.sub_env.step(int(action.numpy()), observation_mode='raw') 

      total_reward += reward_sub
      total += 1

    #print("rollout time",(time.time()-self.fixed_time)/total)
    #self.fixed_time = time.time()

    return total_reward 
  


  def propogate_reward(self,tree,location,reward):


    while True :

      current_total_reward = nx.get_node_attributes(self.Tree, "mean_reward")[location]
      nx.set_node_attributes(self.Tree, {location:current_total_reward + reward}, 'mean_reward')

      if location != 0:
        successors = [pred for pred in self.Tree.successors(location)]

        if len(successors) != 1:
          raise Exception("can't have more than one parent")

        location = successors[0]

      else:
        break

  def move(self,location,action):

      #self.fixed_time = time.time()

    


      action = int(action.numpy())
      state,reward,done, _  = self.sub_env.step(action,observation_mode='raw') 

      location_new = nx.get_node_attributes(self.Tree, "action_node_nums")[location][action]

      self.depth = self.depth + 1

      #print(time.time()-self.fixed_time)

      return state,reward,done,location_new

  
  def add_node_to_tree(self,state,reward,action):

    self.Tree.add_nodes_from(
        [
          (self.nodes_added, {"state":state,"reward":reward,"action":action,"visits":0,"mean_reward":0,"terminal":0,"action_node_nums":[]}),
        ]
    )



  def expand_location(self,root):

    node_nums = []

    for action in range(self.sub_env.action_space.n):

      state,reward,done, _ = deepcopy(self.sub_env).step(action, observation_mode='raw')

      self.add_node_to_tree(state,reward,action)

      self.Tree.add_edge(self.nodes_added, root)

      node_nums.append(self.nodes_added)

      self.nodes_added = self.nodes_added + 1

    nx.set_node_attributes(self.Tree, {root:node_nums}, 'action_node_nums')

    