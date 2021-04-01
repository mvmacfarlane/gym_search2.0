import gym
from gym import error, spaces, utils
from gym.utils import seeding
from copy import deepcopy
import numpy as np
import torch
import time
import os
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.classes.function import all_neighbors
import matplotlib.pyplot as plt
import dgl
import dgl.function as fn
import sys
import pickle
from pympler import muppy, summary
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display





class SearchEnv(gym.Env):
  #think I can get rid of this, can't remember what it is for
  metadata = {'render.modes': ['human']}


  def __init__(self,environment,search_budget,reward_shape,action_num,verbose,agent):




    
    #long term variables
    self.done_overall = False
    self.initial_state = environment.reset()
    self.sub_env_storage = deepcopy(environment)
    self.reward_shape = reward_shape
    self.file_location = '/home/mvmacfarlane/Documents/PhD/PhD/graphs/'
    self.search_budget = search_budget
    self.verbose = verbose
    self.action_num = action_num
    

    #state space variables
    self.max_score = None
    self.Tree = dgl.DGLGraph()  
    self.nodes_added = 0
    self.search_location = 0
    self.depth = 0
    self.plays=0
    self.sub_env = deepcopy(environment)
    self.cumulative_sub_episode_reward = 0
    self.reward = 0
    

    #useful variables
    self.leaf = True
    self.current_path = []
    self.current_actions = []
    self.best_path = []
    self.best_actions = []
    self.expansions = 0 #is this nessesary
    
    
    

    #graph update functions
    self.update_encodings = agent.update_encodings

    self.update_graph_embeddings_type = agent.update_graph_embeddings_type
    self.embed_children = agent.embed_children

    self.embed_children2 = agent.embed_children2

    self.embed_children3 = agent.embed_children3


    try:
      self.update_encodings2 = agent.update_encodings2
    except:
      self.update_encodings2 = None

    try:
      self.update_encodings3 = agent.update_encodings3
    except:
      self.update_encodings3 = None


    try:
      self.get_value = agent.get_value
    except:
      self.get_value = None


    #adding the root to the tree
    self.add_node_to_tree(state = self.initial_state,reward = 0,action = 0)
    self.nodes_added += 1
    self.expand_location(0)

    




    

  
  
  def step(self, action,verbose=True):
    
    
    #reseting the current path
    if self.search_location == 0:
      self.current_path = []
      self.current_actions = []

    self.current_path.append(self.search_location)
    self.current_actions.append(action)


    #checking if we should return to root node
    if action == -1:
      self.finish()
    else:
      _,reward_sub,done,self.search_location = self.move(self.search_location,action)

      self.cumulative_sub_episode_reward += reward_sub

      #we could just get the predessors
      #self.children = self.Tree.nodes[[self.search_location]].data['action_node_nums']
      #self.leaf = (int(self.children[0][0]) == -1)

      self.children = self.Tree.predecessors(self.search_location)
      self.leaf = (len(self.children) == 0 ) 


      if done:
        self.finish()

      else:
        self.reward = 0

        #this function takes ages and has the potential to massively cut down the simulation time
        if self.leaf :
          self.expand_location(self.search_location)
          self.expansions = self.expansions  + 1

      


    children_visits = self.Tree.nodes[self.Tree.predecessors(self.search_location)].data['visits']
    children_reward = self.Tree.nodes[self.Tree.predecessors(self.search_location)].data['mean_reward']
    children_embeddings = self.Tree.nodes[self.Tree.predecessors(self.search_location)].data['embeddings']
    children_embeddings2 = self.Tree.nodes[self.Tree.predecessors(self.search_location)].data['embeddings2']

    
    #this won't work with dgl , will need to convert to networkx first
    #if self.test and self.done_overall:
    #  self.save_image_tree(self.file_location, self.Tree,self.time,self.nodes_added)

    state = {

      "tree":self.Tree,
      "search_location":self.search_location,
      "budget_left":self.search_budget-self.plays,
      "max_score":self.max_score,
      "children":self.children,
      "depth":self.depth,
      "leaf":self.leaf,
      "current_path":self.current_path,
      "children_visit_statistics":children_visits,
      "children_reward_statistics":children_reward,
      "search_budget":self.search_budget,
      "children_embeddings":children_embeddings,
      "children_embeddings2":children_embeddings2,
    }

    return state,self.reward,self.done_overall ,None


  

  def reset(self):

    #full episode variables
    self.plays=0
    self.done_overall = False
    self.initial_state = self.sub_env_storage.reset()
    self.sub_env_storage = deepcopy(self.sub_env_storage)
    self.verbose = self.verbose

    #state space variables
    self.max_score = None
    self.Tree = dgl.DGLGraph()
    self.nodes_added = 0
    self.search_location = 0
    self.depth = 0

    #sub episode variables
    self.sub_env = deepcopy(self.sub_env_storage)
    self.cumulative_sub_episode_reward = 0
    self.leaf = True
    self.current_path = []
    self.current_actions = []
    self.best_path = []
    self.best_actions = []
    self.expansions = 0
    self.reward = 0

    #initilising the first step of the tree
    self.add_node_to_tree(self.initial_state,0,0)
    self.nodes_added += 1
    self.expand_location(0)






  
    #self.children = self.Tree.nodes[[self.search_location]].data['action_node_nums']

    self.children = self.Tree.predecessors(self.search_location)


    children_visits = self.Tree.nodes[self.Tree.predecessors(self.search_location)].data['visits']
    children_reward = self.Tree.nodes[self.Tree.predecessors(self.search_location)].data['mean_reward']


    children_embeddings = self.Tree.nodes[self.Tree.predecessors(self.search_location)].data['embeddings']
    children_embeddings2 = self.Tree.nodes[self.Tree.predecessors(self.search_location)].data['embeddings2']



    state = {

      "tree":self.Tree,
      "search_location":self.search_location,
      "budget_left":self.search_budget-self.plays,
      "max_score":self.max_score,
      "children":self.children,
      "depth":self.depth,
      "leaf":self.leaf,
      "current_path":self.current_path,
      "children_visit_statistics":children_visits,
      "children_reward_statistics":children_reward,
      "search_budget":self.search_budget,
      "children_embeddings":children_embeddings,
      "children_embeddings2":children_embeddings2,
    }

    return state


  #this doesn't really generalise
  def render(self, mode='tiny_rgb_array'):

    return self.sub_env.render(mode=mode)


  def close(self):
    return None

  def reset_sub_game_variables(self):
    self.plays = self.plays + 1
    self.search_location = 0
    self.depth = 0
    self.leaf = False
    self.cumulative_sub_episode_reward = 0
    self.sub_env = deepcopy(self.sub_env_storage)
    self.expansions = 0
    self.reward = 0





  def move(self,location,action):

      action = int(action.numpy())
      state,reward,done, _  = self.sub_env.step(action) 

      location_new = int(self.Tree.nodes[[location]].data['action_node_nums'][0][action])

      self.depth = self.depth + 1

      return state,reward,done,location_new


  def add_node_to_tree(self,state,reward,action):

    state = torch.tensor([state],dtype=torch.float32)
    action = torch.tensor([action],dtype=torch.float32)
    reward = torch.tensor([reward],dtype=torch.float32)
    visits = torch.tensor([0],dtype=torch.float32)
    mean_reward = torch.tensor([0],dtype=torch.float32)
    terminal = torch.tensor([0],dtype=torch.float32)
    action_node_nums = torch.tensor([[-1]*self.action_num],dtype=torch.float32)



    #we immediately encode this if an encoding function is provided
    embeddings = self.embed_children(state)
    embeddings2 = self.embed_children2(state)
    embeddings3 = self.embed_children3(state)





  
    self.Tree.add_nodes(1, {"state":state,"reward":reward,"action":action,"visits":visits,"mean_reward":mean_reward,"terminal":terminal,"action_node_nums":action_node_nums,"embeddings":embeddings,"embeddings2":embeddings2,"embeddings3":embeddings3})

  
  def add_nodes_to_tree(self,state,reward,action):

    state = torch.tensor(state,dtype=torch.float32)
    action = torch.tensor(action,dtype=torch.float32)

    reward = torch.tensor(reward,dtype=torch.float32)
    visits = torch.tensor([0]*self.action_num,dtype=torch.float32)


    mean_reward = torch.tensor([0]*self.action_num,dtype=torch.float32)
    terminal = torch.tensor([0]*self.action_num,dtype=torch.float32)
    action_node_nums = torch.tensor([[-1]*self.action_num for i in range(self.action_num)],dtype=torch.float32)

    #we immediately encode this if an encoding function is provided


    embeddings = self.embed_children(state)
    embeddings2 = self.embed_children2(state)
    embeddings3 = self.embed_children3(state)



    self.Tree.add_nodes(self.action_num, {"state":state,"reward":reward,"action":action,"visits":visits,"mean_reward":mean_reward,"terminal":terminal,"action_node_nums":action_node_nums,"embeddings":embeddings,"embeddings2":embeddings2,"embeddings3":embeddings3})



  def expand_location(self,root):

    states = []
    rewards = []
    dones = []
    actions = [i for i in range(self.sub_env.action_space.n)]

    for action in range(self.sub_env.action_space.n):

      new_env = deepcopy(self.sub_env) 
      state,reward,done, _ = new_env.step(action)

      states.append(state)
      rewards.append(reward)
      dones.append(done)

    self.add_nodes_to_tree(states,rewards,actions)

    roots = [root for i in range(self.sub_env.action_space.n)]
    nodes_to_add = [self.nodes_added + i for i in range(self.sub_env.action_space.n)] 

    self.Tree.add_edges(nodes_to_add,roots)

    self.nodes_added = self.nodes_added + self.sub_env.action_space.n 

    #this might not be nessesary
    new = torch.tensor([nodes_to_add],dtype=torch.float32)
    self.Tree.nodes[[root]].data['action_node_nums'] = new


  
  #this needs to be tidied , looks like an absolute mess
  def finish(self):
      if self.verbose:
        print("finish")

      
      if self.reward_shape:
        if self.max_score != None:
          self.reward = max(self.cumulative_sub_episode_reward - self.max_score,0)
        else:
          self.reward = self.cumulative_sub_episode_reward
      else:
        self.reward = 0


      if self.max_score == None:
        self.best_path = self.current_path
        self.best_actions = self.current_actions
        self.max_score = self.cumulative_sub_episode_reward

      else:

        if self.max_score < self.cumulative_sub_episode_reward :
          self.best_path = self.current_path
          self.best_actions = self.current_actions
          self.max_score = self.cumulative_sub_episode_reward


      if self.verbose:
        print("Attempt:"+str(self.plays),"score:"+str(self.reward),"max score:"+str(self.max_score),"length best path:",len(self.best_actions),"Number of expansions:"+str(self.expansions),"Depth:"+str(self.depth))
        print("nodes in tree is",self.Tree.number_of_nodes())



      self.update_graph_embeddings(self.update_graph_embeddings_type)
      


      self.reset_sub_game_variables()

      if self.plays == self.search_budget:

        print("finished")
        print("Overall score:"+str(self.max_score))
        self.done_overall = True
        
        if not self.reward_shape:
          self.reward = self.max_score


  def update_graph_embeddings(self,type):


    if type == "RAVE":
      value_state = self.get_value(self.Tree.nodes[[self.search_location]].data['state']).detach().numpy()[0]
      self.update_encodings(value_state,self.Tree,self.search_location)


    elif type == "MC":

      value_state = self.rollout() + self.cumulative_sub_episode_reward
      self.update_encodings(value_state,self.Tree,self.search_location)


    elif type == 'GNN':
      self.update_encodings(self.Tree,self.current_path)

      value_state = self.rollout() + self.cumulative_sub_episode_reward
      self.update_encodings2(value_state,self.Tree,self.search_location)

      self.update_encodings3(value_state,self.Tree,self.search_location)



    


    else:
      print('error')






  def rollout(self):
    
    done = False
    total_reward = 0


    while not done:

      action = torch.randint(low=0,high=self.sub_env.action_space.n, size = (1,))
      
      state,reward_sub,done, _  = self.sub_env.step(int(action.numpy())) 

      total_reward += reward_sub

    return total_reward 


  def deepcopy(self):


    new_env = gym.make('gym_search:search-v0',environment = deepcopy(self.sub_env),search_budget = deepcopy(self.search_budget),reward_shape= deepcopy(self.reward_shape),MCTS= deepcopy(self.MCTS))


    new_env.done_overall = deepcopy(self.done_overall)
    new_env.initial_state  = deepcopy(self.initial_state)
    new_env.sub_env_storage =deepcopy(self.sub_env_storage)
    new_env.reward_shape =deepcopy(self.reward_shape)
    new_env.file_location =deepcopy(self.file_location)
    new_env.search_budget =deepcopy(self.search_budget)
    new_env.reached_leaf  =deepcopy(self.reached_leaf)
    new_env.max_score = deepcopy(self.max_score)
    new_env.Tree = dgl.DGLGraph() 
    new_env.Tree.add_edges(self.Tree.edges(form='uv')[0],self.Tree.edges(form='uv')[1])

    state = deepcopy(self.Tree.ndata['state'])
    reward = deepcopy(self.Tree.ndata['reward'])
    action = deepcopy(self.Tree.ndata['action'])
    visits = deepcopy(self.Tree.ndata['visits'])
    mean_reward = deepcopy(self.Tree.ndata['mean_reward'])
    terminal = deepcopy(self.Tree.ndata['terminal'])
    action_node_nums = deepcopy(self.Tree.ndata['action_node_nums'])

    new_env.Tree.add_nodes(state.shape[0], {"state":state,"reward":reward,"action":action,"visits":visits,"mean_reward":mean_reward,"terminal":terminal,"action_node_nums":action_node_nums})

    new_env.nodes_added = deepcopy(self.nodes_added)
    new_env.search_location = deepcopy(self.search_location)
    new_env.depth = deepcopy(self.depth)
    new_env.plays = deepcopy(self.plays)
    new_env.sub_env = deepcopy(self.sub_env)
    new_env.cumulative_sub_episode_reward = deepcopy(self.cumulative_sub_episode_reward)
    new_env.leaf = deepcopy(self.leaf)
    new_env.current_path = deepcopy(self.current_path)
    new_env.current_actions = deepcopy(self.current_actions)
    new_env.expansions = deepcopy(self.expansions)
    new_env.reward = deepcopy(self.reward)
    new_env.best_path = deepcopy(self.best_path)
    new_env.best_actions = deepcopy(self.best_actions)

    return new_env


  

  
  
  '''
  def save_image_tree(self,location,tree,time,nodes_added):
    plt.figure(figsize=(30,40))
    pos=graphviz_layout(tree, prog='dot')
    nx.draw_networkx(tree, pos, with_labels=False, arrows=True,arrowsize=20)
    plt.gca().invert_yaxis()
    plt.savefig(location+str(time)+'/nx_test_'+str(nodes_added)+'.png')
    plt.close()

  '''

