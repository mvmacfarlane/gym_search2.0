import gym
from gym import error, spaces, utils
from gym.utils import seeding

from copy import deepcopy
import random
import time
import os
import sys
import pickle
from IPython import display

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.classes.function import all_neighbors

import matplotlib.pyplot as plt

import dgl
import dgl.function as fn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import ray
 
from torch.distributions import Categorical
import torch






class SearchEnv(gym.Env):
  #think I can get rid of this, can't remember what it is for
  metadata = {'render.modes': ['human']}

  #reorder these inputs when I have time
  def __init__(self,environment,search_budget,reward_shape,verbose,agent,training_mode):

    #settings
    self.verbose = verbose

    #long term variables
    self.done_overall = False
    self.initial_state = environment.reset()  
    self.reward_shape = reward_shape
    self.search_budget = search_budget
    self.action_num = environment.action_space.n

    #state space variables
    self.Tree = dgl.DGLGraph()
    self.reward = 0
    self.max_score = None
    self.search_location = 0
    self.plays=0
    self.sub_env = deepcopy(environment)

    #useful derivates
    self.nodes_added = 0
    self.depth = 0
    self.cumulative_sub_episode_reward = 0
    self.leaf = False
    self.current_path = []
    self.current_actions = []
    self.best_path = []
    self.best_actions = []
    self.expansions = 0
    self.solved = False
    self.last_added = False

    #will need to think about the differences of allowing this to normal open ai envs
    self.agent = agent

    #in training mode we will perform rollouts to generate value targets
    self.training_mode = training_mode

    #adding the root to the tree
    self.add_node_to_tree(state = self.initial_state,reward = 0,action = 0,depth = 0)
    self.nodes_added += 1  #surely this should be taken care of in the add node to tree function
    self.expand_location(0)
    

    #end of search data to use for training
    self.final_probabilities = []
    self.final_MC_estimates = []
    self.final_state_num = []
    
    self.lockin_times = {x:0 for x in range(self.sub_env.max_steps)}   #this matters for generating our probability scores and subtrees

    state = self.make_state()
    state["current_path"] = [0]
    self.update_graph_embeddings(state,True)

    

  def step(self,action,locked,probabilities = False ,verbose=True):

    #think if this is the correct place to do this
    if locked:
      if self.lockin_times[self.depth] == 0:
        self.lockin_times[self.depth] = self.Tree.number_of_nodes()

    #reseting the current path
    if self.search_location == 0:
      self.current_path = []
      self.current_actions = []

    self.current_path.append(self.search_location)
    self.current_actions.append(action)


    if self.search_budget - self.plays == 1:
      self.final_state_num.append(self.search_location)
      self.final_probabilities.append(probabilities)

    
    
    #checking if we should return to root
    if action == -1:
      state = self.make_state()
      self.finish(state)

    else:
      _,reward_sub,done,self.search_location,solved = self.move(self.search_location,action)

      self.update_state_variables(reward_sub,done,solved)

      #yes so we are done
      if done:
        state = self.make_state()
        self.finish(state)
        self.last_added = False

      else:
        self.reward = 0   #what even is the point of this

        if self.leaf :
          self.expand_location(self.search_location)
          self.expansions = self.expansions  + 1


    state = self.make_state()

    return state,self.reward,self.done_overall ,None


  def reset(self,number_of_attempts = False):

    #full episode variables
    self.plays=0
    self.done_overall = False
    self.initial_state = self.sub_env.reset()

    #state space variables
    self.max_score = None
    self.Tree = dgl.DGLGraph()
    self.nodes_added = 0
    self.search_location = 0
    self.depth = 0

    #sub episode variables
    self.cumulative_sub_episode_reward = 0
    self.leaf = False
    self.current_path = []
    self.current_actions = []
    self.best_path = []
    self.best_actions = []
    self.expansions = 0
    self.reward = 0
    self.solved = False

    #initilising the first step of the tree
    self.add_node_to_tree(self.initial_state,0,0,depth = 0)
    self.nodes_added += 1
    self.expand_location(0)

    self.final_probabilities = []
    self.final_state_num = []
    self.final_MC_estimates = []

    if number_of_attempts != False:
      self.search_budget = number_of_attempts

    state = self.make_state()
    state["current_path"] = [0]
    self.update_graph_embeddings(state,True)

    return state



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
    self.expansions = 0
    self.reward = 0

    _ = self.sub_env.reset()


  def all_predecessors(self,graph,node,nodes):
    nodes.append(node)

    for i in graph.predecessors(node):
      nodes = nodes + self.all_predecessors(graph,i,[])

    return nodes

  def locked(self,depth,lockin_times):

    i=0
    while True:
      if lockin_times[i] == 0:
        break
      else:
        i+=1
    
    if depth >= i:
      return False
    else:
      return True


  def move(self,location,action):
  
      action = int(action.numpy())
      state,reward,done, _  = self.sub_env.step(action) 

      solved = False

      #need to find a non hardcoded solution to this
      if reward == 10.9:
        solved = True

      #prevents us having to run the predessesor function
      location_new = int(self.Tree.nodes[[self.search_location]].data['action_node_nums'][0][action].item())

      self.depth = self.depth + 1

      return state,reward,done,location_new,solved


  #make these into one function, not urgent and requires some careful thinking
  def add_node_to_tree(self,state,reward,action,depth):

    state = torch.tensor([state],dtype=torch.float32)
    action = torch.tensor([action],dtype=torch.float32)
    reward = torch.tensor([reward],dtype=torch.float32)
    terminal = torch.tensor([0],dtype=torch.float32)
    action_node_nums = torch.tensor([[-1]*self.action_num],dtype=torch.float32)
    depth = torch.tensor([depth],dtype=torch.float32)

    keys = ["state","reward","depth","action","terminal","action_node_nums","Monte_Carlo_Value_Estimate"]
    values = [state,reward,depth,action,terminal,action_node_nums,torch.zeros(1)]

    embedding_values = [x(state,depth.unsqueeze(1),reward) for x in self.agent.embeddings()]
    embedding_keys = ["embeddings_"+str(i+1) for i,x in enumerate(self.agent.embeddings())]

    total_values = values+embedding_values
    total_keys = keys + embedding_keys  

    embedding_dict = { x : y for x,y in zip(total_keys,total_values)}

    self.Tree.add_nodes(1,embedding_dict)


  def add_nodes_to_tree(self,state,reward,action,depth):

    state = torch.tensor(state,dtype=torch.float32)
    action = torch.tensor(action,dtype=torch.float32)
    reward = torch.tensor(reward,dtype=torch.float32)

    terminal = torch.tensor([0]*self.action_num,dtype=torch.float32)
    action_node_nums = torch.tensor([[-1]*self.action_num for i in range(self.action_num)],dtype=torch.float32)

    depth = torch.tensor([depth]*self.action_num,dtype=torch.float32)


    keys = ["state","reward","depth","action","terminal","action_node_nums","Monte_Carlo_Value_Estimate"]
    values = [state,reward,depth,action,terminal,action_node_nums,torch.zeros(self.action_num)]


    embedding_values = [x(state,depth.unsqueeze(1),reward) for x in self.agent.embeddings()]
    embedding_keys = ["embeddings_"+str(i+1) for i,x in enumerate(self.agent.embeddings())]

    total_values = values+embedding_values
    total_keys = keys + embedding_keys  

    embedding_dict = { x : y for x,y in zip(total_keys,total_values)}


    self.Tree.add_nodes(self.action_num,embedding_dict)


  def make_state(self):

    state = {

      "tree":self.Tree,
      "search_location":self.search_location,
      "budget_left":self.search_budget-self.plays,
      "max_score":self.max_score,
      "depth":self.depth,
      "leaf":self.leaf,
      "current_path":self.current_path,
      "search_budget":self.search_budget,
      "solution_probabilities":self.final_probabilities,
      "solved":self.solved,
      "MC_estimates":self.final_MC_estimates,
      "solution_num":self.final_state_num,
      "lockin_iterations":self.lockin_times,
      "last_added":self.last_added,
    }

    return state



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

  
    self.add_nodes_to_tree(states,rewards,actions,self.depth+1)

    roots = [root for i in range(self.sub_env.action_space.n)]
    nodes_to_add = [self.nodes_added + i for i in range(self.sub_env.action_space.n)] 


    self.Tree.add_edges(nodes_to_add,roots)
    self.nodes_added = self.nodes_added + self.sub_env.action_space.n 

    new = torch.tensor([nodes_to_add],dtype=torch.float32)
    self.Tree.nodes[[root]].data['action_node_nums'] = new
    self.last_added = nodes_to_add


  def finish(self,state):

      def set_reward():
        if self.reward_shape:
          if self.max_score != None:
            self.reward = max(self.cumulative_sub_episode_reward - self.max_score,0)
          else:
            self.reward = self.cumulative_sub_episode_reward
        else:
          self.reward = 0
      def set_best_variables():
        if self.max_score == None:
          self.best_path = self.current_path
          self.best_actions = self.current_actions
          self.max_score = self.cumulative_sub_episode_reward

        else:

          if self.max_score < self.cumulative_sub_episode_reward :
            self.best_path = self.current_path
            self.best_actions = self.current_actions
            self.max_score = self.cumulative_sub_episode_reward

      set_reward()
      set_best_variables()

  
      if self.verbose:
        print("Attempt:"+str(self.plays),"score:"+str(self.reward),"max score:"+str(self.max_score),"length best path:",len(self.best_actions),"Number of expansions:"+str(self.expansions),"Depth:"+str(self.depth))
        print("nodes in tree is",self.Tree.number_of_nodes())

      #1
      self.update_graph_embeddings(state,True)

      #2
      self.reset_sub_game_variables()


      if self.plays == self.search_budget:

        if self.verbose:
          print("finished")
          print("Overall score:"+str(self.max_score))

        self.done_overall = True
        
        if not self.reward_shape:
          self.reward = self.max_score



  def update_state_variables(self,reward_sub,done,solved):
    if self.solved == False:
      self.solved = solved

    self.cumulative_sub_episode_reward += reward_sub
    self.children = self.Tree.predecessors(self.search_location) 
    self.leaf = (len(self.children) == 0 )


    
  def update_graph_embeddings(self,state,estimate_value = False):

    if estimate_value:
      value_estimates = self.rollout(number_of_rollouts =1,random_rollout = True)
      value_estimate = sum(value_estimates)/len(value_estimates)
    else:
      value_estimate = 0
 
    for update in self.agent.update_embeddings():

      nodes_to_update = state["current_path"]

      if state["last_added"] != False:
        nodes_to_update =  nodes_to_update + state["last_added"]

      nodes_to_update.reverse()
      nodes_to_update = [[x] for x in nodes_to_update]
      
      update(state,value_estimate,nodes_to_update)

      


  #think we should just stick with a random rollout for now on this one
  def rollout(self,number_of_rollouts,random_rollout):

    total_rewards = number_of_rollouts*[0]
    current_reward = 0

    for i in range(number_of_rollouts):
      total_reward = current_reward
      done = False
      
      depth = self.depth
      state = self.Tree.nodes[[self.search_location]].data["embeddings_1"]
      sub_env = deepcopy(self.sub_env)

      while not done:

        #this can be implemented once we start doing policy iteration
        """
        if not random_rollout:
          
          probabilities = self.agent.predict_policy(state)
          m = Categorical(probabilities)
          action = int(m.sample().item())
        """

        action = random.randint(0, self.sub_env.action_space.n-1)

        new_state,reward_sub,done, _  = sub_env.step(action) 
        depth += 1


        total_reward += reward_sub

      total_rewards[i] = total_reward


    return total_rewards
























  #keep this for now as might have to do something with the rollouts at some point
    '''
    #don't think this is going to work for other search proceedures. Only works cause the last route taken for MCTS is by definition the best since it has locked everthing in!
    #Ultimately this idea probably works the best to be honest. Probabilities for each action read out in the final step before the episode terminates
    #wrap this in a function
    #this probably isn't ultimately actually nessesary
    if self.search_budget - self.plays == 1:
        self.final_moves.append(action)
        self.final_states.append(self.Tree.nodes[[self.search_location]].data["embeddings_1"].squeeze(0))  #but in some instances we change the embeddings, we needs a raw version
        self.final_state_num.append(self.search_location) #is this ever actually used?

        self.final_probabilities.append(probabilities) #this is hoewever very important

        #hmm what is the behaviour of this in the GNN case!(should wrap this in a function as hard to know what it is doing)
        if self.training_mode:
          if self.agent.policy_weighting == 0:
            value_estimates = self.rollout(iterf =self.agent.mc_sims,random_rollout = True,printy = True)
          else:
            value_estimates = self.rollout(iterf =self.agent.mc_sims,random_rollout = False,printy = True)

          self.final_MC_estimates.append(value_estimates)
    '''