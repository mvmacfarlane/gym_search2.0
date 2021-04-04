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
import ray
from torch.distributions import Categorical
import random




class SearchEnv(gym.Env):
  #think I can get rid of this, can't remember what it is for
  metadata = {'render.modes': ['human']}

  def __init__(self,environment,search_budget,reward_shape,action_num,verbose,agent):

    #settings
    self.verbose = verbose

    #long term variables
    self.done_overall = False
    self.initial_state = environment.reset()  
    #self.sub_env_storage = deepcopy(environment)   #we don't need this anymore, since we are using the environment to reset itself
    self.reward_shape = reward_shape                #this is not nessesary anymore (remove)
    self.search_budget = search_budget
    self.action_num = action_num

    
    #state space variables
    self.max_score = None
    self.Tree = dgl.DGLGraph()  
    self.nodes_added = 0
    self.search_location = 0
    self.depth = 0
    self.plays=0
    self.sub_env = deepcopy(environment)   #so guessing this is the one we need to use to play
    self.cumulative_sub_episode_reward = 0
    self.reward = 0    #don't think this is nessesary
    

    #useful variables
    self.leaf = False   #so we ensure the Monte Carlo Estimate is performed (probably unessesary now due to the later estimate) , should test this , you dont need it to be performed!
    self.current_path = []
    self.current_actions = []
    self.best_path = []
    self.best_actions = []
    self.expansions = 0 #just for debugging purposes , can be removed at a later date

    #self.locked_in_value = 0     #this is not general enough, not sure that this belongs here
    self.solved = False
    
    #get rid of this at some point
    self.update_embeddings = agent.update_embeddings()
    self.embeddings = agent.embeddings()

    self.agent = agent
    self.mc_sims = agent.mc_sims




    #adding the root to the tree
    self.add_node_to_tree(state = self.initial_state,reward = 0,action = 0,depth = 0)
    self.nodes_added += 1
    self.expand_location(0)


    #final stuff, really need to think about this at some point
    self.final_moves = []
    self.final_probabilities = []
    self.final_states = []
    self.final_MC_estimates = []
    self.final_MC_variance = []

    

    

  def step(self, action,probabilities = False ,verbose=True):

    #print("start:")
    #start = time.time()
    
    
    #don't think this is going to work for other search proceedures. Only works cause the last route taken for MCTS is by definition the best since it has locked everthing in!
    #Ultimately this idea probably works the best to be honest. Probabilities for each action read out in the final step before the episode terminates
    if self.search_budget - self.plays == 1:
        self.final_moves.append(action)
        self.final_states.append(self.Tree.nodes[[self.search_location]].data["embeddings_1"].squeeze(0))

        self.final_probabilities.append(probabilities)

        #huh we have we got too values here?
        if self.agent.policy_weighting == 0:
          value_estimates = self.rollout(iterf =self.mc_sims,random_rollout = True,printy = True)
        else:
          value_estimates = self.rollout(iterf =self.mc_sims,random_rollout = False,printy = True)

        #val2 = self.rollout(iterf =self.mc_sims,random_rollout = True)


        self.final_MC_estimates.append(value_estimates)
        self.final_MC_variance.append(0)

    
    #reseting the current path
    if self.search_location == 0:
      self.current_path = []
      self.current_actions = []

    self.current_path.append(self.search_location)
    self.current_actions.append(action)



    if action == -1:

      state = self.make_state(self.Tree,self.search_location,self.search_budget-self.plays,self.max_score,self.depth,self.leaf,self.current_path,self.search_budget,self.final_moves,self.final_states,self.final_probabilities,self.solved)
      self.finish(state)


    else:

      _,reward_sub,done,self.search_location,solved = self.move(self.search_location,action)

      

      if self.solved == False:
        self.solved = solved



      self.cumulative_sub_episode_reward += reward_sub

      self.children = self.Tree.predecessors(self.search_location) 
      self.leaf = (len(self.children) == 0 )


      if done:
        state = self.make_state(self.Tree,self.search_location,self.search_budget-self.plays,self.max_score,self.depth,self.leaf,self.current_path,self.search_budget,self.final_moves,self.final_states,self.final_probabilities,self.solved)
        self.finish(state)

      else:
        self.reward = 0

        if self.leaf :
          self.expand_location(self.search_location)
          self.expansions = self.expansions  + 1


    state = self.make_state(self.Tree,self.search_location,self.search_budget-self.plays,self.max_score,self.depth,self.leaf,self.current_path,self.search_budget,self.final_moves,self.final_states,self.final_probabilities,self.solved)


    return state,self.reward,self.done_overall ,None




  def make_state(self,tree,location,budget,max_score,depth,leaf,current_path,search_budget,solution,solution_states,solution_probs,solved):

    state = {

      "tree":self.Tree,
      "search_location":self.search_location,
      "budget_left":self.search_budget-self.plays,
      "max_score":self.max_score,
      "depth":self.depth,   #is this really nessesary
      "leaf":self.leaf,      #is this really nessesary
      "current_path":self.current_path,
      "search_budget":self.search_budget,
      "solution":self.final_moves,
      "solution_states":self.final_states,
      "solution_probabilities":self.final_probabilities,
      "solved":self.solved,
      "MC_estimates":self.final_MC_estimates,
      "MC_variance":self.final_MC_variance,

    }

    return state
  


  def reset(self,number_of_attempts = False):

    #full episode variables
    self.plays=0
    self.done_overall = False
    self.initial_state = self.sub_env.reset()
    #self.sub_env_storage = deepcopy(self.sub_env_storage)
    self.verbose = self.verbose

    #state space variables
    self.max_score = None
    self.Tree = dgl.DGLGraph()
    self.nodes_added = 0
    self.search_location = 0
    self.depth = 0

    #sub episode variables
    #self.sub_env = deepcopy(self.sub_env_storage)
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

    self.final_moves = []
    self.final_probabilities = []
    self.final_states = []
    self.final_MC_estimates = []
    self.final_MC_variance = []

    #the ability to alter the number of attempts which I like quite a bit
    if number_of_attempts != False:
      self.search_budget = number_of_attempts

    state = self.make_state(self.Tree,self.search_location,self.search_budget-self.plays,self.max_score,self.depth,self.leaf,self.current_path,self.search_budget,self.final_moves,self.final_states,self.final_probabilities,self.solved)

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

    #print("checking we have same game")
    #print((test==self.initial_state ).all())
    #time.sleep(10)

    
    

  

  def move(self,location,action):
  

      action = int(action.numpy())
      state,reward,done, _  = self.sub_env.step(action) 

      solved = False


      if reward == 10.9:
        solved = True


      #prevents us having to run the predessesor function
      location_new = int(self.Tree.nodes[[self.search_location]].data['action_node_nums'][0][action].item())

      self.depth = self.depth + 1

      return state,reward,done,location_new,solved


  def add_node_to_tree(self,state,reward,action,depth):

    state = torch.tensor([state],dtype=torch.float32)
    action = torch.tensor([action],dtype=torch.float32)
    reward = torch.tensor([reward],dtype=torch.float32)
    terminal = torch.tensor([0],dtype=torch.float32)
    action_node_nums = torch.tensor([[-1]*self.action_num],dtype=torch.float32)
    depth = torch.tensor([depth],dtype=torch.float32).unsqueeze(1)

    keys = ["state","reward","action","terminal","action_node_nums","Monte_Carlo_Value_Estimate"]
    values = [state,reward,action,terminal,action_node_nums,torch.zeros(1)]

    embedding_values = [x(state,depth) for x in self.embeddings]
    embedding_keys = ["embeddings_"+str(i+1) for i,x in enumerate(self.embeddings)]

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

    depth = torch.tensor([depth]*self.action_num,dtype=torch.float32).unsqueeze(1)


    keys = ["state","reward","action","terminal","action_node_nums","Monte_Carlo_Value_Estimate"]
    values = [state,reward,action,terminal,action_node_nums,torch.zeros(self.action_num)]


    embedding_values = [x(state,depth) for x in self.embeddings]
    embedding_keys = ["embeddings_"+str(i+1) for i,x in enumerate(self.embeddings)]

    total_values = values+embedding_values
    total_keys = keys + embedding_keys  

    embedding_dict = { x : y for x,y in zip(total_keys,total_values)}


    self.Tree.add_nodes(self.action_num,embedding_dict)



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




  #get rid of all the reward shaping nonsense
  def finish(self,state):

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


      #these are the complex operations that are taking up all of the time...!

      #1
      self.update_graph_embeddings(state)

      #2
      self.reset_sub_game_variables()

      

      if self.plays == self.search_budget:

        if self.verbose:

          print("finished")
          print("Overall score:"+str(self.max_score))
        self.done_overall = True
        
        if not self.reward_shape:
          self.reward = self.max_score


    
  #at some point need to make the value estimate optional as only normal MCTS needs it
  def update_graph_embeddings(self,state):

    

    value_estimates = self.rollout(iterf =1,random_rollout = True)

    value_estimate = sum(value_estimates)/len(value_estimates)


    for update in self.update_embeddings:
      update(state,value_estimate)



  def rollout(self,iterf = 1,random_rollout = True,printy = False):

    total_rewards = iterf*[0]
    current_reward = 0


    
    for i in range(iterf):
      total_reward = current_reward
      done = False
      state = self.Tree.nodes[[self.search_location]].data["embeddings_1"]   #can be made more efficient at some point!
      depth = self.depth

    


      sub_env = deepcopy(self.sub_env)

      while not done:

        if not random_rollout:
          probabilities = self.agent.predict_policy(state)

          m = Categorical(probabilities)
          action = int(m.sample().item())
        else:
          action = random.randint(0, self.sub_env.action_space.n-1)

        #This needs to be updated!
        
        
        state,reward_sub,done, _  = sub_env.step(action) 
        
        depth += 1
        

        state = torch.tensor([state],dtype=torch.float32)

        state = torch.flatten(state,start_dim=1)

        depth = torch.tensor([depth],dtype=torch.float32).unsqueeze(1)   #this needs to be given the correct depth!
        
        state = torch.cat((state,depth),dim = 1)


        total_reward += reward_sub



      total_rewards[i] = total_reward




    return total_rewards
































  '''
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

    terminal = deepcopy(self.Tree.ndata['terminal'])
    action_node_nums = deepcopy(self.Tree.ndata['action_node_nums'])

    new_env.Tree.add_nodes(state.shape[0], {"state":state,"reward":reward,"action":action,"terminal":terminal,"action_node_nums":action_node_nums})

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


  

  
  
  '''
  def save_image_tree(self,location,tree,time,nodes_added):
    plt.figure(figsize=(30,40))
    pos=graphviz_layout(tree, prog='dot')
    nx.draw_networkx(tree, pos, with_labels=False, arrows=True,arrowsize=20)
    plt.gca().invert_yaxis()
    plt.savefig(location+str(time)+'/nx_test_'+str(nodes_added)+'.png')
    plt.close()

  '''

