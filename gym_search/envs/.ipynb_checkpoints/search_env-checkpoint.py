import gym
from gym import error, spaces, utils
from gym.utils import seeding
import networkx as nx
from copy import deepcopy


class SearchEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self,environment,steps):
    self.plays=0
    self.done_overall = False

    initial_state = environment.reset()

    self.sub_env = environment
    self.steps = steps

    self.action_space = spaces.Discrete(2)

    self.max_score = 0
    self.cumulative = 0

    #tree variables we need to add the initial state to this graph!
    self.Tree = nx.Graph()
    self.nodes_added = 0

    self.Tree.add_nodes_from(
      [
        (self.nodes_added, {"state":initial_state,"reward":0,"action":None}),
      ]
    )

    self.nodes_added += 1



    for action in range(self.sub_env.action_space.n):


      #no attribute of copy which is kind of anoying
      state,reward,done, _ = deepcopy(self.sub_env).step(action)

      self.Tree.add_nodes_from(
        [
          (self.nodes_added, {"state":state,"reward":reward,"action":action}),
        ]
      )

      #need to make sure that this is bi-directional (think we are find as but going to have to check)
      self.Tree.add_edge(self.nodes_added, 0)

      self.nodes_added = self.nodes_added + 1




    self.search_location = 0
    
    
    
    
    
  def preprocess(image, cuda=False):
    image = np.array(image)
    image = image / 255.0
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0).float()
    if cuda: image = image.cuda()
    return image



  def step(self, action):
    #print("step")
    self.preprocess(obs["image"], self.cuda)


    #a step should result in two possibilities

    #1 the change in the location in the search tree

    #2 the change in the location in the search tree and the addition of the children of the resulting nodes to the Tree
      #this case happens when the change in the search tree location means that the node has been visited zero times

    reward_real = 0

    def move(location,action):

      #just required to make the change in this environment , we don't need any of the output
      state,reward,done, _  = self.sub_env.step(action)
      
      #we need to make sure this only considers neighbours that go to this node, only one direction
      for n, nbrs in self.Tree.adj.items():
        for nbr, eattr in nbrs.items():

          if nbr == location and nx.get_node_attributes(self.Tree, "action")[n] == action and n > nbr :
            return n,done

      print("error")

      return location,done


    #changing search tree location as the new state will by construction already be in the tree
    self.old_search_location = self.search_location
    self.search_location,done = move(self.search_location,action)




    def is_leaf(location):

      for n, nbrs in self.Tree.adj.items():

        if n == self.search_location and len(nbrs.items()) >1:
          #print("not leaf")
          return False

      #print("leaf")

      return True



    leaf_node = is_leaf(self.search_location)





    if leaf_node :

      for action in range(self.sub_env.action_space.n):

        #no attribute of copy which is kind of anoying
        state,reward,done, _ = deepcopy(self.sub_env).step(action)

        self.Tree.add_nodes_from(
          [
            (self.nodes_added, {"state":state,"reward":reward,"action":action}),
          ]
        )

        #need to make sure that this is bi-directional (think we are find as but going to have to check)
        self.Tree.add_edge(self.nodes_added, self.search_location)



        self.nodes_added = self.nodes_added + 1

        



        #still need to somehow add the connection between these added nodes and the parents (shouldn't be very hard)



    if done:
      self.plays = self.plays + 1

      self.max_score = max(self.max_score,self.cumulative)
      self.cumulative = 0

      state = self.sub_env.reset()

      self.search_location = 0

      if self.plays == self.steps:
        self.done_overall = True
        reward_real = self.max_score





    return [self.Tree,self.search_location,self.steps-self.plays,self.max_score],reward_real,self.done_overall 



  #this need to be written properly , although don't think I make use of it yet
  def reset(self):

    self.plays=0
    self.done_overall = False

    state = self.sub_env.reset()


    return state 


  def render(self, mode='human'):
    return None


  def close(self):
    return None