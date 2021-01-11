import gym
from gym import error, spaces, utils
from gym.utils import seeding

class SearchEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self,sub_env):
    self.plays=0
    self.done_overall = False

    sub_env.reset()

    self.sub_env = sub_env




  def step(self, action):

    state,reward,done, _ = sub_env.step(action)

    if done:
      self.plays = self.plays + 1

      state = self.sub_env.reset()

      if self.plays == 10:
        self.done_overall = True


    return state,reward,done_overall, _ 


  def reset(self):

    self.plays=0
    self.done_overall = False

    state = sub_env.reset()

    self.sub_env = sub_env

    return state 


  def render(self, mode='human'):
    return None


  def close(self):
    return None