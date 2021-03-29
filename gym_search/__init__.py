from gym.envs.registration import register

register(
    id='search-v0',
    entry_point='gym_search.envs:SearchEnv',
)
