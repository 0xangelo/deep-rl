import torch
from gym.spaces import Discrete, Box

def observation_input(ob_space):
    '''
    Defines closure for building observation input with encoding 
    depending on the observation space type
    
    Arguments:
    ob_space: observation_space (should be one of gym.spaces)

    returns: function mapping observations to tensors
    '''
    # if isinstance(ob_space, Discrete):
    #     def obs_inpt(obs):
    #         obs = torch.as_tensor(obs, dtype=torch.int32)
    #         zeros = torch.zeros(ob_space.n + obs.shape)[obs] = 1
    #         return obs
    #     return obs_inpt
    if isinstance(ob_space, Box):
        def obs_inpt(obs):
            return torch.as_tensor(obs, dtype=torch.float32)
        return obs_inpt
    else:
        raise NotImplementedError

