import torch
import numpy as np
from gym.spaces import Discrete, Box

def n_features(space):
    if isinstance(space, Box):
        return np.prod(space.shape)
    elif isinstance(space, Discrete):
        return space.n
    else:
        raise ValueError("{} is not a valid space type".format(str(space)))


def obs_to_tensor(ob_space):
    '''
    Defines closure for building observation input with encoding 
    depending on the observation space type
    
    Arguments:
    ob_space: observation_space (should be one of gym.spaces)

    returns: function mapping observations to tensors
    '''
    if isinstance(ob_space, Discrete):
        num_classes = ob_space.n
        def process(obs):
            obs = torch.tensor(obs).reshape(-1, 1)
            return (
                obs == torch.arange(num_classes).reshape(1,num_classes).float()
            )
        return process
    elif isinstance(ob_space, Box):
        return torch.Tensor
    else:
        raise ValueError("{} is not a valid space type".format(str(ob_space)))

