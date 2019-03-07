# deep-rl
PyTorch implementation of reinforcement learning algorithms. Many of the algorithms and utilities are based on the excellent 
materials provided by the [Berkley DeepRL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/home?authuser=0) and 
OpenAI's [SpinningUp](https://spinningup.openai.com/en/latest/).

## Requirements

* Python 3
* [PyTorch](http://pytorch.org/) (tested with version 1.0.1)
* [OpenAI baselines](https://github.com/openai/baselines) (for logging and vectorized environment interfaces)

## Installation

With the dependencies above installed, navigate to the root of the repository and run:
```bash
pip install -e .
```

## Running experiments

Use the same [interface](https://spinningup.openai.com/en/latest/user/running.html) described in SpinningUp:
```bash
python -m proj.run ALG [experiment flags]
```
No shortcuts are configured except for the `--env ENV_NAME` flag. See the examples section below for tested hyperparameters.
Experiment results are saved under `data/<exp_name>` by default.

## Utilities

You can simulate/record trained policies and plot experiment results. These utilities use 
[click](https://click.palletsprojects.com/en/7.x/) to handle command line arguments and can be run as follows.
```bash
# using one of <sim_policy,record_policy,plot> in UTIL
python -m proj.run UTIL [OPTIONS] ARGS
# to get help with arguments 
python -m proj.run UTIL --help
```

## Examples
### Vanilla Policy Gradient
```bash
python -m proj.run vanilla --env CartPole-v0 --policy:class MlpPolicy --policy:hidden_sizes [32,32] --optimizer:lr 7e-3 --total_samples 'int(2e5)'
```
### TRPO
```bash
python -m proj.run trpo --env HalfCheetah-v2 --policy:class MlpPolicy --policy:hidden_sizes [64,32] --total_samples 'int(3e6)' --n_envs 8 --steps 500
```

### ACKTR
```bash
python -m proj.run acktr --env HalfCheetah-v2 --policy:class MlpPolicy --policy:hidden_sizes [64,32] --total_samples 'int(3e6)' --n_envs 16 --steps 125
```

### A2C
```bash
python -m proj.run a2c --env PongNoFrameskip-v4 --policy:class CNNWeightSharingAC --log_interval 10 --save_interval 100 --seed 234
```

### DDPG
```bash
python -m proj.run ddpg --env HalfCheetah-v2 --policy:class MlpDeterministicPolicy --policy:hidden_sizes [400,300] --policy:activation relu --total_samples 'int(2e5)' --epoch 10000 --save_interval 1
```

### Twin Delayed DDPG (TD3)
```bash
python -m proj.run td3 --env HalfCheetah-v2 --policy:class MlpDeterministicPolicy --policy:hidden_sizes [400,300] --policy:activation relu --total_samples 'int(2e5)' --epoch 10000 --save_interval 1
```

### Soft Actor-Critic
```bash
python -m proj.run sac2 --env Ant-v2 --policy:class MlpPolicy False --policy:hidden_sizes [256,256] --policy:activation relu --policy:indep_std --policy:clamp_acts --total_samples 'int(3e6)' --save_interval 1 --epoch 10000 --lr 3e-4 --mb_size 256 --target_entropy auto
```
