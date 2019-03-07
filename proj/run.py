"""
The MIT License

Copyright (c) 2018 OpenAI (http://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

Adapted from OpenAI's Spinning Up: https://github.com/openai/spinningup
"""
import sys
import os
import os.path as osp
import subprocess
from textwrap import dedent
import proj.algorithms
from random import randrange
from proj.utils.exp_grid import ExperimentGrid
from proj.common.models import *


# Command line args that will go to ExperimentGrid.run, and must possess unique
# values (therefore must be treated separately).
RUN_KEYS = ['log_dir', 'format_strs', 'datestamp']


def friendly_err(err_msg):
    # add whitespace to error message to make it more readable
    return '\n\n' + err_msg + '\n\n'


def parse_and_execute_grid_search(cmd, args):

    algo = eval('proj.algorithms.'+cmd)

    # Before all else, check to see if any of the flags is 'help'.
    valid_help = ['--help', '-h', 'help']
    if any([arg in valid_help for arg in args]):
        print('\n\nShowing docstring for spinup.'+cmd+':\n')
        print(algo.__doc__)
        sys.exit()

    def process(arg):
        # Process an arg by eval-ing it, so users can specify more
        # than just strings at the command line (eg allows for
        # users to give functions as args).
        try:
            return eval(arg)
        except:
            return arg

    # Make first pass through args to build base arg_dict. Anything
    # with a '--' in front of it is an argument flag and everything after,
    # until the next flag, is a possible value.
    arg_dict = dict()
    for i, arg in enumerate(args):
        assert i > 0 or '--' in arg, \
            friendly_err("You didn't specify a first flag.")
        if '--' in arg:
            arg_key = arg.lstrip('-')
            arg_dict[arg_key] = []
        else:
            arg_dict[arg_key].append(process(arg))

    # Make second pass through, to catch flags that have no vals.
    # Assume such flags indicate that a boolean parameter should have
    # value True.
    for k,v in arg_dict.items():
        if len(v)==0:
            v.append(True)

    # Final pass: check for the special args that go to the 'run' command
    # for an experiment grid, separate them from the arg dict, and make sure
    # that they have unique values. The special args are given by RUN_KEYS.
    run_kwargs = dict()
    for k in RUN_KEYS:
        if k in arg_dict:
            val = arg_dict[k]
            assert len(val)==1, \
                friendly_err("You can only provide one value for %s."%k)
            run_kwargs[k] = val[0]
            del arg_dict[k]

    # Determine experiment name. If not given by user, will be determined
    # by the algorithm name.
    if 'exp_name' in arg_dict:
        assert len(arg_dict['exp_name'])==1, \
            friendly_err("You can only provide one value for exp_name.")
        exp_name = arg_dict['exp_name'][0]
        del arg_dict['exp_name']
    else:
        exp_name = 'cmd_' + cmd

    # Construct and execute the experiment grid.
    eg = ExperimentGrid(name=exp_name)
    for k,v in arg_dict.items():
        eg.add(k, v)
    eg.run(algo, **run_kwargs)


if __name__ == '__main__':
    cmd = sys.argv[1]
    valid_algos = ['vanilla', 'trpo', 'a2c', 'ppo', 'acktr', 'a2c_kfac', 'ddpg',
                   'td3', 'sac', 'sac2']
    valid_utils = ['plot', 'sim_policy', 'record_policy']
    valid_cmds = valid_algos + valid_utils
    assert cmd in valid_cmds, \
        "Select an algorithm or utility which is implemented in proj."

    if cmd in valid_algos:
        args = sys.argv[2:]
        parse_and_execute_grid_search(cmd, args)

    elif cmd in valid_utils:
        # Execute the correct utility file.
        if cmd == 'plot':
            cmd = osp.join('viskit', 'frontend')
        runfile = osp.join(
            osp.abspath(osp.dirname(__file__)), 'utils', cmd + '.py'
        )
        args = [sys.executable if sys.executable else 'python', runfile] + \
               sys.argv[2:]
        subprocess.check_call(args, env=os.environ)
