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
import string
import subprocess
import base64
import cloudpickle
import zlib
import time
from collections import OrderedDict
from subprocess import CalledProcessError
from textwrap import dedent
from proj.utils.json_util import convert_json

DIV_LINE_WIDTH = 80
DEFAULT_SHORTHAND = True
LOG_FMTS = 'stdout,csv'


def all_bools(vals):
    return all([isinstance(v,bool) for v in vals])

def valid_str(v):
    """
    Convert a value or values to a string which could go in a filepath.

    Partly based on `this gist`_.

    .. _`this gist`: https://gist.github.com/seanh/93666

    """
    if hasattr(v, '__name__'):
        return valid_str(v.__name__)

    if isinstance(v, tuple) or isinstance(v, list):
        return '-'.join([valid_str(x) for x in v])

    # Valid characters are '-', '_', and alphanumeric. Replace invalid chars
    # with '-'.
    str_v = str(v).lower()
    valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)
    str_v = ''.join(c if c in valid_chars else '-' for c in str_v)
    return str_v


def create_experiment(exp_name, thunk, seed=42, log_dir=None, format_strs=None,
                      datestamp=None, **kwargs):
    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])

    # Make a seed-specific subfolder in the experiment directory.
    if datestamp:
        hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
    else:
        subfolder = ''.join([exp_name, '_s', str(seed)])
    relpath = osp.join(relpath, subfolder)
    log_dir = log_dir or 'data'
    log_dir = osp.join(log_dir, relpath)

    def thunk_plus():
        from baselines import logger
        from proj.utils.tqdm_util import tqdm_out
        from proj.common.utils import set_global_seeds
        from proj.common.log_utils import save_config
        from proj.common.env_makers import VecEnvMaker

        set_global_seeds(seed)

        if 'env' in kwargs:
            kwargs['env_maker'] = VecEnvMaker(kwargs['env'])
            del kwargs['env']

        with tqdm_out(), logger.scoped_configure(log_dir, format_strs):
            logger.set_level(logger.WARN)
            save_config({'exp_name': exp_name})
            thunk(**kwargs)

    return thunk_plus


class ExperimentGrid(object):
    """
    Tool for running many experiments given hyperparameter ranges.
    """

    def __init__(self, name=''):
        self.keys = []
        self.vals = []
        self.shs = []
        self.in_names = []
        self.name(name)

    def name(self, _name):
        assert isinstance(_name, str), "Name has to be a string."
        self._name = _name

    def print(self):
        """Print a helpful report about the experiment grid."""
        print('='*DIV_LINE_WIDTH)

        # Prepare announcement at top of printing. If the ExperimentGrid has a
        # short name, write this as one line. If the name is long, break the
        # announcement over two lines.
        base_msg = 'ExperimentGrid %s runs over parameters:\n'
        name_insert = '['+self._name+']'
        if len(base_msg%name_insert) <= 80:
            msg = base_msg%name_insert
        else:
            msg = base_msg%(name_insert+'\n')
        print(msg)

        # List off parameters, shorthands, and possible values.
        for k, v, sh in zip(self.keys, self.vals, self.shs):
            color_k = k.ljust(40)
            print('', color_k, '['+sh+']' if sh is not None else '', '\n')
            for i, val in enumerate(v):
                print('\t' + str(convert_json(val)))
            print()

        # Count up the number of variants. The number counting seeds
        # is the total number of experiments that will run; the number not
        # counting seeds is the total number of otherwise-unique configs
        # being investigated.
        nvars_total = 1
        for v in self.vals:
            nvars_total *= len(v)
        if 'seed' in self.keys:
            num_seeds = len(self.vals[self.keys.index('seed')])
            nvars_seedless = int(nvars_total / num_seeds)
        else:
            nvars_seedless = nvars_total
        print(' Variants, counting seeds: '.ljust(40), nvars_total)
        print(' Variants, not counting seeds: '.ljust(40), nvars_seedless)
        print()
        print('='*DIV_LINE_WIDTH)


    def _default_shorthand(self, key):
        # Create a default shorthand for the key, built from the first
        # three letters of each colon-separated part.
        # But if the first three letters contains something which isn't
        # alphanumeric, shear that off.
        valid_chars = "%s%s" % (string.ascii_letters, string.digits)
        def shear(x):
            return ''.join(z for z in x[:3] if z in valid_chars)
        sh = '-'.join([shear(x) for x in key.split(':')])
        return sh

    def add(self, key, vals, shorthand=None, in_name=False):
        """
        Add a parameter (key) to the grid config, with potential values (vals).

        By default, if a shorthand isn't given, one is automatically generated
        from the key using the first three letters of each colon-separated
        term. To disable this behavior, change ``DEFAULT_SHORTHAND`` in the
        ``spinup/user_config.py`` file to ``False``.

        Args:
            key (string): Name of parameter.

            vals (value or list of values): Allowed values of parameter.

            shorthand (string): Optional, shortened name of parameter. For
                example, maybe the parameter ``steps_per_epoch`` is shortened
                to ``steps``.

            in_name (bool): When constructing variant names, force the
                inclusion of this parameter into the name.
        """
        assert isinstance(key, str), "Key must be a string."
        assert shorthand is None or isinstance(shorthand, str), \
            "Shorthand must be a string."
        if not isinstance(vals, list):
            vals = [vals]
        if DEFAULT_SHORTHAND and shorthand is None:
            shorthand = self._default_shorthand(key)
        self.keys.append(key)
        self.vals.append(vals)
        self.shs.append(shorthand)
        self.in_names.append(in_name)

    def variant_name(self, variant):
        """
        Given a variant (dict of valid param/value pairs), make an exp_name.

        A variant's name is constructed as the grid name (if you've given it
        one), plus param names (or shorthands if available) and values
        separated by underscores.

        Note: if ``seed`` is a parameter, it is not included in the name.
        """

        def get_val(v, k):
            # Utility method for getting the correct value out of a variant
            # given as a nested dict. Assumes that a parameter name, k,
            # describes a path into the nested dict, such that k='a:b:c'
            # corresponds to value=variant['a']['b']['c']. Uses recursion
            # to get this.
            if k in v:
                return v[k]
            else:
                splits = k.split(':')
                k0, k1 = splits[0], ':'.join(splits[1:])
                return get_val(v[k0], k1)

        # Start the name off with the name of the variant generator.
        var_name = self._name

        # Build the rest of the name by looping through all parameters,
        # and deciding which ones need to go in there.
        for k, v, sh, inn in zip(self.keys, self.vals, self.shs, self.in_names):

            # Include a parameter in a name if either 1) it can take multiple
            # values, or 2) the user specified that it must appear in the name.
            # Except, however, when the parameter is 'seed'. Seed is handled
            # differently so that runs of the same experiment, with different
            # seeds, will be grouped by experiment name.
            if (len(v)>1 or inn) and not(k=='seed'):

                # Use the shorthand if available, otherwise the full name.
                param_name = sh if sh is not None else k
                param_name = valid_str(param_name)

                # Get variant value for parameter k
                variant_val = get_val(variant, k)

                # Append to name
                if all_bools(v):
                    # If this is a param which only takes boolean values,
                    # only include in the name if it's True for this variant.
                    var_name += ('_' + param_name) if variant_val else ''
                else:
                    var_name += '_' + param_name + valid_str(variant_val)

        return var_name.lstrip('_')

    def _variants(self, keys, vals):
        """
        Recursively builds list of valid variants.
        """
        if len(keys)==1:
            pre_variants = [dict()]
        else:
            pre_variants = self._variants(keys[1:], vals[1:])

        variants = []
        for val in vals[0]:
            for pre_v in pre_variants:
                v = {}
                v[keys[0]] = val
                v.update(pre_v)
                variants.append(v)
        return variants

    def variants(self):
        """
        Makes a list of dicts, where each dict is a valid config in the grid.

        There is special handling for variant parameters whose names take
        the form

            ``'full:param:name'``.

        The colons are taken to indicate that these parameters should
        have a nested dict structure. eg, if there are two params,

            ====================  ===
            Key                   Val
            ====================  ===
            ``'base:param:one'``  1
            ``'base:param:two'``  2
            ====================  ===

        the variant dict will have the structure

        .. parsed-literal::

            variant = {
                base: {
                    param : {
                        a : 1,
                        b : 2
                        }
                    }
                }
        """
        flat_variants = self._variants(self.keys, self.vals)

        def unflatten_var(var):
            """
            Build the full nested dict version of var, based on key names.
            """
            new_var = dict()
            unflatten_set = set()

            for k,v in var.items():
                if ':' in k:
                    splits = k.split(':')
                    k0 = splits[0]
                    assert k0 not in new_var or isinstance(new_var[k0], dict), \
                        "You can't assign multiple values to the same key."

                    if not(k0 in new_var):
                        new_var[k0] = dict()

                    sub_k = ':'.join(splits[1:])
                    new_var[k0][sub_k] = v
                    unflatten_set.add(k0)
                else:
                    assert not(k in new_var), \
                        "You can't assign multiple values to the same key."
                    new_var[k] = v

            # Make sure to fill out the nested dicts.
            for k in unflatten_set:
                new_var[k] = unflatten_var(new_var[k])

            return new_var

        new_variants = [unflatten_var(var) for var in flat_variants]
        return new_variants

    def run(self, thunk, log_dir=None, format_strs=LOG_FMTS, datestamp=False):
        """
        Run each variant in the grid with function 'thunk'.

        Note: 'thunk' must be either a callable function, or a string. If it is
        a string, it must be the name of a parameter whose values are all
        callable functions.

        Uses ``call_experiment`` to actually launch each experiment, and gives
        each variant a name using ``self.variant_name()``.

        Maintenance note: the args for ExperimentGrid.run should track closely
        to the args for call_experiment. However, ``seed`` is omitted because
        we presume the user may add it as a parameter in the grid.
        """

        # Print info about self.
        self.print()

        # Make the list of all variants.
        variants = self.variants()

        # Print variant names for the user.
        var_names = OrderedDict.fromkeys(map(self.variant_name, variants))
        var_names = list(var_names.keys())
        line = '='*DIV_LINE_WIDTH
        preparing = 'Preparing to run the following experiments...'
        joined_var_names = '\n'.join(var_names)
        announcement = f"\n{preparing}\n\n{joined_var_names}\n\n{line}"
        print(announcement)


        # Run the variants.
        for var in variants:
            exp_name = self.variant_name(var)
            print("Running experiment:", exp_name)

            thunk_plus = create_experiment(
                exp_name, thunk, log_dir=log_dir, datestamp=datestamp,
                format_strs=format_strs.split(','), **var
            )
            # Prepare to launch a script to run the experiment
            pickled_thunk = cloudpickle.dumps(thunk_plus)
            encoded_thunk = base64.b64encode(zlib.compress(pickled_thunk)).decode('utf-8')

            entrypoint = osp.join(osp.abspath(osp.dirname(__file__)),'run_entrypoint.py')
            cmd = [sys.executable if sys.executable else 'python', entrypoint, encoded_thunk]
            try:
                subprocess.check_call(cmd, env=os.environ)
            except CalledProcessError:
                err_msg = '\n'*3 + '='*DIV_LINE_WIDTH + '\n' + dedent("""

                There appears to have been an error in your experiment.

                Check the traceback above to see what actually went wrong. The
                traceback below, included for completeness (but probably not useful
                for diagnosing the error), shows the stack leading up to the
                experiment launch.

                """) + '='*DIV_LINE_WIDTH + '\n'*3
                print(err_msg)
                raise
