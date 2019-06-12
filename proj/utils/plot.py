import os
import json
import itertools
from math import sqrt
from functools import reduce
from contextlib import nullcontext
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def latexify(fig_width=None, fig_height=None, columns=1, max_height_inches=8.0):
    """Return matplotlib's RC params for LaTeX plotting.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert columns in [1, 2]

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    if fig_height > max_height_inches:
        print(
            "WARNING: fig_height too large:"
            + fig_height
            + "so will reduce to"
            + max_height_inches
            + "inches."
        )
        fig_height = max_height_inches

    new_params = {
        "axes.labelsize": 8,  # fontsize for x and y labels (was 10)
        "axes.titlesize": 8,
        "legend.fontsize": 8,  # was 10
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": [fig_width, fig_height],
        "font.family": ["serif"],
    }
    return matplotlib.rc_context(rc=new_params)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_progress(progress_path, verbose=True):
    if verbose:
        print("Reading %s" % progress_path)

    if progress_path.endswith(".csv"):
        return pd.read_csv(progress_path, index_col=None, comment="#")

    dicts = []
    with open(progress_path, "rt") as file:
        for line in file:
            dicts.append(json.loads(line))
    return pd.DataFrame(dicts)


def flatten_dict(dic):
    flat_params = dict()
    for key, val in dic.items():
        if isinstance(val, dict):
            val = flatten_dict(val)
            for subk, subv in flatten_dict(val).items():
                flat_params[key + "." + subk] = subv
        else:
            flat_params[key] = val
    return flat_params


def load_params(params_json_path):
    with open(params_json_path, "r") as file:
        data = json.load(file)
        if "args_data" in data:
            del data["args_data"]
        if "exp_name" not in data:
            data["exp_name"] = params_json_path.split("/")[-3]
    return data


def first_that(criterion, lis):
    return next(x for x in lis if criterion(x))


def exp_folder_paths(directories, isprogress):
    return list(
        filter(  # entries that have progress files
            lambda x: any(isprogress(y) for y in x[1]),
            filter(  # entries that have >0 files
                lambda x: x[1],
                map(  # only (path, files)
                    lambda x: (x[0], x[2]),
                    reduce(  # (path, subpath, files) for all dirs
                        itertools.chain,
                        map(  # (path, subpath, files) for each dir
                            os.walk, directories
                        ),
                    ),
                ),
            ),
        )
    )


def read_exp_path_data(exp_paths, isprogress, isconfig, verbose=False):
    exps_data = []
    for exp_path, files in exp_paths:
        try:
            progress_path = os.path.join(exp_path, first_that(isprogress, files))
            progress = load_progress(progress_path, verbose=verbose)
            if any(isconfig(file) for file in files):
                params = load_params(
                    os.path.join(exp_path, first_that(isconfig, files))
                )
            else:
                params = dict(exp_name="experiment")
            exps_data.append(
                AttrDict(
                    progress=progress, params=params, flat_params=flatten_dict(params)
                )
            )
        except (IOError, pd.errors.EmptyDataError) as error:
            if verbose:
                print(error)

    return exps_data


def load_exps_data(
    directories,
    progress_prefix="progress",
    config_prefix="variant",
    ignore_missing_keys=True,
    verbose=False,
):
    if isinstance(directories, str):
        directories = [directories]

    def isprogress(file):
        return file.startswith(progress_prefix)

    def isconfig(file):
        return file.startswith(config_prefix)

    exp_paths = exp_folder_paths(directories, isprogress)
    if verbose:
        print("finished walking exp folders")

    exps_data = read_exp_path_data(exp_paths, isprogress, isconfig, verbose=verbose)
    for num, exp_data in enumerate(exps_data):
        exp_data.progress.insert(len(exp_data.progress.columns), "unit", num)

    # if any data does not have some key, specify the value of it
    if not ignore_missing_keys:
        # a dictionary of all keys and types of values
        all_keys = {
            key: type(data.flat_params[key])
            for data in exps_data
            for key in data.flat_params.keys()
        }

        default_values = dict()
        for data in exps_data:
            for key in sorted(all_keys.keys()):
                if key not in data.flat_params:
                    if key not in default_values:
                        default_values[key] = None
                    data.flat_params[key] = default_values[key]

    return exps_data


def smart_repr(obj):
    if isinstance(obj, tuple):
        if not obj:
            return "tuple()"
        if len(obj) == 1:
            return "(%s,)" % smart_repr(obj[0])
        return "(" + ",".join(map(smart_repr, obj)) + ")"
    if callable(obj):
        return "__import__('pydoc').locate('%s')" % (
            obj.__module__ + "." + obj.__name__
        )
    return repr(obj)


def extract_distinct_params(exps_data, excluded_params=("seed", "log_dir")):
    repr_config_pairs = [
        smart_repr(kv) for d in exps_data for kv in d.flat_params.items()
    ]
    uniq_pairs = list(set(repr_config_pairs))
    evald_pairs = map(eval, uniq_pairs)
    stringified_pairs = sorted(
        evald_pairs, key=lambda x: tuple("" if it is None else str(it) for it in x)
    )

    proposals = [
        (k, [x[1] for x in v])
        for k, v in itertools.groupby(stringified_pairs, lambda x: x[0])
    ]

    filtered = [
        (k, v)
        for (k, v) in proposals
        if len(v) > 1
        and all([k.find(excluded_param) != 0 for excluded_param in excluded_params])
    ]
    return filtered


class Selector:
    def __init__(self, exps_data, filters=None):
        self._exps_data = exps_data
        self._filters = tuple() if filters is None else tuple(filters)

    def where(self, key, val):
        return Selector(
            self._exps_data,
            self._filters
            + (lambda exp: str(exp.flat_params.get(key, None)) == str(val),),
        )

    def where_not(self, key, val):
        return Selector(
            self._exps_data,
            self._filters
            + (lambda exp: str(exp.flat_params.get(key, None)) != str(val),),
        )

    def _check_exp(self, exp):
        return all(condition(exp) for condition in self._filters)

    def extract(self):
        return list(filter(self._check_exp, self._exps_data))


def lineplot_instructions(
    exps_data, split=None, include=None, exclude=None, **lineplot_kwargs
):
    selector = Selector(exps_data)
    include, exclude = include or [], exclude or []
    for key, val in include:
        selector = selector.where(key, str(val))
    for key, val in exclude:
        selector = selector.where_not(key, str(val))

    if split is not None:
        values = dict(sorted(extract_distinct_params(exps_data))).get(split, [])
        split_selectors = [selector.where(split, v) for v in values]
        split_titles = list(map(str, values))
    else:
        split_selectors = [selector]
        split_titles = ["Experiment"]

    plots = []
    for split_selector, split_title in zip(split_selectors, split_titles):
        split_exps_data = split_selector.extract()
        if not split_exps_data:
            continue

        dataframes = [exp.progress for exp in split_exps_data]
        distinct_params = OrderedDict(sorted(extract_distinct_params(split_exps_data)))
        split_kwargs = dict(
            data=pd.concat(dataframes, ignore_index=True, sort=False),
            hue_order=distinct_params.get(lineplot_kwargs.get("hue")),
            size_order=distinct_params.get(lineplot_kwargs.get("size")),
            style_order=distinct_params.get(lineplot_kwargs.get("style")),
        )

        plots.append(
            AttrDict(
                title=split_title,
                lineplot_kwargs=AttrDict({**lineplot_kwargs, **split_kwargs}),
            )
        )
    return plots


def format_latex_dict(dictionary):
    new_items = []
    for key, val in dictionary.items():
        new_key, new_val = key, val
        if isinstance(key, str):
            new_key = key.replace("_", "-")
        if isinstance(val, str):
            new_val = val.replace("_", "-")
        new_items.append((new_key, new_val))
    return dict(new_items)


def format_latex_params(exps_data):
    for exp_data in exps_data:
        exp_data.flat_params = format_latex_dict(exp_data.flat_params)
        exp_data.params = format_latex_dict(exp_data.params)


def substitute_key(dictionary, key, subs_key):
    if key in dictionary:
        val = dictionary[key]
        del dictionary[key]
        dictionary[subs_key] = val


def substitute_val(dictionary, val, subs_val):
    for key in dictionary.keys():
        if dictionary[key] == val:
            dictionary[key] = subs_val


def rename_params(exps_data, subskey=None, subsval=None):
    subskey = subskey or []
    subsval = subsval or []
    for exp_data in exps_data:
        for key, subs_key in subskey:
            substitute_key(exp_data.flat_params, key, subs_key)
            substitute_key(exp_data.params, key, subs_key)
        for val, subs_val in subsval:
            substitute_val(exp_data.flat_params, val, subs_val)
            substitute_val(exp_data.params, val, subs_val)


def insert_params_dataframe(exps_data, *param_keys):
    for exp_data in exps_data:
        params, dataframe = exp_data.flat_params, exp_data.progress
        for key in filter(None, param_keys):
            dataframe.insert(len(dataframe.columns), str(key), params[key])


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", nargs="*")
    parser.add_argument("--xaxis", "-x", default="TotalNSamples")
    parser.add_argument("--yaxis", "-y", default="AverageReturn")
    parser.add_argument("--hue", default=None)
    parser.add_argument("--size", default=None)
    parser.add_argument("--style", default=None)
    parser.add_argument("--estimator", default="mean")
    parser.add_argument("--split", default=None)
    parser.add_argument("--include", nargs="*")
    parser.add_argument("--exclude", nargs="*")
    parser.add_argument("--subskey", nargs="*")
    parser.add_argument("--subsval", nargs="*")
    parser.add_argument("--context", default="paper")
    parser.add_argument("--axstyle", default="darkgrid")
    parser.add_argument("--latexcol", type=int, default=2)
    parser.add_argument("--nolatex", action="store_true")
    parser.add_argument("--saveonly", action="store_true")
    parser.add_argument("--fccolor", default="#F5F5F5")  # http://latexcolor.com (Snow)
    args = parser.parse_args()

    if not args.nolatex:
        matplotlib.rcParams.update(
            {
                "backend": "ps",
                "text.latex.preamble": ["\\usepackage{gensymb}"],
                "text.usetex": True,
            }
        )

    include, exclude, subskey, subsval = [
        [pair.split(":") for pair in pairs] if pairs else []
        for pairs in (args.include, args.exclude, args.subskey, args.subsval)
    ]

    exps_data = load_exps_data(args.logdir)
    format_latex_params(exps_data)
    rename_params(exps_data, subskey, subsval)
    insert_params_dataframe(exps_data, args.hue, args.size, args.style)

    plot_instructions = lineplot_instructions(
        exps_data,
        split=args.split,
        include=include,
        exclude=exclude,
        x=args.xaxis,
        y=args.yaxis,
        hue=args.hue,
        size=args.size,
        style=args.style,
        estimator=args.estimator,
        units="unit" if args.estimator is None else None,
    )

    plotting_context = sns.plotting_context(args.context)
    axes_style = sns.axes_style(args.axstyle)
    latex_style = nullcontext() if args.nolatex else latexify(columns=args.latexcol)
    with plotting_context, axes_style, latex_style:
        for idx, plot_inst in enumerate(plot_instructions):
            print("Dataframe for figure {idx}:".format(idx=idx))
            print(plot_inst.lineplot_kwargs.data.head())

            plt.figure(facecolor=args.fccolor)
            plt.title(plot_inst.title)
            sns.lineplot(legend="full", **plot_inst.lineplot_kwargs)

            if np.max(np.asarray(plot_inst.lineplot_kwargs.data[args.xaxis])) > 5e3:
                # Just some formatting niceness:
                # x-axis scale in scientific notation if max x is large
                plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            plt.tight_layout(pad=0.5)

        if args.saveonly:
            plt.savefig("fig.pdf", facecolor=args.fccolor)
        else:
            plt.show()


if __name__ == "__main__":
    main()
