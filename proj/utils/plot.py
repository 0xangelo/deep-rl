import itertools
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from math import sqrt
from itertools import chain
from functools import reduce
from collections import OrderedDict


matplotlib.rcParams.update(
    {
        "backend": "ps",
        "text.latex.preamble": ["\\usepackage{gensymb}"],
        "text.usetex": True,
    }
)


def latexify(fig_width=None, fig_height=None, columns=1):
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

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print(
            "WARNING: fig_height too large:"
            + fig_height
            + "so will reduce to"
            + MAX_HEIGHT_INCHES
            + "inches."
        )
        fig_height = MAX_HEIGHT_INCHES

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

    ds = []
    with open(progress_path, "rt") as fh:
        for line in fh:
            ds.append(json.loads(line))
    return pd.DataFrame(ds)


def flatten_dict(d):
    flat_params = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            v = flatten_dict(v)
            for subk, subv in flatten_dict(v).items():
                flat_params[k + "." + subk] = subv
        else:
            flat_params[k] = v
    return flat_params


def load_params(params_json_path):
    with open(params_json_path, "r") as f:
        data = json.load(f)
        if "args_data" in data:
            del data["args_data"]
        if "exp_name" not in data:
            data["exp_name"] = params_json_path.split("/")[-3]
    return data


def first_that(criterion, l):
    return next(x for x in l if criterion(x))


def load_exps_data(
    exp_folder_paths,
    ignore_missing_keys=False,
    verbose=False,
    isprogress=lambda x: x.startswith("progress"),
    isconfig=lambda x: x.startswith("variant"),
):
    if isinstance(exp_folder_paths, str):
        exp_folder_paths = [exp_folder_paths]
    exps = list(
        filter(  # entries that have progress files
            lambda x: any(isprogress(y) for y in x[1]),
            filter(  # entries that have >0 files
                lambda x: x[1],
                map(  # only (path, files)
                    lambda x: (x[0], x[2]),
                    reduce(  # (path, subpath, files) for all dirs
                        chain,
                        map(  # (path, subpath, files) for each dir
                            os.walk, exp_folder_paths
                        ),
                    ),
                ),
            ),
        )
    )
    if verbose:
        print("finished walking exp folders")
    exps_data = []
    for exp_path, files in exps:
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
        except (IOError, pd.errors.EmptyDataError) as e:
            if verbose:
                print(e)

    # a dictionary of all keys and types of values
    all_keys = dict()
    for data in exps_data:
        for key in data.flat_params.keys():
            if key not in all_keys:
                all_keys[key] = type(data.flat_params[key])

    # if any data does not have some key, specify the value of it
    if not ignore_missing_keys:
        default_values = dict()
        for data in exps_data:
            for key in sorted(all_keys.keys()):
                if key not in data.flat_params:
                    if key not in default_values:
                        default = None
                        default_values[key] = default
                    data.flat_params[key] = default_values[key]

    return exps_data


def smart_repr(x):
    if isinstance(x, tuple):
        if len(x) == 0:
            return "tuple()"
        elif len(x) == 1:
            return "(%s,)" % smart_repr(x[0])
        else:
            return "(" + ",".join(map(smart_repr, x)) + ")"
    else:
        if callable(x):
            return "__import__('pydoc').locate('%s')" % (
                x.__module__ + "." + x.__name__
            )
        else:
            return repr(x)


def extract_distinct_params(exps_data, excluded_params=("seed", "log_dir")):
    try:
        repr_config_pairs = [
            smart_repr(kv) for d in exps_data for kv in d.flat_params.items()
        ]
        uniq_pairs = list(set(repr_config_pairs))
        evald_pairs = map(eval, uniq_pairs)
        stringified_pairs = sorted(
            evald_pairs, key=lambda x: tuple("" if it is None else str(it) for it in x)
        )
    except Exception as e:
        print(e)
        import ipdb

        ipdb.set_trace()

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


class Selector(object):
    def __init__(self, exps_data, filters=None):
        self._exps_data = exps_data
        self._filters = tuple() if filters is None else tuple(filters)

    def where(self, k, v):
        return Selector(
            self._exps_data,
            self._filters + (lambda exp: str(exp.flat_params.get(k, None)) == str(v),),
        )

    def where_not(self, k, v):
        return Selector(
            self._exps_data,
            self._filters + (lambda exp: str(exp.flat_params.get(k, None)) != str(v),),
        )

    def _check_exp(self, exp):
        return all(condition(exp) for condition in self._filters)

    def extract(self):
        return list(filter(self._check_exp, self._exps_data))


def lineplot_instructions(
    exps_data,
    x,
    y,
    hue=None,
    size=None,
    style=None,
    estimator="mean",
    split=None,
    include=None,
    exclude=None,
):
    plot_kwargs = dict(
        x=x, y=y, hue=hue, size=size, style=style, estimator=estimator, ci="sd"
    )
    if estimator is None:
        plot_kwargs["units"] = "unit"

    selector = Selector(exps_data)
    include, exclude = include or {}, exclude or {}
    for k, v in include.items():
        selector = selector.where(k, str(v))
    for k, v in exclude.items():
        selector = selector.where_not(k, str(v))

    if split is not None:
        vs = dict(sorted(extract_distinct_params(exps_data))).get(split, [])
        split_selectors = [selector.where(split, v) for v in vs]
        split_titles = list(map(str, vs))
    else:
        split_selectors = [selector]
        split_titles = ["Experiment"]

    plots = []
    keys = tuple(filter(None, (hue, size, style))) + ("unit",)
    for split_selector, split_title in zip(split_selectors, split_titles):
        split_exps_data = split_selector.extract()
        if len(split_exps_data) < 1:
            continue

        distinct_params = OrderedDict(sorted(extract_distinct_params(split_exps_data)))
        key_orders = dict(
            hue_order=distinct_params.get(hue, None),
            size_order=distinct_params.get(size, None),
            style_order=distinct_params.get(style, None),
        )
        lineplot_kwargs = AttrDict({**plot_kwargs, **key_orders})

        dataframes = []
        configs_and_data = ((exp.flat_params, exp.progress) for exp in split_exps_data)
        for enum, (config, dataframe) in enumerate(configs_and_data):
            config["unit"] = enum
            for key in filter(lambda key: key not in dataframe, keys):
                dataframe.insert(len(dataframe.columns), str(key), config[key])
            dataframes.append(dataframe)
        lineplot_kwargs.data = pd.concat(dataframes, ignore_index=True, sort=False)

        plots.append(AttrDict(title=split_title, lineplot_kwargs=lineplot_kwargs))
    return plots


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", nargs="*")
    parser.add_argument("--xaxis", "-x", default="TotalNSamples")
    parser.add_argument("--yaxis", "-y", default="AverageReturn")
    parser.add_argument("--hue", default=None)
    parser.add_argument("--size", default=None)
    parser.add_argument("--style", default=None)
    parser.add_argument("--est", default="mean")
    parser.add_argument("--split", default=None)
    parser.add_argument("--include", nargs="*")
    parser.add_argument("--exclude", nargs="*")
    args = parser.parse_args()

    include = (
        {} if args.include is None else dict(pair.split(":") for pair in args.include)
    )
    exclude = (
        {} if args.exclude is None else dict(pair.split(":") for pair in args.exclude)
    )

    exps_data = load_exps_data(args.logdir)
    plot_instructions = lineplot_instructions(
        exps_data,
        x=args.xaxis,
        y=args.yaxis,
        hue=args.hue,
        size=args.size,
        style=args.style,
        estimator=args.est,
        split=args.split,
        include=include,
        exclude=exclude,
    )

    with sns.plotting_context("paper"), sns.axes_style("darkgrid"), latexify(columns=2):
        for plot_inst in plot_instructions:
            plt.figure()
            sns.lineplot(legend="full", **plot_inst.lineplot_kwargs)
            plt.title(plot_inst.title)
            xscale = (
                np.max(np.asarray(plot_inst.lineplot_kwargs.data[args.xaxis])) > 5e3
            )
            if xscale:
                # Just some formatting niceness:
                # x-axis scale in scientific notation if max x is large
                plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            plt.tight_layout(pad=0.5)
        plt.show()


if __name__ == "__main__":
    main()
