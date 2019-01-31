import os
import math
import click
from baselines import logger


@click.command()
@click.argument("path")
@click.argument("key", type=str)
@click.argument("min_", type=float)
@click.argument("max_", type=float, default=math.inf)
def main(path, key, min_, max_):
    for p in os.listdir(path):
        progress = os.path.join(path, p, 'progress')
        if os.path.exists(progress + '.csv'):
            data = logger.read_csv(progress + '.csv')
        elif os.path.exists(progress + '.json'):
            data = logger.read_json(progress + '.json')
        else:
            continue
        vals = data.get(key, None)
        vals = vals.to_numpy() if vals is not None else None
        if vals is not None and all((vals[-1] >= min_, vals[-1] <= max_)):
            print(os.path.join(path, p))


if __name__ == "__main__":
    main()
