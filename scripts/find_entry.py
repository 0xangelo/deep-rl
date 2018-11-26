import os, click, numpy as np, json

def load_progress(progress_json_path, verbose=True):
    if verbose:
        print("Reading %s" % progress_json_path)
    entries = dict()
    rows = []
    with open(progress_json_path, 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            if len(line) > 0:
                row = json.loads(line)
                rows.append(row)
    all_keys = set(k for row in rows for k in row.keys())
    for k in all_keys:
        if k not in entries:
            entries[k] = []
        for row in rows:
            if k in row:
                v = row[k]
                try:
                    entries[k].append(float(v))
                except:
                    entries[k].append(np.nan)
            else:
                entries[k].append(np.nan)
    entries = dict([(k, np.array(v)) for k, v in entries.items()])
    return entries

@click.command()
@click.argument("path")
@click.argument("key", type=str)
@click.argument("val", type=float)
def main(path, key, val):    
    for p in os.listdir(path):
        progress = os.path.join(path, p, 'progress.json')
        entries = load_progress(progress, verbose=False)
        if entries[key][-1] >= val:
            print(os.path.join(path, p))


if __name__ == "__main__":
    main()
