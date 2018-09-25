import os, cloudpickle


# ==============================
# Saving snapshots
# ==============================

class SnapshotSaver(object):
    def __init__(self, path, interval=1, latest_only=None):
        self.path = path
        self.interval = interval
        if latest_only is None:
            latest_only = True
            snapshots_folder = os.path.join(path, "snapshots")
            if os.path.exists(snapshots_folder):
                if os.path.exists(os.path.join(snapshots_folder, "latest.pkl")):
                    latest_only = True
                elif len(os.listdir(snapshots_folder)) > 0:
                    latest_only = False
        self.latest_only = latest_only

    @property
    def snapshots_folder(self):
        return os.path.join(self.path, "snapshots")

    def get_snapshot_path(self, index):
        if self.latest_only:
            return os.path.join(self.snapshots_folder, "latest.pkl")
        else:
            return os.path.join(self.snapshots_folder, "%d.pkl" % index)

    def save_state(self, index, state):
        if index % self.interval == 0:
            file_path = self.get_snapshot_path(index)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                cloudpickle.dump(state, f, protocol=-1)

    def get_state(self, index=None):
        if self.latest_only:
            try:
                with open(self.get_snapshot_path(0), "rb") as f:
                    return cloudpickle.load(f)
            except EOFError:
                pass
        elif index is not None:
            try:
                with open(self.get_snapshot_path(index), "rb") as f:
                    return cloudpickle.load(f)
            except EOFError:
                pass
        else:
            snapshot_files = os.listdir(self.snapshots_folder)
            snapshot_files = sorted(
                snapshot_files, key=lambda x: int(x.split(".")[0]))[::-1]
            for file in snapshot_files:
                file_path = os.path.join(self.snapshots_folder, file)
                try:
                    with open(file_path, "rb") as f:
                        return cloudpickle.load(f)
                except EOFError:
                    pass

