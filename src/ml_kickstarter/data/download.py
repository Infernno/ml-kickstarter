from pathlib import Path

from dvc.repo import Repo


def ensure_data(raw_path: str | Path, remote: str) -> None:
    """Restore the tracked raw dataset from DVC when it is missing."""
    data_path = Path(raw_path)
    if data_path.is_file():
        return

    with Repo() as repo:
        repo.pull(targets=[str(data_path)], remote=remote)
