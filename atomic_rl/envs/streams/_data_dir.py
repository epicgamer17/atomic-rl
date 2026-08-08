"""Shared cache directory resolution for stream environments.

Downloaded datasets live in a per-user cache dir (never inside the package or
repository), following the XDG cache convention.
"""

import os


def get_default_data_dir(subdir: str = "env_data") -> str:
    """Return a user cache dir for downloaded data.

    Respects ``XDG_CACHE_HOME`` when set, otherwise falls back to
    ``~/.cache``.  The returned path is ``<cache>/atomic-rl/<subdir>``.
    """
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "atomic-rl", subdir)
