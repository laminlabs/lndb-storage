from datetime import datetime
from os import PathLike, utime
from pathlib import Path, PurePosixPath

import fsspec
from dateutil.parser import isoparse  # type: ignore


class FsspecPath(PathLike):
    def __init__(self, actual_path: str, cache_dir: str):
        self._str = actual_path
        self.protocol = fsspec.utils.get_protocol(self._str)
        self._fs = fsspec.filesystem(self.protocol)
        self._path = PurePosixPath(self._fs._strip_protocol(self._str))
        self._cache_dir = cache_dir
        # what if the name is the same?
        self._cache = Path(cache_dir) / self._path.name

    def _modified(self):
        try:
            mtime = self._fs.modified(self._str)
        except NotImplementedError:
            # todo: check more protocols
            # here only for gs
            mtime = self._fs.stat(self._str)["updated"]
            mtime = isoparse(mtime)
        return datetime.timestamp(mtime)

    def _refresh_cache(self, force_overwrite_from_cloud=False):
        if not self._fs.exists(self._str):
            return None

        actual_mtime = self._modified()

        if (
            not self._cache.exists()
            or actual_mtime > self._cache.stat().st_mtime  # noqa: W503
            or force_overwrite_from_cloud  # noqa: W503
        ):
            self._cache.parent.mkdir(parents=True, exist_ok=True)
            self._fs.get(self._str, str(self._cache))
            utime(str(self._cache), times=(actual_mtime, actual_mtime))

        if actual_mtime < self._cache.stat().st_mtime:
            raise Exception("Cache is newer than actual.")

    def is_file(self):
        return self._fs.isfile(self._str)

    def __fspath__(self):
        """Return the file system path representation of the object."""
        if self.is_file():
            self._refresh_cache(force_overwrite_from_cloud=False)
        return str(self._cache)

    @property
    def fspath(self):
        """Return the file system path representation of the object."""
        return self.__fspath__()

    def __truediv__(self, other):
        """Form a new path."""
        cache_dir = self._cache_dir
        actual_path = str(self._path / other)
        actual_path = fsspec._utils._unstrip_protocol(actual_path, self._fs)
        return FsspecPath(actual_path, cache_dir)
