import shutil
from pathlib import Path
from typing import Union

from lndb import settings
from lndb.dev.upath import UPath

from .._file import ProgressCallback


class StorageObject:
    delete_registry: dict = {}
    upload_registry: dict = {}
    download_registry: dict = {}

    def __init__(self, storagekey: str, localpath: Union[str, Path] = None):
        self.storage_path = settings.storage.key_to_filepath(storagekey)
        self.is_remote = isinstance(self.storage_path, UPath)

        self.local_path: Path = localpath  # type: ignore
        if self.local_path is None:
            if self.is_remote:
                self.local_path = settings.storage.cloud_to_local_no_update(
                    self.storage_path
                )
        else:
            self.local_path = Path(self.local_path)

        if self.storage_path.exists():
            check_path = self.storage_path
        else:
            check_path = self.local_path

        if check_path.is_file():
            self.store_type = "file"
        elif check_path.is_dir():
            self.store_type = "dir"
        else:
            FileNotFoundError(f"{check_path} is not an existing path!")

        self.size = None
        if self.local_path.exists():
            if self.store_type == "file":
                self.size = self.local_path.stat().st_size
            else:
                self.size = sum(
                    f.stat().st_size for f in self.local_path.rglob("*") if f.is_file()
                )

    def delete(self):
        self.delete_registry[(self.is_remote, self.store_type)](self)

    def upload(self):
        self.upload_registry[(self.is_remote, self.store_type)](self)
        return self.size

    def download(self):
        self.download_registry[(self.is_remote, self.store_type)](self)
        return self.local_path


def _delete_file(obj: StorageObject):
    if obj.storage_path.exists():
        obj.storage_path.unlink()
    if obj.local_path.exists():
        obj.local_path.unlink()


StorageObject.delete_registry[(False, "file")] = _delete_file
StorageObject.delete_registry[(True, "file")] = _delete_file


def _delete_local_dir(obj: StorageObject):
    if obj.storage_path.exists():
        shutil.rmtree(obj.storage_path)
    if obj.local_path.exists():
        shutil.rmtree(obj.local_path)


StorageObject.delete_registry[(False, "dir")] = _delete_local_dir


def _delete_remote_dir(obj: StorageObject):
    if obj.storage_path.exists():
        obj.storage_path.rmdir()
    if obj.local_path.exists():
        shutil.rmtree(obj.local_path)


StorageObject.delete_registry[(True, "dir")] = _delete_remote_dir


def _upload_remote(obj: StorageObject):
    cb = ProgressCallback()
    obj.storage_path.upload_from(obj.local_path, recursive=True, callback=cb)


StorageObject.upload_registry[(True, "file")] = _upload_remote
StorageObject.upload_registry[(True, "dir")] = _upload_remote


def _copy_local_file(obj: StorageObject):
    try:
        shutil.copyfile(obj.local_path, obj.storage_path)
    except shutil.SameFileError:
        pass


StorageObject.upload_registry[(False, "file")] = _copy_local_file


def _copy_local_dir(obj: StorageObject):
    if obj.storage_path.exists():
        shutil.rmtree(obj.storage_path)
    shutil.copytree(obj.local_path, obj.storage_path)


StorageObject.upload_registry[(False, "dir")] = _copy_local_dir


def _download_remote_file(obj: StorageObject):
    obj.local_path.parent.mkdir(parents=True, exist_ok=True)
    obj.storage_path.synchronize(obj.local_path)


StorageObject.upload_registry[(True, "file")] = _download_remote_file


def _download_remote_dir(obj: StorageObject):
    obj.local_path.parent.mkdir(parents=True, exist_ok=True)
    obj.storage_path.download_to(obj.local_path, recursive=True)


StorageObject.upload_registry[(True, "dir")] = _download_remote_dir
