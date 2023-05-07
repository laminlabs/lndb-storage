from functools import cached_property
from typing import Union

import h5py
import pandas as pd
import zarr
from anndata._core.sparse_dataset import SparseDataset
from anndata._io.h5ad import read_dataframe
from anndata._io.spec.registry import get_spec
from anndata._io.specs.registry import read_elem, read_elem_partial
from lndb.dev.upath import infer_filesystem as _infer_filesystem
from lnschema_core import File
from lnschema_core.dev._storage import filepath_from_file


def _try_backed_full(elem):
    # think what to do for compatibility with old var and obs
    if isinstance(elem, (h5py.Dataset, zarr.Dataset)):
        return elem

    if isinstance(elem, (h5py.Group, zarr.Group)):
        encoding_type = get_spec(elem)
        if encoding_type in ("csr_matrix", "csc_matrix"):
            return SparseDataset(elem)
        if "h5sparse_format" in elem.attrs:
            return SparseDataset(elem)

    return read_elem(elem)


class _MapAccessor:
    def __init__(self, elem, indices=None):
        self.elem = elem
        self.indices = indices

    def __getitem__(self, key):
        if self.indices is None:
            return _try_backed_full(self.elem[key])
        else:
            return read_elem_partial(self.elem[key], indices=self.indices)

    def keys(self):
        return self.elem.keys()


class _AnnDataAttrsMixin:
    storage: Union[h5py.File, zarr.Group]

    @cached_property
    def obs(self) -> pd.DataFrame:
        indices = getattr(self, "indices", None)
        if indices is not None:
            indices = (indices[0], slice(None))
            return read_elem_partial(self.storage["obs"], indices=indices)
        else:
            return read_dataframe(self.storage["obs"])

    @cached_property
    def var(self) -> pd.DataFrame:
        indices = getattr(self, "indices", None)
        if indices is not None:
            indices = (indices[1], slice(None))
            return read_elem_partial(self.storage["obs"], indices=indices)
        else:
            return read_dataframe(self.storage["obs"])

    @cached_property
    def uns(self):
        return read_elem(self.storage["uns"])

    @cached_property
    def X(self):
        indices = getattr(self, "indices", None)
        if indices is not None:
            return read_elem_partial(self.storage["var"], indices=indices)
        else:
            return _try_backed_full(self.storage["X"])

    @cached_property
    def obsm(self):
        indices = getattr(self, "indices", None)
        if indices is not None:
            indices = (indices[0], slice(None))
        return _MapAccessor(self.storage["obsm"], indices)

    @cached_property
    def varm(self):
        indices = getattr(self, "indices", None)
        if indices is not None:
            indices = (indices[1], slice(None))
        return _MapAccessor(self.storage["obsm"], indices)

    @cached_property
    def obsp(self):
        indices = getattr(self, "indices", None)
        if indices is not None:
            indices = (indices[0], indices[0])
        return _MapAccessor(self.storage["obsp"], indices)

    @cached_property
    def varp(self):
        indices = getattr(self, "indices", None)
        if indices is not None:
            indices = (indices[1], indices[1])
        return _MapAccessor(self.storage["varp"], indices)

    @cached_property
    def layers(self):
        indices = getattr(self, "indices", None)
        return _MapAccessor(self.storage["layers"], indices)


class AnnDataAccessor(_AnnDataAttrsMixin):
    def __init__(self, file: File):
        fs, file_path_str = _infer_filesystem(filepath_from_file(file))

        if file.suffix == ".h5ad":
            self._conn = fs.open(file_path_str, mode="rb")
            self.storage = h5py.File(self._conn, mode="r")
        elif file.suffix in (".zarr", ".zrad"):
            self._conn = None
            mapper = fs.get_mapper(file_path_str, check=True)
            self.storage = zarr.open(mapper, mode="r")
        else:
            raise ValueError(
                f"file should have .h5ad, .zarr or .zrad suffix, not {file.suffix}."
            )

    def __del__(self):
        """Closes the connection."""
        if self._conn is not None:
            self.storage.close()
            self._conn.close()
