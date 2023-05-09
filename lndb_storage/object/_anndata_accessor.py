from functools import cached_property
from typing import Union

import h5py
import pandas as pd
import zarr
from anndata._core.index import Index, _normalize_indices
from anndata._core.sparse_dataset import SparseDataset
from anndata._core.views import _resolve_idx
from anndata._io.specs.methods import read_indices
from anndata._io.specs.registry import get_spec, read_elem, read_elem_partial
from lndb.dev.upath import infer_filesystem as _infer_filesystem
from lnschema_core import File
from lnschema_core.dev._storage import filepath_from_file

from ._subset_anndata import _read_dataframe


def _try_backed_full(elem):
    # think what to do for compatibility with old var and obs
    if isinstance(elem, (h5py.Dataset, zarr.Array)):
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
            return _read_dataframe(self.storage["obs"])

    @cached_property
    def var(self) -> pd.DataFrame:
        indices = getattr(self, "indices", None)
        if indices is not None:
            indices = (indices[1], slice(None))
            return read_elem_partial(self.storage["obs"], indices=indices)
        else:
            return _read_dataframe(self.storage["obs"])

    @cached_property
    def uns(self):
        return read_elem(self.storage["uns"])

    @cached_property
    def X(self):
        indices = getattr(self, "indices", None)
        if indices is not None:
            return read_elem_partial(self.storage["X"], indices=indices)
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

    @property
    def obs_names(self):
        return self._obs_names

    @property
    def var_names(self):
        return self._var_names

    @cached_property
    def shape(self):
        return len(self._obs_names), len(self._var_names)


class AnnDataAccessorSubset(_AnnDataAttrsMixin):
    def __init__(self, storage, indices, obs_names, var_names, ref_shape):
        self.storage = storage
        self.indices = indices

        self._obs_names, self._var_names = obs_names, var_names

        self._ref_shape = ref_shape

    def __getitem__(self, index: Index):
        """Access a subset of the underlying AnnData object."""
        oidx, vidx = _normalize_indices(index, self._obs_names, self._var_names)
        new_obs_names, new_var_names = self._obs_names[oidx], self._var_names[vidx]
        oidx = _resolve_idx(self.indices[0], oidx, self._ref_shape[0])
        vidx = _resolve_idx(self.indices[1], vidx, self._ref_shape[1])
        return AnnDataAccessorSubset(
            self.storage, (oidx, vidx), new_obs_names, new_var_names, self._ref_shape
        )


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

        self._name = file.name

        self._obs_names, self._var_names = read_indices(self.storage)

    def __del__(self):
        """Closes the connection."""
        if self._conn is not None:
            self.storage.close()
            self._conn.close()

    def __getitem__(self, index: Index) -> AnnDataAccessorSubset:
        """Access a subset of the underlying AnnData object."""
        oidx, vidx = _normalize_indices(index, self._obs_names, self._var_names)
        new_obs_names, new_var_names = self._obs_names[oidx], self._var_names[vidx]
        return AnnDataAccessorSubset(
            self.storage, (oidx, vidx), new_obs_names, new_var_names, self.shape
        )

    def __repr__(self):
        """Description of the AnnDataAccessor object."""
        n_obs, n_vars = self.shape
        descr = f"AnnDataAccessor object with n_obs × n_vars = {n_obs} × {n_vars}"
        descr += f"\n  constructed for the AnnData object {self._name}"
        for attr in self.storage.keys():
            if attr == "X":
                continue
            attr_obj = self.storage[attr]
            if attr in ("obs", "var") and isinstance(
                attr_obj, (h5py.Dataset, zarr.Array)
            ):
                keys = list(attr_obj.dtype.fields.keys())
            else:
                keys = list(attr_obj.keys())
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(keys)}"
        return descr
