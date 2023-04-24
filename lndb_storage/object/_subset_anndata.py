from functools import cached_property
from typing import Literal, Optional, Union

import h5py
import pandas as pd
import zarr
from anndata import AnnData
from anndata._io.specs.methods import _read_partial
from anndata._io.specs.registry import read_elem, read_elem_partial
from lndb.dev.upath import infer_filesystem as _infer_filesystem
from lnschema_core import File
from lnschema_core.dev._storage import filepath_from_file

from ._lazy_field import LazySelector


def _indices(base_indices, select_indices):
    if len(base_indices) == len(select_indices):
        return slice(None)
    else:
        return list(base_indices.get_indexer(select_indices))


def _subset_adata_storage(
    storage: Union[zarr.Group, h5py.File],
    query_obs: Optional[Union[str, LazySelector]] = None,
    query_var: Optional[Union[str, LazySelector]] = None,
) -> Union[AnnData, None]:
    with storage as access:
        obs = read_elem(access["obs"])
        var = read_elem(access["var"])

        if query_obs is not None:
            if hasattr(query_obs, "evaluate"):
                obs_result = obs[query_obs.evaluate(obj=obs)]  # type: ignore
            else:
                obs_result = obs.query(query_obs)
        else:
            obs_result = obs

        if query_var is not None:
            if hasattr(query_var, "evaluate"):
                var_result = var[query_var.evaluate(obj=var)]  # type: ignore
            else:
                var_result = var.query(query_var)
        else:
            var_result = var

        if obs_result.index.empty or var_result.index.empty:
            return None

        obs_idx = _indices(obs.index, obs_result.index)
        var_idx = _indices(var.index, var_result.index)

        prepare_adata = {}
        prepare_adata["obs"] = obs_result
        prepare_adata["var"] = var_result
        X = read_elem_partial(access["X"], indices=(obs_idx, var_idx))
        prepare_adata["X"] = X
        if "obsm" in access:
            obsm = _read_partial(access["obsm"], indices=(obs_idx, slice(None)))
            prepare_adata["obsm"] = obsm
        if "varm" in access:
            varm = _read_partial(access["varm"], indices=(var_idx, slice(None)))
            prepare_adata["varm"] = varm
        if "obsp" in access:
            obsp = _read_partial(access["obsp"], indices=(obs_idx, obs_idx))
            prepare_adata["obsp"] = obsp
        if "varp" in access:
            varp = _read_partial(access["varp"], indices=(var_idx, var_idx))
            prepare_adata["varp"] = varp
        if "layers" in access:
            layers = _read_partial(access["layers"], indices=(obs_idx, var_idx))
            prepare_adata["layers"] = layers
        if "uns" in access:
            prepare_adata["uns"] = read_elem(access["uns"])

        return AnnData(**prepare_adata)


def _subset_anndata_file(
    file: File,
    query_obs: Optional[Union[str, LazySelector]] = None,
    query_var: Optional[Union[str, LazySelector]] = None,
) -> Union[AnnData, None]:
    file_path = filepath_from_file(file)
    fs, file_path_str = _infer_filesystem(file_path)

    adata: Union[AnnData, None] = None

    if file.suffix == ".h5ad":
        with fs.open(file_path_str, mode="rb") as file:
            storage = h5py.File(file, mode="r")
            adata = _subset_adata_storage(storage, query_obs, query_var)
    elif file.suffix == ".zarr":
        mapper = fs.get_mapper(file_path_str, check=True)
        storage = zarr.open(mapper, mode="r")
        adata = _subset_adata_storage(storage, query_obs, query_var)

    return adata


def get_obs_var_storage(
    storage: Union[zarr.Group, h5py.File], which=Literal["obs", "var"]
) -> pd.DataFrame:
    with storage as access:
        result = read_elem(access[which])
    if isinstance(result, pd.DataFrame):
        return result
    # is array
    elif "index" in result.dtype.fields:
        return pd.DataFrame.from_records(result, index="index")
    else:
        return pd.DataFrame.from_records(result)


class CloudAnnData:
    def __init__(self, file: File):
        self._file = file

    def _get_obs_var(self, which=Literal["obs", "var"]) -> pd.DataFrame:
        file_path = filepath_from_file(self._file)
        fs, file_path_str = _infer_filesystem(file_path)
        if self._file.suffix == ".h5ad":
            with fs.open(file_path_str, mode="rb") as f:
                storage = h5py.File(f, mode="r")
                df = get_obs_var_storage(storage, which)
        elif self._file.suffix == ".zarr" or self._file.suffix == ".zrad":
            mapper = fs.get_mapper(file_path_str, check=True)
            storage = zarr.open(mapper, mode="r")
            df = get_obs_var_storage(storage, which)
        return df

    @cached_property
    def obs(self) -> pd.DataFrame:
        return self._get_obs_var(which="obs")

    @cached_property
    def var(self) -> pd.DataFrame:
        return self._get_obs_var(which="var")
