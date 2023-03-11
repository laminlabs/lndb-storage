"""Storage → object.

File → memory:

.. autosummary::
   :toctree: .

   h5ad_to_anndata

Store files:

.. autosummary::
   :toctree: .

   store_object
   store_png
   delete_storage
"""

from ._file import delete_storage, load_to_memory, store_object
from ._filesystem import _infer_filesystem
from ._h5ad import h5ad_to_anndata
from ._images import store_png
from ._upath_ext import UPath
from ._zarr import read_adata_zarr, write_adata_zarr
