from pathlib import Path

import nbproject_test as test
from lamin_logger import logger


def test_notebooks():
    # assuming this is in the tests folder
    docs_folder = Path(__file__).parents[1] / "docs/"

    for check_folder in docs_folder.glob("./**"):
        # these are the notebook testpaths
        if not str(check_folder).endswith(("guide", "faq")):
            continue
        logger.debug(f"\n{check_folder}")
        test.execute_notebooks(check_folder, write=True)
