import nox
import requests  # type: ignore
from laminci import move_built_docs_to_docs_slash_project_slug, upload_docs_artifact
from laminci.nox import build_docs, login_testuser1, run_pre_commit, run_pytest

nox.options.reuse_existing_virtualenvs = True


@nox.session(python=["3.7", "3.8", "3.9", "3.10", "3.11"])
def lint(session: nox.Session) -> None:
    run_pre_commit(session)


@nox.session(python=["3.7", "3.8", "3.9", "3.10", "3.11"])
def build(session):
    login_testuser1(session)
    session.install(".[dev,test]")
    response = requests.get("https://github.com/laminlabs/lamindb/tree/rename")
    if response.status_code < 400:
        session.install("git+https://github.com/laminlabs/lamindb@rename")
    else:
        session.install("git+https://github.com/laminlabs/lamindb")
    run_pytest(session)
    build_docs(session)
    upload_docs_artifact()
    move_built_docs_to_docs_slash_project_slug()
