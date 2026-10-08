"""Test harness for the manifold pipelines.

Two things make these pipelines awkward to import in a test process, and this
module deals with both so the test files can stay about behaviour:

1. **Heavy third-party imports.** A pipeline imports fastapi, pydantic, httpx,
   requests and psycopg2 at module scope, none of which the code under test
   needs: the error formatters and the cache-control placement use only ``re``,
   ``json`` and ``self.valves``. Installing the real packages just to import the
   module would make the suite slow and network-dependent, so the imports are
   stubbed before any pipeline module loads. If a future test exercises code
   that genuinely calls one of these libraries, give that test the real package
   rather than widening the stubs.

2. **Pipelines are files, not a package.** ``pipelines/`` has no ``__init__``
   and the filenames are the deployment contract (PIPELINES_URLS points at
   them), so they are loaded by path.

``Pipeline.__init__`` reads environment variables and talks to LiteLLM, so the
fixtures below build instances with ``object.__new__`` and attach only the
valves a test needs. That keeps a unit test from depending on a live LiteLLM.
"""

import importlib.util
import pathlib
import sys
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PIPELINES = REPO_ROOT / "pipelines"

# ``schemas`` lives at the repo root and is imported by every pipeline.
sys.path.insert(0, str(REPO_ROOT))


def _install_stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _HTTPException(Exception):
    """Stands in for fastapi.HTTPException (status_code + detail)."""

    def __init__(self, status_code=None, detail=None):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class _BaseModel:
    """Enough of pydantic.BaseModel to declare and instantiate Valves."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _install_third_party_stubs():
    # Only installed if absent, so a developer with the real packages in their
    # environment tests against those instead.
    if "fastapi" not in sys.modules:
        _install_stub("fastapi", HTTPException=_HTTPException)
    if "pydantic" not in sys.modules:
        _install_stub(
            "pydantic",
            BaseModel=_BaseModel,
            ConfigDict=dict,
            Field=lambda *args, **kwargs: None,
        )
    if "httpx" not in sys.modules:
        _install_stub("httpx", Client=object, AsyncClient=object)
    if "requests" not in sys.modules:
        exceptions = _install_stub(
            "requests.exceptions",
            ConnectionError=type("ConnectionError", (Exception,), {}),
            Timeout=type("Timeout", (Exception,), {}),
            RequestException=type("RequestException", (Exception,), {}),
        )
        _install_stub("requests", exceptions=exceptions, get=None, post=None)
    if "psycopg2" not in sys.modules:
        _install_stub(
            "psycopg2",
            Error=type("Error", (Exception,), {}),
            extras=types.SimpleNamespace(),
        )


_install_third_party_stubs()


def load_pipeline_module(filename):
    """Import a pipeline by filename and return its module."""
    path = PIPELINES / filename
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The variant each environment loads is decided by PIPELINES_URLS; see
# tests/README.md. Both are tested because the error formatting is shared.
IMAGE_MANIFOLD = "litellm_image_manifold_pipeline_with_headers_and_keys.py"
TEXT_MANIFOLD = "litellm_manifold_pipeline_with_headers_and_keys.py"


def build_pipeline(filename, **valves):
    """A Pipeline with only the valves a test cares about.

    ``__init__`` is bypassed deliberately: it resolves env vars and calls
    LiteLLM, neither of which a unit test should need.
    """
    module = load_pipeline_module(filename)
    pipeline = object.__new__(module.Pipeline)
    defaults = {
        "LITELLM_PIPELINE_DEBUG": False,
        "LITELLM_USER_BUDGET_PERIOD": "1d",
    }
    defaults.update(valves)
    pipeline.valves = types.SimpleNamespace(**defaults)
    return pipeline


class FakeResponse:
    """The parts of a requests.Response that _handle_litellm_error reads."""

    def __init__(self, status_code, payload=None, invalid_json=False):
        self.status_code = status_code
        self._payload = payload
        self._invalid_json = invalid_json

    def json(self):
        if self._invalid_json:
            import json

            raise json.JSONDecodeError("no json", "", 0)
        return self._payload


def litellm_error(status_code, message, error_type):
    """A LiteLLM error response body, as the proxy actually shapes it."""
    return FakeResponse(
        status_code, {"error": {"message": message, "type": error_type}}
    )
