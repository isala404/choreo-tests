Internal API
============

Project-scoped FastAPI service exposing simple endpoints for testing.

Endpoints
---------

- /success: Returns 200 with a message indicating it is from internal-api
- /failure: Returns 500 with a message indicating it is from internal-api
- /random: Randomly returns success or 500 error, indicating it is from internal-api

Run locally
-----------

Using uv (recommended):

```bash
cd internal-api
uv pip install -r <(uv pip compile --generate-hashes --quiet pyproject.toml) # or use your preferred installer
uv run python main.py
```

Or with pip:

```bash
cd internal-api
python -m venv .venv && source .venv/bin/activate
pip install -e .
python main.py
```

Choreo configuration
--------------------

The component is configured with Project network visibility in `.choreo/component.yaml`.

