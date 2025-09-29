from __future__ import annotations

import json
from services.compute_api.app import app

print(json.dumps(app.openapi(), indent=2))