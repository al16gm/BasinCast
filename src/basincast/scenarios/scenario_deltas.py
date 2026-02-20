from __future__ import annotations

from typing import Dict, List

# Monthly deltas by scenario.
# Convention:
# - precip_mult: multiplicative factor on precipitation (e.g., -0.10 means -10%)
# - t2m_add: additive delta on temperature in °C (e.g., +1.0 means +1°C)
SCENARIO_DELTAS: Dict[str, Dict[str, List[float]]] = {
    "Base": {
        "precip_mult": [0.0] * 12,
        "t2m_add": [0.0] * 12,
    },
    "Favorable": {
        "precip_mult": [0.05] * 12,
        "t2m_add": [-0.2] * 12,
    },
    "Unfavorable": {
        "precip_mult": [-0.10] * 12,
        "t2m_add": [0.8] * 12,
    },
}