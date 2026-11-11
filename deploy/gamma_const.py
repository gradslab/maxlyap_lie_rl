"""THE WORKING-BALL RADIUS, in ONE place. Read this before changing anything in it."""
from __future__ import annotations

import math

GAMMA_CERTIFIED: float = 2.7488935

GAMMA_EXACT: float = math.pi - math.pi / 8

GAMMA_SHORTFALL: float = GAMMA_EXACT - GAMMA_CERTIFIED

assert 0.0 < GAMMA_SHORTFALL < 1e-7, "gamma constants drifted; re-read this file before proceeding"
assert GAMMA_CERTIFIED < GAMMA_EXACT, "certified radius must stay BELOW the exact one (conservative)"
