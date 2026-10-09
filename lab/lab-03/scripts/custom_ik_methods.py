"""Compatibility re-exports. Run batches from the dedicated scripts:

    python3 scripts/ik_dls_pick_place.py --episodes 25 --no-gui
    python3 scripts/ik_qp_pick_place.py  --episodes 25 --no-gui
"""

from ik_dls_pick_place import solve_ik_dls
from ik_qp_pick_place import solve_ik_qp

__all__ = ["solve_ik_dls", "solve_ik_qp"]
