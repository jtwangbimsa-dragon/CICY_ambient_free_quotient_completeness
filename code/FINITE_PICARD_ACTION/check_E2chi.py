#!/usr/bin/env python3
"""Public E2-chi program: recompute the criterion from configurations.

Run this only after ``check_E1.py``.  The same entry point handles ordinary
ellipsoid enumeration, the exhaustive layered calculation for Num 3936, and
the explicit configuration-isomorphism transport to Num 4081.
"""

from intrinsic_e2chi_complete.e2chi_recompute import main


if __name__ == "__main__":
    raise SystemExit(main())
