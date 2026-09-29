#!/usr/bin/env python3
"""Public E1 program: recompute E1 from the supplied configurations.

The program does not read the distributed frozen E1 result files.  It
recomputes the SNF, augmented lattice/HRR and stable-map stages in sequence
and writes the full integral-E1 ledger required by ``check_E2chi.py``.
"""

from intrinsic_e2chi_complete.e1_recompute import main


if __name__ == "__main__":
    raise SystemExit(main())
