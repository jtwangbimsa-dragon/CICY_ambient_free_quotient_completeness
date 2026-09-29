# Extended E1 and E2-chi verification

This directory has two public programs, to be run in this order:

```bash
python check_E1.py --workers 4
python check_E2chi.py --workers 4
```

`check_E1.py` starts from the configuration records. It recomputes the curve
witnesses, Smith normal form, finite-index lattice tests and, only where
needed, the stable-map top-Chern witnesses. It writes
`../../reproduction_output/e1_complete_2635.jsonl`.

`check_E2chi.py` reads that newly produced E1 file. It recomputes the
intersection tensor and `c2`, proves boundedness of the column regions,
enumerates all integral columns and compatible unimodular Picard actions, and
applies the free-action divisibility obstructions. It refuses to certify a
case if E1 is missing or unresolved.

The modules under `e1_generators/` and `intrinsic_e2chi_complete/` are the
mathematical implementation used by these two entry points. They are not
precomputed answers. In particular, neither public program reads a frozen E1
or E2-chi result table.

Resource limits are diagnostic only. If `--point-cap`, `--matrix-cap` or
`--node-cap-special` is reached, the result is `UNKNOWN`, never a certificate.

See the root `README.md` for the mathematical hypotheses, the single-matrix
workflow, expected full-dataset counts and tests.
