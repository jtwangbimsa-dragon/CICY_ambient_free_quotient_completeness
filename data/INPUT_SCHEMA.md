# Input record

The public programs read JSON Lines: one JSON object per configuration.

For an untransformed case the required mathematical information is:

- `Num`: a stable integer label;
- `original_selected_conf` and `working_conf`: factor-by-equation matrices.
  The former is the matrix selected from `Conf` or `FavourConf` before any
  graph re-embedding; it is not always the original standard-list `Conf`;
- `original_ambient_dimensions` and `working_ambient_dimensions`, equal to the
  row sums minus one;
- `official_fields.H11`: the independently known Picard number;
- `hypotheses.rational_favourability_certified`: whether the chosen
  presentation is known to be favourable over the rationals;
- `hypotheses.standard_nonproduct_cicy`: the theorem-level gate used to apply
  the standard torsion-free Picard conclusion.

The first hypothesis records only the independently known *rational*
favourability statement.  It is not an integral-E1 answer.  The
Smith-normal-form, finite-index and stable-map parts of integral E1 are always
recomputed.  The input contains no saved E2, E2-chi or E3 decision.

The helper `code/prepare_case.py` validates the matrix and computes its Euler
characteristic and `c2` pairings for its diagnostic printout.  The criterion
programs recompute the intersection tensor, `c2`, Euler characteristic and
all curve and section-space tests from the matrices.

For the graph re-embeddings used in the supplied dataset, the record also
contains `obstruction_mappings`.  These identify every replaced
`2 e_r + e_i` column and the two replacement `e_r + e_i` columns.  E1 is
transported only after the program checks the elimination identity, the two
projective charts, and the projection back to the original presentation.

Configuration orientation is always:

```text
ambient projective factors x defining equations.
```

Transposed or ambiguous matrices are not accepted silently.

The one-command program `code/verify_configuration.py` performs an exact
lookup across this JSON input and the `OriginalConf` fields in the two bundled
Mathematica tables.  The latter provide the original standard-list matrices
for all 7,820 numbers.  It verifies that the tables' `WorkingConf` fields
match this JSON input.  If any supplied matrix occurs under several CICY
numbers, the program lists them and requires `--num` to disambiguate them.
See [PROVENANCE.md](PROVENANCE.md) for sources and lookup limitations.
