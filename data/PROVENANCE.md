# Sources and matrix lookup

The CICY number `Num` is the number in the standard 7,890-entry CICY list.
The original configuration matrix is the `Conf` field of that list.  The
original list and its primary publication are available from the
[Oxford CICY data page](https://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/)
(P. Candelas, A. M. Dale, C. A. Lütken and R. Schimmrigk, *Complete
intersection Calabi–Yau manifolds*, Nuclear Physics B **298** (1988) 493).

For cases requiring a favourable product-projective presentation, we use the
favourable list accompanying L. B. Anderson, X. Gao, J. Gray and S.-J. Lee,
[*Fibrations in CICY Threefolds*](https://arxiv.org/abs/1708.07907), JHEP
**10** (2017) 077.  The [associated data page](https://www1.phys.vt.edu/cicydata/)
provides the favourable matrices and explains their numbering and fields.

The two tables in this folder have one entry for each of the 7,820 CICY
numbers considered here: 7,509 certified by the stated criteria and 311
still open.  In each entry:

- `OriginalConf` is the standard-list `Conf` for that `Num`;
- `WorkingConf` is the matrix tested by the programs;
- `WorkingSource` says whether the selected matrix came from `Conf` or
  `FavourConf`;
- `GraphEmbedded` notes whether a further graph re-embedding was used.

The original matrices were exported from a local parsed copy of the 7,890
entries whose SHA-256 is
`4a80db13ff57829f9492fdf135dd2d7a1aa174eb255dd3b1144097f7aeb09fc8`.
This hash is for the **parsed inventory**, not for the file served by Oxford.
An exact comparison of all 7,820 bundled `OriginalConf` matrices against
that frozen 7,890-entry inventory found zero mismatches; 70 original-list
entries lie outside the scope of this archive.
The 7,820-case JSON input has SHA-256
`597af98fd6f797802003809ebb6401245b24b0891c0a5ad5270bc6b59f697d77`.
The complete distributed-file checksums are in `SHA256SUMS` at the archive root.

In the JSON input, `original_selected_conf` means the matrix selected *before*
any graph re-embedding.  It is not necessarily the original-list `Conf`:
for the 2,946 entries with `source = "FavourConf"`, it is the favourable
matrix.  This is why the standard original matrix is supplied separately in
the two Mathematica tables.

## Querying a matrix

`code/verify_configuration.py` accepts a matrix from any of these three
stages: standard original, selected, or working.  It reads the two table
files as data (without executing Mathematica), joins them to the JSON input
by `Num`, and checks that their `WorkingConf` fields agree.  The matrix must
have the same row and column order as the bundled entry.  The lookup does
not try to recognize arbitrary row/column permutations or other equivalent
presentations.  If several numbers have the identical supplied matrix, use
`--num`; a guess is never made.  For a long matrix, save it as a JSON array
and use `--conf-file path/to/matrix.json`.

The output field `input_matrix_roles` reports how the supplied matrix matched.
The criterion calculation always uses the matched **working** presentation.
Identifying an original matrix with its favourable presentation is a geometric
input from the cited favourable-list construction, not something proved merely
by matrix equality or by this lookup program.  The program checks graph
re-embeddings using the bundled graph metadata where relevant.  A result
`UNKNOWN_NOT_CERTIFIED_BY_THESE_CRITERIA`, including one of the 311 entries,
means that this method has not settled the case; it does **not** say that no
free quotient exists.

Within Mathematica, the original and working matrices can also be inspected
directly, for example:

```Mathematica
Get["data/cicy_certified_7509_original_and_working_confs.wl"];
CertifiedCICYLookup7509[414]["OriginalConf"]
CertifiedCICYLookup7509[414]["WorkingConf"]
```
