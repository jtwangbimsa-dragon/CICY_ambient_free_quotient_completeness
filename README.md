# Configuration-level verification of the CICY completeness criteria

This folder contains the configuration matrices and the programs needed to
recompute the four criteria used in the accompanying paper:

1. the coefficient-independent section-space condition E3;
2. the direct simultaneous E1--E2 certificate from vertical curves;
3. the integral Picard-lattice condition E1 for the extended set;
4. the E2-chi obstruction for possible frame-changing free automorphisms.

The programs recompute these quantities from the supplied configuration
records.  They do not use defining-polynomial coefficients, and they do not
accept the previously generated result tables as proof input.  Output is
written to `reproduction_output/`, which is intentionally absent from the
distributed archive.

This archive does **not** contain or attempt to reproduce the subsequent
classification of free actions into 2,625 negative cases and ten `C_3`
families.  Its scope ends with verification of the completeness criteria.

## 1. Mathematical input

The main input is:

```text
data/cicy_criteria_input_7820.jsonl
```

Each line contains one CICY number, the configuration selected for the
calculation (the original-list `Conf` in 4,874 cases and a `FavourConf` in
2,946 cases), the displayed working configuration, the known value of `h11`,
and, when a graph re-embedding was used, the exact combinatorial description
of that re-embedding.  The standard original-list `Conf` for **every** one of
the 7,820 numbers is given separately as `OriginalConf` in the two bundled
Mathematica tables.  See [data/PROVENANCE.md](data/PROVENANCE.md) for the
source publications, download pages, and the meaning of these three matrices.
The programs recompute all intersection-theoretic and
cohomological quantities needed below.  This input file contains no saved
E1, E2, E3 or E2-chi verdict.

One point cannot be inferred from a bare integer matrix: E1 requires the
statement that the chosen ambient divisor classes span `Pic(X)` over the
rationals, equivalently here that the presentation is a certified favourable
one with

```text
h11 = number of ambient projective factors.
```

This favourable-rank statement is therefore explicit input.  The integral
part of E1 is then proved by the programs.  The absence of Picard torsion is a
theorem-level input for smooth standard non-product CICY threefolds, not a
numerical inference made by the code.

The file

```text
data/extended_2635_nums.json
```

contains only the list of CICY numbers on which the extended E1 and E2-chi
tests are to be run.  It contains no precomputed certificates.

## 2. Software environment

The frozen reference environment was Python 3.12.2 with NumPy 1.26.4 and
SymPy 1.14.0.  From the root of this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No Mathematica, Macaulay2, Singular, database server, network connection, or
defining polynomial is required for these four calculations.

## 3. One-command verification of one configuration

For any configuration contained in the supplied 7,820-case catalogue, a
reader can start with its standard original matrix, selected favourable
matrix, or displayed working matrix:

```bash
python code/verify_configuration.py \
  --conf '[[3],[3]]' \
  --output-dir verification_output/bicubic
```

The program identifies the matching CICY number in the two Mathematica tables
and the JSON input, checks that their working matrices agree, and then
recomputes E3 and the direct E1--E2 certificate.  If the direct route is
inconclusive, it automatically continues with integral E1 and E2-chi.  Its final answer is
`verification_output/bicubic/verification_summary.json`, and the complete
fresh certificates are stored beside it.  The summary states the matched CICY
number, which kind of matrix was supplied (`input_matrix_roles`), the route
used, and the status of every criterion.  A long matrix may instead be placed
in a JSON file and supplied with `--conf-file`.  If the same matrix occurs
under several CICY numbers, use `--num` to identify the intended entry.  For
example, after copying `OriginalConf` for number 414 from the bundled table
into `original_414.json`, run:

```bash
python code/verify_configuration.py \
  --conf-file original_414.json --num 414 \
  --output-dir verification_output/414
```

This looks up number 414 and runs the tests on its **working** presentation;
the lookup is not itself a proof that two presentations describe the same
smooth threefold.  The published favourable descriptions and the explicit
graph data supply the relevant presentation information.  One may also
select a record directly:

```bash
python code/verify_configuration.py \
  --record data/cicy_criteria_input_7820.jsonl \
  --num 414 \
  --output-dir verification_output/my_graph_case
```

For a matrix outside the bundled catalogue, the reader must additionally give
`--h11`, `--favourable-certified`, and `--standard-nonproduct-cicy`, as in
Section 5.  These are mathematical hypotheses that cannot be proved from the
integer entries of a bare matrix.  Omitting either hypothesis causes the
E1-dependent conclusion to remain unresolved.

## 4. The four component programs

### 4.1 E3

```bash
python code/check_E3.py
```

For every ambient factor this computes the relevant Koszul--Bott terms and
checks that restriction of ambient linear forms gives the complete intrinsic
section space.  For the supplied input the expected summary is

```text
7,820 configurations certified
57,885 ambient-factor tests
0 unresolved
```

An independent implementation is retained as
`code/E3/validate_e3_all_7820.py`.

### 4.2 Direct E1 and E2

```bash
python code/check_E1_E2.py
```

For each row `i`, the program computes the top Chern number of the compact
vertical-line problem on

```text
Gr(2,n_i+1) x product_{j != i} P^{n_j}.
```

A nonzero result forces an effective curve `C_i` with
`J_j.C_i = delta_ij`.  Such curves in every coordinate direction prove both
the integral saturation needed for E1 and equality of the nef cone with the
ambient nonnegative orthant needed for E2.  A zero is reported only as
unresolved.  The expected result is

```text
4,874 PASS_E1_AND_E2
2,946 NOT_CERTIFIED_BY_VERTICAL_LINE_TEST
```

### 4.3 Integral E1 for the extended set

```bash
python code/FINITE_PICARD_ACTION/check_E1.py --workers 4
```

This is one sequential calculation, with three possible stopping points:

```text
Smith normal form
    -> augmented curve lattice plus exact intersection/c2/HRR tests
    -> stable-map top-Chern curve witnesses.
```

The projection, line and conic witnesses are freshly recomputed from each
configuration.  The program does not read `e1_complete_2635.jsonl` or any
other frozen E1 result.  The expected route counts are

```text
SNF                                      2,200
augmented lattice + intersection/c2/HRR    403
stable-map top-Chern                        32
total                                     2,635
```

For a single case, for example:

```bash
python code/FINITE_PICARD_ACTION/check_E1.py \
  --only-nums 3398 \
  --output reproduction_output/e1_3398.jsonl \
  --summary reproduction_output/e1_3398_summary.json
```

### 4.4 E2-chi

Run this only with the output of the preceding E1 calculation:

```bash
python code/FINITE_PICARD_ACTION/check_E2chi.py --workers 4
```

For each possible image of an ambient divisor, the program computes the exact
compact ellipsoid determined by the intersection tensor and `c2`, enumerates
every integral point in it, constructs every compatible unimodular Picard
matrix, and applies the necessary divisibility conditions for a free cyclic
action.  Any surviving frame-changing matrix remains unresolved.

The public entry point includes two non-generic routes:

- Num 3936 is exhausted in the disjoint layers `0<=k<=5` and `k=6,...,11`,
  where `k` is the number of changed ambient-frame columns;
- Num 4081 is transported only after the program reconstructs an explicit
  row/column isomorphism with Num 3936.

The default output is compact: the exhaustive calculation is performed, but
the full matrix arrays are not copied to disk.  Use `--full-certificates` only
if the complete per-matrix output is desired.  The options `--point-cap`,
`--matrix-cap`, and `--node-cap-special` are diagnostic only; reaching a cap gives
`UNKNOWN` and can never certify a case.

The calculation can be divided by Picard rank on different machines, for
example:

```bash
python code/FINITE_PICARD_ACTION/check_E2chi.py \
  --min-rank 5 --max-rank 7 --workers 4 \
  --records-output reproduction_output/e2chi_rank5_7.jsonl \
  --summary-output reproduction_output/e2chi_rank5_7_summary.json
```

With no diagnostic caps, the supplied 2,635-case calculation is expected to
certify frame preservation in all 2,635 cases.  This is a completeness
criterion for free actions; it is not the later classification of which free
groups actually occur.

## 5. Running the four components separately on one matrix

For an untransformed favourable presentation, first create the input record:

```bash
python code/prepare_case.py \
  --conf '[[3],[3]]' \
  --h11 2 \
  --favourable-certified \
  --standard-nonproduct-cicy \
  --num 900001 \
  --output my_case.jsonl
```

Then run the same programs on that record:

```bash
python code/check_E3.py \
  --input my_case.jsonl --expected-count 1 \
  --certificates reproduction_output/my_case_E3.jsonl \
  --summary reproduction_output/my_case_E3_summary.json

python code/check_E1_E2.py \
  --input my_case.jsonl --expected-count 1 \
  --certificates reproduction_output/my_case_direct_E1_E2.jsonl \
  --summary reproduction_output/my_case_direct_E1_E2_summary.json

python code/FINITE_PICARD_ACTION/check_E1.py \
  --dataset my_case.jsonl --all-input \
  --output reproduction_output/my_case_E1.jsonl \
  --summary reproduction_output/my_case_E1_summary.json

python code/FINITE_PICARD_ACTION/check_E2chi.py \
  --dataset my_case.jsonl \
  --e1-records reproduction_output/my_case_E1.jsonl \
  --records-output reproduction_output/my_case_E2chi.jsonl \
  --summary-output reproduction_output/my_case_E2chi_summary.json
```

If the direct E1--E2 program gives `PASS_E1_AND_E2`, that direct route is
already sufficient and the weaker E2-chi replacement is not needed.  It is
included above to show that a reader-supplied matrix is accepted by every
public entry point.

The two explicit flags above are hypotheses, not numerical conclusions:

- `--favourable-certified` asserts the rational Picard-rank statement;
- `--standard-nonproduct-cicy` places the case within the torsion-free theorem.

If either hypothesis is unavailable, the appropriate E1 conclusion is
`UNKNOWN`.  For a graph re-embedding, use the richer record format in the
distributed 7,820-case input so that the program can verify the graph
isomorphism and transport the Picard frame.

## 6. Fail-closed interpretation

Only an explicit `pass` field is a positive certificate.  The following all
mean unresolved:

- a zero top Chern witness;
- failure of the negative-definiteness test;
- a surviving Picard matrix;
- a missing favourable or torsion hypothesis;
- a resource cap, exception, or incomplete enumeration.

No unresolved outcome is counted as a negative theorem.

## 7. Tests

```bash
cd code/FINITE_PICARD_ACTION
python -m pytest -q tests
```

The small end-to-end example in Section 4 additionally checks that all four
public entry points accept a fresh configuration rather than a frozen result
record.
