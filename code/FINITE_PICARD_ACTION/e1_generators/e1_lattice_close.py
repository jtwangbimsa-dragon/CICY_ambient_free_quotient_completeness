#!/usr/bin/env python3
"""Augmented integral-E1 audit for favourable CICYs.

Goal
----
Let  L = sum_i Z H_i  be the lattice spanned by the restricted ambient
hyperplane classes and  M = Pic(X)/tors.  Favourability gives rank L = rank M
= h^{1,1}, hence  L  is a finite-index sublattice of  M  inside  L (x) Q.
Integral E1 is the statement  L = M.

Two independent certificates are combined.

(A) LOWER CERTIFICATE  (bounds M from above by a lattice N).
    Every integral homology class  C in H_2(X,Z)  pairs integrally with every
    divisor class, so
        M  subset  N := { x in L(x)Q : <x, C_alpha> in Z for every known
                                        integral class C_alpha }.
    The columns C_alpha used are, in increasing order of novelty:
      (A1) PD(H_j cup H_k)            -- pairing column (d_{ijk})_i
      (A2) PD(c_2(TX))                -- pairing column (c_2 . H_i)_i
      (A3) degree-e_i stable-map class from a non-zero deformation-invariant
           degree-one witness of the freshly recomputed projection audit  -- column e_i
      (A4) NEW: type-2 conic class 2C_i, from a non-zero deformation-invariant
           Porteous conic witness                     -- column 2 e_i
      (A5) NEW: vertical fibre curves of the partial coordinate projections
           (Proposition B below)                      -- column supported on S
      (A6) NEW: vertical LINES in the i-th projective factor, for EVERY row i,
           certified by a non-zero Grassmannian Euler number L_i
           (Proposition C below)                      -- column e_i
    (A1)-(A3) are exactly the columns of the baseline Smith-normal-form stage.
    (A4)-(A6) are new and are reported separately so that the effect of each
    is visible.  (A6) subsumes (A3): on the rows the baseline stage already
    handled it reproduces the same unit column, and it additionally covers the
    "forced_positive_dimensional_fiber" rows, which the baseline stage skipped
    entirely -- that omission is the sole reason the 74 official-Conf records
    failed.

(B) UPPER CERTIFICATE  (excludes every proper overlattice of L inside N).
    On a smooth projective Calabi-Yau threefold the following are integers for
    every  D, D', D'' in Pic(X):
        d(D,D',D'')  (an intersection number),
        c_2(X) . D,
        chi(X, O_X(D)) = D^3/6 + c_2 . D /12    (Hirzebruch-Riemann-Roch,
                                                 c_1 = 0, td = 1 + c_2/12).
    So M is an intermediate lattice  L subset M subset N  on which the cubic
    form d, the linear form c_2, and the integer-valued cubic chi are all
    integral.  If no intermediate lattice strictly bigger than L has that
    property, then M = L and integral E1 holds.
    It suffices to test the lattices  L + Z v  for  v in N  whose class in N/L
    has PRIME order: any M != L contains such a lattice, and all three
    integrality properties are inherited by sublattices.

Proposition C (vertical lines; uniform, valid for EVERY member of the family).
    Fix a row i, put B = prod_{j != i} P^{n_j} and consider the smooth compact
    parameter space  P = G(2, n_i+1) x B, whose points are pairs (ell, p) with
    ell a line in P^{n_i}.  Let S be the tautological rank-2 subbundle on the
    Grassmannian, so that H^0(ell, O(d)) = Sym^d S^*.  Restriction of the
    defining polynomial f_a to {p} x ell is a section of the vector bundle
        E_a = Sym^{q_{a,i}} S^*  (x)  O_B(q_a|_B),   rank q_{a,i} + 1,
    and (ell,p) lies in the common zero scheme Z of the induced section of
    E = (+)_a E_a  if and only if  {p} x ell  is contained in X.  Ranks and
    dimensions always match:
        dim P = 2(n_i - 1) + (n_tot - n_i) = n_i + n_tot - 2,
        rk E  = sum_a (q_{a,i}+1) = (n_i+1) + K = n_i + n_tot - 2,
    using the Calabi-Yau row condition sum_a q_{a,i} = n_i+1 and K = n_tot - 3.
    Therefore
        L_i := int_P c_top(E) = int_P prod_a prod_{r=0}^{q_{a,i}}
                                     ( r x + (q_{a,i}-r) y + q_a|_B . J )
    is a well-defined integer (x,y the Chern roots of S^*, and
    int_{G(2,n+1)} P(x,y) = [x^{n-1}y^{n-1}]P - [x^n y^{n-2}]P).

    If  L_i != 0  then  Z is non-empty.
    Indeed the localized top Chern class of E with respect to the section lives
    in A_0(Z), and its image in A_0(P) is c_top(E) cap [P]; if Z were empty that
    class would be zero, contradicting L_i != 0.  No transversality, genericity,
    excess-dimension or deformation hypothesis is used, so the conclusion holds
    for every X cut out by polynomials of these multidegrees.

    A point of Z gives an honest line ell = P^1 inside X with
        J_i . ell = 1,      J_j . ell = 0   for j != i,
    i.e. the unit pairing column e_i.

    Validation.  L_i reproduces the official CICY-database degree-one
    Gromov-Witten entry GWI for all 30924 rows of the 4874 official-Conf
    records, with zero mismatches -- including the 121 rows of kind
    "forced_positive_dimensional_fiber" that the earlier projection scan excluded from
    its own regression -- and reproduces the classical line counts 2875 on the
    quintic, 1280 on (2,4) in P^5, 1053 on (3,3) in P^5, 720 on (2,2,3) in P^6,
    512 on (2,2,2,2) in P^7 and 189 on (3,3) in P^2 x P^2.

Proposition B (vertical fibre curves).
    X = V(f_1,...,f_K) subset A = prod_i P^{n_i}, the f_a general of multidegree
    q_a.  Let S be a non-empty set of rows, A_S = prod_{j in S} P^{n_j},
    n_S = sum_{j in S} n_j, Supp(S) = { a : q_{a,j} > 0 for some j in S },
    m_S = |Supp(S)|.  Choose e_j >= 0 (j in S) with m_S + sum_j e_j = n_S - 1
    and set
        gamma = prod_{j in S} J_j^{e_j} * prod_{a in Supp(S)} (sum_{j in S} q_{a,j} J_j)
              in H^{2(n_S-1)}(A_S, Z).
    If gamma != 0 then X contains an effective 1-cycle Z with
        J_j . Z = [ prod_{l in S} J_l^{n_l} ] ( J_j * gamma )   for j in S,
        J_j . Z = 0                                             for j not in S.
    Proof.  Let B = prod_{j not in S} P^{n_j} and let B_0 subset B be the common
    zero locus of the f_a with a not in Supp(S); those polynomials involve only
    B-coordinates, so pr_B(X) subset B_0, in particular B_0 is non-empty.  For
    p in B_0 the fibre pr_B^{-1}(p) is the zero locus in {p} x A_S of the m_S
    polynomials f_a(-,p), a in Supp(S), of multidegrees (q_{a,j})_{j in S}.  For
    p fixed the evaluation map from multidegree-q_a forms on A to
    multidegree-q_a|_S forms on A_S is surjective, so for general f the
    f_a(-,p) are general members of base-point-free linear systems on A_S; the
    same is true of the e_j extra general hyperplanes.  A_S is homogeneous under
    prod_{j in S} PGL_{n_j+1}, so by Kleiman transversality each successive
    general member either meets a given component properly or misses it
    entirely (the latter only for components contracted by the corresponding
    morphism).  Hence the total intersection Z is either empty or of pure
    dimension n_S - m_S - sum e_j = 1 with cycle class gamma.  As gamma != 0 it
    is non-empty, so Z is a curve of class gamma inside {p} x A_S subset X.  Its
    intersection numbers with J_j, j not in S, vanish because Z lies in a single
    fibre.  QED

    The |S| = 1 case is the "forced positive dimensional fibre" row of the
    freshly recomputed projection audit and is uniform (valid on EVERY member of the family): in
    P^{n_i} an intersection of m_i <= n_i - 1 hypersurfaces has dimension
    >= n_i - m_i >= 1 at every point of the non-empty image of pr_B, by Krull.
    The class computation, and the |S| >= 2 case, use generality of f, so the
    (A5) columns are flagged generic_member_scope.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------- exact ring


def ambient_dims(conf: list[list[int]]) -> list[int]:
    return [sum(r) - 1 for r in conf]


def poly_mul_linear(poly: dict, weights: tuple[int, ...], cap: tuple[int, ...]) -> dict:
    out: dict = {}
    for mono, coeff in poly.items():
        for idx, w in enumerate(weights):
            if not w or mono[idx] >= cap[idx]:
                continue
            key = list(mono)
            key[idx] += 1
            key = tuple(key)
            out[key] = out.get(key, 0) + coeff * w
    return {k: v for k, v in out.items() if v}


def normal_product(conf: list[list[int]]) -> dict:
    """prod_a (q_a . J) truncated in the Chow ring of the ambient product."""
    dims = tuple(ambient_dims(conf))
    poly = {(0,) * len(conf): 1}
    for col in zip(*conf):
        poly = poly_mul_linear(poly, col, dims)
    return poly


def triple_numbers(conf: list[list[int]]) -> dict[tuple[int, int, int], int]:
    """d_{rst} = int_X H_r H_s H_t, exact."""
    dims = tuple(ambient_dims(conf))
    rho = len(conf)
    base = normal_product(conf)
    out: dict[tuple[int, int, int], int] = {}
    for r, s, t in itertools.combinations_with_replacement(range(rho), 3):
        exp = [0] * rho
        exp[r] += 1
        exp[s] += 1
        exp[t] += 1
        comp = tuple(dims[i] - exp[i] for i in range(rho))
        out[(r, s, t)] = 0 if min(comp) < 0 else base.get(comp, 0)
    return out


def d_lookup(triples, r, s, t):
    return triples[tuple(sorted((r, s, t)))]


def c2_from_conf(conf: list[list[int]]) -> list[int]:
    """c_2(X).H_i, exact:  c_2 = 1/2 [ sum_a (q_a.J)^2 - sum_i (n_i+1) J_i^2 ]."""
    dims = ambient_dims(conf)
    rho = len(conf)
    cap = tuple(dims)
    base = normal_product(conf)          # [X]
    cols = list(zip(*conf))
    # quadratic class Q = sum_a (q_a.J)^2 - sum_i (n_i+1) J_i^2, as a dict
    quad: dict = {}

    def add(mono, val):
        quad[mono] = quad.get(mono, 0) + val

    for col in cols:
        for i in range(rho):
            for j in range(rho):
                if not col[i] or not col[j]:
                    continue
                exp = [0] * rho
                exp[i] += 1
                exp[j] += 1
                add(tuple(exp), col[i] * col[j])
    for i in range(rho):
        exp = [0] * rho
        exp[i] = 2
        add(tuple(exp), -(dims[i] + 1))
    quad = {k: v for k, v in quad.items() if v and all(k[i] <= cap[i] for i in range(rho))}

    out = []
    for i in range(rho):
        total = 0
        for qmono, qval in quad.items():
            exp = list(qmono)
            exp[i] += 1
            if exp[i] > cap[i]:
                continue
            comp = tuple(cap[k] - exp[k] for k in range(rho))
            if min(comp) < 0:
                continue
            total += qval * base.get(comp, 0)
        assert total % 2 == 0, "c_2 halving failed"
        out.append(total // 2)
    return out


# ------------------------------------------------------- integer linear algebra


def smith_with_row_transform(mat: list[list[int]]):
    """Smith normal form of an integer matrix, tracking the LEFT transform.

    Returns (invariant_factors, U) with U unimodular (rows x rows) such that
    U * mat * V = diag(invariant_factors) for some unimodular V.  Only U is
    needed: the column lattice Lambda = mat * Z^cols equals
    Lambda = (+)_i s_i * (U^{-1} e_i) Z, so the dual lattice
        N = { x : <x, lambda> in Z for all lambda in Lambda }
    has basis  (row i of U) / s_i.
    """
    a = [row[:] for row in mat]
    rows, cols = len(a), (len(a[0]) if a else 0)
    u = [[1 if i == j else 0 for j in range(rows)] for i in range(rows)]
    res: list[int] = []
    top, left = 0, 0
    while top < rows and left < cols:
        # find pivot
        piv = None
        best = None
        for i in range(top, rows):
            for j in range(left, cols):
                if a[i][j]:
                    v = abs(a[i][j])
                    if best is None or v < best:
                        best, piv = v, (i, j)
        if piv is None:
            break
        pi, pj = piv
        a[top], a[pi] = a[pi], a[top]
        u[top], u[pi] = u[pi], u[top]
        for r in range(rows):
            a[r][left], a[r][pj] = a[r][pj], a[r][left]
        while True:
            changed = False
            for i in range(top + 1, rows):
                if a[i][left]:
                    q = a[i][left] // a[top][left]
                    for j in range(left, cols):
                        a[i][j] -= q * a[top][j]
                    for j in range(rows):
                        u[i][j] -= q * u[top][j]
                    if a[i][left]:
                        a[top], a[i] = a[i], a[top]
                        u[top], u[i] = u[i], u[top]
                        changed = True
            for j in range(left + 1, cols):
                if a[top][j]:
                    q = a[top][j] // a[top][left]
                    for i in range(top, rows):
                        a[i][j] -= q * a[i][left]
                    if a[top][j]:
                        for i in range(top, rows):
                            a[i][left], a[i][j] = a[i][j], a[i][left]
                        changed = True
            if not changed:
                break
        if a[top][left] < 0:
            a[top] = [-x for x in a[top]]
            u[top] = [-x for x in u[top]]
        res.append(a[top][left])
        top += 1
        left += 1
    return res, u


def smith_diagonal(mat: list[list[int]]) -> list[int]:
    """Invariant factors (divisibility chain enforced) of an integer matrix."""
    import math

    res, _ = smith_with_row_transform(mat)
    res = [abs(x) for x in res]
    changed = True
    while changed:
        changed = False
        for i in range(len(res) - 1):
            x, y = res[i], res[i + 1]
            if y % x:
                g = math.gcd(x, y)
                res[i], res[i + 1] = g, x * y // g
                changed = True
    return res


def hnf_rows(mat: list[list[int]]) -> list[list[int]]:
    """Row-style Hermite form; returns a basis (list of rows) of the row lattice."""
    a = [row[:] for row in mat if any(row)]
    if not a:
        return []
    cols = len(a[0])
    basis: list[list[int]] = []
    row = 0
    for col in range(cols):
        # find rows with non-zero entry in this column among a[row:]
        while True:
            nz = [i for i in range(row, len(a)) if a[i][col]]
            if len(nz) <= 1:
                break
            nz.sort(key=lambda i: abs(a[i][col]))
            p = nz[0]
            for i in nz[1:]:
                q = a[i][col] // a[p][col]
                for j in range(cols):
                    a[i][j] -= q * a[p][j]
        nz = [i for i in range(row, len(a)) if a[i][col]]
        if not nz:
            continue
        p = nz[0]
        a[row], a[p] = a[p], a[row]
        if a[row][col] < 0:
            a[row] = [-x for x in a[row]]
        row += 1
    return [r for r in a[:row] if any(r)]


# ---------------------------------------------------------------- the columns


def columns_A123(conf, c2_vec, e2_rows) -> tuple[list[list[int]], dict]:
    """Frozen audit columns: PD(H_j H_k), PD(c_2), degree-one stable maps."""
    rho = len(conf)
    triples = triple_numbers(conf)
    cols: list[list[int]] = []
    for j, k in itertools.combinations_with_replacement(range(rho), 2):
        cols.append([d_lookup(triples, i, j, k) for i in range(rho)])
    n_div = len(cols)
    cols.append(list(c2_vec))
    deg1_rows = []
    for row in e2_rows:
        if not row.get("uniform_vertical_curve_witness", False):
            continue
        if row["kind"] == "type1_square_linear_system":
            val, field = row.get("degree_one_GW"), "degree_one_GW"
        elif row["kind"] == "type2_quadratic_linear_system":
            val, field = row.get("degree_one_GW_line_class_Ci"), "degree_one_GW_line_class_Ci"
        else:
            continue
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            continue
        i = row["row_1based"] - 1
        v = [0] * rho
        v[i] = 1
        cols.append(v)
        deg1_rows.append({"row_1based": i + 1, "witness_field": field, "witness_value": val})
    prov = {
        "PD_Hj_cup_Hk_column_count": n_div,
        "PD_c2_column_count": 1,
        "degree_one_stable_map_column_count": len(deg1_rows),
        "degree_one_stable_map_rows": deg1_rows,
    }
    return cols, prov


def columns_A4(conf, e2_rows) -> tuple[list[list[int]], list[dict]]:
    """Type-2 Porteous conic class 2C_i (deformation invariant, uniform)."""
    rho = len(conf)
    cols, prov = [], []
    for row in e2_rows:
        if row["kind"] != "type2_quadratic_linear_system":
            continue
        n = row.get("porteous_linear_corank_degree_conic_class_2Ci")
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            continue
        i = row["row_1based"] - 1
        v = [0] * rho
        v[i] = 2
        cols.append(v)
        prov.append(
            {
                "row_1based": i + 1,
                "witness_field": "porteous_linear_corank_degree_conic_class_2Ci",
                "witness_value": n,
                "class_degree_vector": v,
            }
        )
    return cols, prov


def columns_A6(conf) -> tuple[list[list[int]], list[dict]]:
    """Proposition C: unit column e_i for every row with L_i != 0."""
    try:
        from .lines import line_number
    except ImportError:  # pragma: no cover - retained for direct script execution
        from lines import line_number

    rho = len(conf)
    cols, prov = [], []
    for i in range(rho):
        val = line_number(conf, i)
        if val == 0:
            continue
        v = [0] * rho
        v[i] = 1
        cols.append(v)
        prov.append(
            {
                "row_1based": i + 1,
                "witness_field": "grassmannian_line_euler_number_L_i",
                "witness_value": val,
                "class_degree_vector": v,
                "uniform_scope": True,
            }
        )
    return cols, prov


def columns_A5(conf) -> tuple[list[list[int]], list[dict]]:
    """Proposition B vertical fibre curves for every proper subset of rows."""
    rho = len(conf)
    dims = ambient_dims(conf)
    cols_by_vec: dict[tuple[int, ...], dict] = {}
    cols_mat = list(zip(*conf))
    for size in range(1, rho):  # proper subsets only; S = all rows reproduces A1
        for S in itertools.combinations(range(rho), size):
            Sl = list(S)
            nS = sum(dims[j] for j in Sl)
            supp = [a for a, col in enumerate(cols_mat) if any(col[j] > 0 for j in Sl)]
            mS = len(supp)
            slack = nS - 1 - mS
            if slack < 0:
                continue
            capS = tuple(dims[j] for j in Sl)
            # distribute `slack` extra generic hyperplanes among the factors of S
            for extra in itertools.combinations_with_replacement(range(len(Sl)), slack):
                e = [0] * len(Sl)
                for t in extra:
                    e[t] += 1
                if any(e[t] > capS[t] for t in range(len(Sl))):
                    continue
                poly = {(0,) * len(Sl): 1}
                for a in supp:
                    w = tuple(cols_mat[a][j] for j in Sl)
                    poly = poly_mul_linear(poly, w, capS)
                    if not poly:
                        break
                if not poly:
                    continue
                for t in range(len(Sl)):
                    for _ in range(e[t]):
                        w = tuple(1 if u == t else 0 for u in range(len(Sl)))
                        poly = poly_mul_linear(poly, w, capS)
                        if not poly:
                            break
                    if not poly:
                        break
                if not poly:  # gamma = 0, no conclusion
                    continue
                vec = [0] * rho
                for t, j in enumerate(Sl):
                    exp = list(capS)
                    exp[t] -= 1
                    if exp[t] < 0:
                        vec[j] = 0
                        continue
                    vec[j] = poly.get(tuple(exp), 0)
                if not any(vec):
                    continue
                key = tuple(vec)
                if key not in cols_by_vec:
                    cols_by_vec[key] = {
                        "rows_S_1based": [j + 1 for j in Sl],
                        "m_S": mS,
                        "n_S": nS,
                        "extra_hyperplanes": {str(Sl[t] + 1): e[t] for t in range(len(Sl)) if e[t]},
                        "class_degree_vector": list(vec),
                        "uniform_scope": size == 1 and slack == 0,
                    }
    cols = [list(k) for k in cols_by_vec]
    prov = list(cols_by_vec.values())
    return cols, prov


# --------------------------------------------------- lattice N and the HRR test


def dual_lattice_quotient(cols: list[list[int]], rho: int):
    """Return (invariant_factors, coset reps of N/L as Fraction vectors).

    N = { x in L(x)Q : <x, C_alpha> in Z for every column C_alpha }.
    With  U * P * V = diag(s_1..s_rho)  for the rho x ncols pairing matrix P,
    a basis of N is  b_i = (row i of U) / s_i, and  N/L = (+)_i Z/s_i, so the
    enumeration is over prod_i s_i elements, never over g^rho.
    """
    mat = [[cols[a][i] for a in range(len(cols))] for i in range(rho)]  # rho x ncols
    inv = smith_diagonal(mat)
    if len(inv) < rho:
        return inv, None  # not full rank: N is not a lattice
    raw, u = smith_with_row_transform(mat)
    basis = [[Fraction(u[i][j], raw[i]) for j in range(rho)] for i in range(rho)]
    # verify the basis really is the dual lattice: it must contain L = Z^rho
    # and pair integrally with every column
    for b in basis:
        for a in range(len(cols)):
            assert sum(b[j] * cols[a][j] for j in range(rho)).denominator == 1
    orders = [abs(x) for x in raw]
    reps = []
    for k in itertools.product(*[range(o) for o in orders]):
        x = [Fraction(0)] * rho
        for i, ki in enumerate(k):
            if ki:
                for j in range(rho):
                    x[j] += ki * basis[i][j]
        reps.append(tuple(xx - int(xx) if xx >= 0 else xx - (int(xx) - 1) for xx in x))
    # reduce modulo L and de-duplicate
    seen = {}
    for r in reps:
        key = tuple(Fraction(x.numerator % x.denominator, x.denominator) for x in r)
        seen[key] = None
    return inv, list(seen)


def lattice_basis_from(v: tuple[Fraction, ...], rho: int) -> list[list[Fraction]]:
    """Basis of L + Z v  (L = Z^rho)."""
    den = 1
    for x in v:
        den = den * x.denominator // __import__("math").gcd(den, x.denominator)
    rows = [[den if i == j else 0 for j in range(rho)] for i in range(rho)]
    rows.append([int(x * den) for x in v])
    hb = hnf_rows(rows)
    return [[Fraction(e, den) for e in r] for r in hb]


def nonzero_triples(triples, rho):
    return [
        (r, s, t, d_lookup(triples, r, s, t))
        for r in range(rho)
        for s in range(rho)
        for t in range(rho)
        if d_lookup(triples, r, s, t)
    ]


def chi_value(v: list[Fraction], nz, c2_vec, rho) -> Fraction:
    cube = Fraction(0)
    for r, s, t, dv in nz:
        cube += dv * v[r] * v[s] * v[t]
    lin = sum(Fraction(c2_vec[i]) * v[i] for i in range(rho))
    return cube / 6 + lin / 12


def lattice_passes_integrality(basis, triples, c2_vec, rho) -> tuple[bool, str]:
    k = len(basis)
    for a in range(k):
        s = sum(Fraction(c2_vec[i]) * basis[a][i] for i in range(rho))
        if s.denominator != 1:
            return False, "c2_non_integral"
    nz = nonzero_triples(triples, rho)
    for a, b, c in itertools.combinations_with_replacement(range(k), 3):
        tot = Fraction(0)
        for r, s, t, dv in nz:
            tot += dv * basis[a][r] * basis[b][s] * basis[c][t]
        if tot.denominator != 1:
            return False, "triple_intersection_non_integral"
    # chi is an integer-valued cubic, so integrality on the whole lattice is
    # equivalent to integrality on the degree-<=3 simplex of the basis, i.e. on
    # the C(k+3,3) points  sum_a coeffs[a] b_a  with coeffs >= 0, sum <= 3.
    for m in range(4):
        for pick in itertools.combinations_with_replacement(range(k), m):
            coeffs = [0] * k
            for a in pick:
                coeffs[a] += 1
            v = [sum(Fraction(coeffs[a]) * basis[a][i] for a in range(k)) for i in range(rho)]
            if chi_value(v, nz, c2_vec, rho).denominator != 1:
                return False, "chi_non_integral"
    return True, "passes"


def prime_order_reps(reps, rho):
    """Coset reps of prime order in N/L."""
    out = []
    for v in reps:
        den = 1
        for x in v:
            den = den * x.denominator // __import__("math").gcd(den, x.denominator)
        if den <= 1:
            continue
        # order of v in N/L is den
        p = den
        isprime = p > 1 and all(p % q for q in range(2, int(p**0.5) + 1))
        if isprime:
            out.append(v)
    return out


# ---------------------------------------------------------------------- driver


def audit(dsrec, e2rec, mode: str, do_hrr: bool = True) -> dict:
    conf = dsrec["original_selected_conf"]
    rho = len(conf)
    triples = triple_numbers(conf)
    c2_own = c2_from_conf(conf)
    c2_supplied = dsrec.get("official_fields", {}).get("C2_selected")
    if c2_supplied is not None and c2_own != [int(value) for value in c2_supplied]:
        raise ValueError(
            f"Num {dsrec['Num']}: supplied c2 disagrees with configuration"
        )

    cols, prov = columns_A123(conf, c2_own, e2rec["original_rows"])
    prov4, prov5, prov6 = [], [], []
    if mode in ("A4", "A45", "A456", "A6"):
        c4, prov4 = columns_A4(conf, e2rec["original_rows"])
        cols = cols + c4
    if mode in ("A45", "A456"):
        c5, prov5 = columns_A5(conf)
        cols = cols + c5
    if mode in ("A456", "A6"):
        c6, prov6 = columns_A6(conf)
        cols = cols + c6

    inv, reps = dual_lattice_quotient(cols, rho)
    result = {
        "Num": dsrec["Num"],
        "source": dsrec["source"],
        "rho": rho,
        "conf": conf,
        "mode": mode,
        "smith_invariant_factors": inv,
        "index_N_over_L": None,
        "snf_unit": len(inv) == rho and all(x == 1 for x in inv),
        "conic_columns": prov4,
        "vertical_fibre_columns": prov5,
        "vertical_line_columns": prov6,
    }
    if reps is None:
        result["status"] = "rank_deficient_known_columns"
        return result
    idx = 1
    for s in inv:
        idx *= s
    result["index_N_over_L"] = idx
    if result["snf_unit"]:
        result["status"] = "pass_snf_unit_invariants"
        result["overlattice_excluded"] = True
        return result

    if not do_hrr:
        result["status"] = "unverified_known_columns_not_saturated"
        result["overlattice_excluded"] = None
        return result

    survivors = []
    for v in prime_order_reps(reps, rho):
        basis = lattice_basis_from(v, rho)
        ok, why = lattice_passes_integrality(basis, triples, c2_own, rho)
        if ok:
            survivors.append({"rep": [str(x) for x in v], "reason": why})
    result["prime_order_candidates"] = len(prime_order_reps(reps, rho))
    result["surviving_overlattices"] = survivors
    result["overlattice_excluded"] = not survivors
    result["status"] = (
        "pass_hrr_overlattice_exclusion" if not survivors else "unverified_overlattice_survives"
    )
    return result

