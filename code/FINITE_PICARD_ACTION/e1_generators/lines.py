#!/usr/bin/env python3
"""Vertical-line Euler numbers L_i for every row of a CICY configuration.

For a row i let  B = prod_{j != i} P^{n_j}  and consider the parameter space
    P = G(2, n_i+1) x B      (2-planes in C^{n_i+1} = lines in P^{n_i}).
For each defining column a, the restriction of f_a to {p} x ell is a section of
    E_a = Sym^{q_{a,i}} S^*  |X|  O_B(q_a|_B),      rank q_{a,i}+1,
where S is the tautological subbundle.  X contains the line {p} x ell exactly
when all these sections vanish.  Ranks and dimensions always agree:
    dim P   = 2(n_i-1) + (n_tot - n_i) = n_i + n_tot - 2,
    sum_a rank E_a = sum_a q_{a,i} + K = (n_i+1) + (n_tot-3) = n_i + n_tot - 2.
Hence L_i := int_P prod_a c_top(E_a) is a well defined integer, and

    L_i != 0  ==>  the zero scheme is NON-EMPTY,

because the localized top Chern class of a bundle lives in the Chow group of
the zero scheme of the section, which vanishes when that scheme is empty.  No
transversality, genericity or deformation argument is needed: the conclusion
holds for EVERY member of the family.  A resulting line ell satisfies
    J_i . ell = 1,   J_j . ell = 0  (j != i),
so it contributes the unit pairing column e_i.

Grassmannian integration: for a symmetric P(x,y) of degree 2(n-1),
    int_{G(2,n+1)} P = [x^{n-1} y^{n-1}] P - [x^n y^{n-2}] P.
"""


def pmul(left, right, target):
    out = {}
    for lm, lc in left.items():
        for rm, rc in right.items():
            mono = tuple(a + b for a, b in zip(lm, rm))
            if all(e <= t for e, t in zip(mono, target)):
                out[mono] = out.get(mono, 0) + lc * rc
    return {k: v for k, v in out.items() if v}


def linear_poly(weights):
    out = {}
    for idx, c in enumerate(weights):
        if c:
            mono = [0] * len(weights)
            mono[idx] = 1
            out[tuple(mono)] = c
    return out


def product_of_linears(forms, target):
    res = {(0,) * len(target): 1}
    for w in forms:
        res = pmul(res, linear_poly(w), target)
        if not res:
            break
    return res


def line_number(conf, row_index):
    dims = [sum(r) - 1 for r in conf]
    n = dims[row_index]
    base_rows = [i for i in range(len(conf)) if i != row_index]
    base_target = tuple(dims[i] for i in base_rows)
    cols = list(zip(*conf))
    base_linears = [tuple(c[i] for i in base_rows) for c in cols]
    row_degrees = conf[row_index]
    target = (n, n - 1) + base_target
    factors = []
    for degree, bl in zip(row_degrees, base_linears):
        for r in range(degree + 1):
            factors.append((r, degree - r) + bl)
    poly = product_of_linears(factors, target)
    first = poly.get((n - 1, n - 1) + base_target, 0)
    second = 0 if n == 1 else poly.get((n, n - 2) + base_target, 0)
    return first - second
