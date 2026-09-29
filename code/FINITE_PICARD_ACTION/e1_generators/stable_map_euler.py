#!/usr/bin/env python3
"""Exact genus-zero Euler numbers for square-free CICY curve degrees.

Let A = prod_r P^{n_r} and let X be the zero locus of the CICY bundle

    E = direct_sum_a O_A(q_a).

For a nonzero square-free multi-degree beta, this module computes

    N_beta = integral_[Mbar_0,0(A,beta)] c_top(pi_* ev^* E)

by exact torus localization.  All defining line bundles are convex, and the
Calabi--Yau row-sum condition makes the Euler-class rank equal the dimension
of the stable-map space.  Therefore N_beta != 0 is a uniform, defining-
polynomial-independent certificate that every section of E has a stable map
of total ambient degree beta in its zero scheme.

The current implementation is deliberately restricted to beta_r in {0,1}.
Every nonzero degree occurs on a unique degree-one edge of a torus-fixed tree.
This is enough for the first A7 closure pass and keeps the graph certificates
small and independently checkable.
"""

from __future__ import annotations

import itertools
from collections import defaultdict, deque
from fractions import Fraction
from typing import Iterable, Sequence


Label = tuple[int, ...]
Edge = tuple[Label, Label, int]
GraphKey = tuple[tuple[Label, ...], tuple[Edge, ...]]


def ambient_dimensions(conf: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Infer the projective dimensions from the Calabi--Yau row sums."""

    if not conf or not conf[0]:
        raise ValueError("configuration must be nonempty")
    width = len(conf[0])
    if any(len(row) != width for row in conf):
        raise ValueError("configuration must be rectangular")
    if any(value < 0 for row in conf for value in row):
        raise ValueError("configuration entries must be nonnegative")
    return tuple(sum(row) - 1 for row in conf)


def _prufer_trees(vertex_count: int) -> Iterable[tuple[tuple[int, int], ...]]:
    """Enumerate all labelled trees on range(vertex_count)."""

    if vertex_count == 2:
        yield ((0, 1),)
        return
    for code in itertools.product(range(vertex_count), repeat=vertex_count - 2):
        degree = [1] * vertex_count
        for vertex in code:
            degree[vertex] += 1
        edges: list[tuple[int, int]] = []
        for vertex in code:
            leaf = next(i for i, value in enumerate(degree) if value == 1)
            edges.append((leaf, vertex))
            degree[leaf] -= 1
            degree[vertex] -= 1
        leaves = [i for i, value in enumerate(degree) if value == 1]
        edges.append((leaves[0], leaves[1]))
        yield tuple(edges)


def _canonical_graph(
    vertex_labels: Sequence[Label],
    abstract_edges: Sequence[tuple[int, int]],
    edge_factors: Sequence[int],
) -> GraphKey:
    vertices = tuple(sorted(vertex_labels))
    edges: list[Edge] = []
    for (left, right), factor in zip(abstract_edges, edge_factors):
        left_label = vertex_labels[left]
        right_label = vertex_labels[right]
        if right_label < left_label:
            left_label, right_label = right_label, left_label
        edges.append((left_label, right_label, factor))
    return vertices, tuple(sorted(edges))


def fixed_graphs_squarefree(
    dimensions: Sequence[int], beta: Sequence[int]
) -> list[GraphKey]:
    """Return each torus-fixed decorated graph exactly once.

    Since every positive beta component is one, distinct vertices have
    distinct target labels.  Consequently every decorated graph has trivial
    automorphism group, and no additional graph multiplicity is needed.
    """

    if len(dimensions) != len(beta):
        raise ValueError("dimension and degree vectors have different lengths")
    if any(value not in (0, 1) for value in beta):
        raise ValueError("only square-free degree vectors are supported")
    support = tuple(i for i, value in enumerate(beta) if value)
    if not support:
        raise ValueError("degree zero is not supported")

    vertex_count = len(support) + 1
    fixed_labels = itertools.product(*(range(n + 1) for n in dimensions))
    graph_keys: set[GraphKey] = set()

    for abstract_edges in _prufer_trees(vertex_count):
        adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for edge_index, (left, right) in enumerate(abstract_edges):
            adjacency[left].append((right, edge_index))
            adjacency[right].append((left, edge_index))

        for edge_factors in itertools.permutations(support):
            for root_label in fixed_labels:
                alternatives = [
                    tuple(
                        value
                        for value in range(dimensions[factor] + 1)
                        if value != root_label[factor]
                    )
                    for factor in edge_factors
                ]
                for alternate_values in itertools.product(*alternatives):
                    labels: list[Label | None] = [None] * vertex_count
                    labels[0] = tuple(root_label)
                    queue: deque[int] = deque([0])
                    while queue:
                        vertex = queue.popleft()
                        assert labels[vertex] is not None
                        for neighbor, edge_index in adjacency[vertex]:
                            if labels[neighbor] is not None:
                                continue
                            factor = edge_factors[edge_index]
                            next_label = list(labels[vertex])
                            next_label[factor] = alternate_values[edge_index]
                            labels[neighbor] = tuple(next_label)
                            queue.append(neighbor)
                    graph_keys.add(
                        _canonical_graph(
                            [label for label in labels if label is not None],
                            abstract_edges,
                            edge_factors,
                        )
                    )
            # itertools.product objects are exhausted; rebuild on next pass.
            fixed_labels = itertools.product(*(range(n + 1) for n in dimensions))

    return sorted(graph_keys)


def torus_weights(dimensions: Sequence[int], base: int = 17) -> tuple[tuple[int, ...], ...]:
    """A generic exact specialization of the product-torus characters."""

    if base <= 2 * max(dimensions, default=1) + 3:
        raise ValueError("base is too small for safe signed flag sums")
    return tuple(
        tuple((coordinate + 1) * base**factor for coordinate in range(n + 1))
        for factor, n in enumerate(dimensions)
    )


def _tangent_euler(label: Label, weights: Sequence[Sequence[int]]) -> int:
    answer = 1
    for factor, coordinate in enumerate(label):
        chosen = weights[factor][coordinate]
        for other, value in enumerate(weights[factor]):
            if other != coordinate:
                answer *= chosen - value
    return answer


def _flag_weight(left: Label, right: Label, factor: int, weights: Sequence[Sequence[int]]) -> int:
    return weights[factor][left[factor]] - weights[factor][right[factor]]


def _edge_inverse_h(edge: Edge, weights: Sequence[Sequence[int]]) -> int:
    """Euler of moving H^0(f^*T_A) divided by moving H^1, for d=1."""

    left, right, factor = edge
    left_coordinate = left[factor]
    right_coordinate = right[factor]
    if left_coordinate == right_coordinate:
        raise ValueError("an edge must change its labelled projective factor")
    if any(left[i] != right[i] for i in range(len(left)) if i != factor):
        raise ValueError("an edge may change only its labelled factor")

    left_weight = weights[factor][left_coordinate]
    right_weight = weights[factor][right_coordinate]
    answer = -(left_weight - right_weight) ** 2

    # Normal O(1) directions inside the varying projective factor.
    for coordinate, value in enumerate(weights[factor]):
        if coordinate not in (left_coordinate, right_coordinate):
            answer *= (left_weight - value) * (right_weight - value)

    # Constant O(0) tangent directions from every other product factor.
    for other_factor, coordinate in enumerate(left):
        if other_factor == factor:
            continue
        chosen = weights[other_factor][coordinate]
        for other_coordinate, value in enumerate(weights[other_factor]):
            if other_coordinate != coordinate:
                answer *= chosen - value
    return answer


def _target_localization_factor(graph: GraphKey, weights: Sequence[Sequence[int]]) -> Fraction:
    vertices, edges = graph
    incident: dict[Label, list[tuple[Label, int]]] = defaultdict(list)
    answer = Fraction(1)
    for left, right, factor in edges:
        incident[left].append((right, factor))
        incident[right].append((left, factor))
        answer /= _edge_inverse_h((left, right, factor), weights)

    for vertex in vertices:
        flags = [
            _flag_weight(vertex, neighbor, factor, weights)
            for neighbor, factor in incident[vertex]
        ]
        valence = len(flags)
        if valence == 1:
            answer *= flags[0]
        elif valence == 2:
            answer *= Fraction(_tangent_euler(vertex, weights), flags[0] + flags[1])
        else:
            flag_product = 1
            reciprocal_sum = Fraction(0)
            for flag in flags:
                flag_product *= flag
                reciprocal_sum += Fraction(1, flag)
            answer *= Fraction(
                _tangent_euler(vertex, weights) ** (valence - 1),
                flag_product,
            )
            answer *= reciprocal_sum ** (valence - 3)
    return answer


def _line_bundle_euler(
    graph: GraphKey,
    multidegree: Sequence[int],
    weights: Sequence[Sequence[int]],
) -> Fraction:
    vertices, edges = graph

    def fibre_weight(label: Label) -> int:
        return sum(
            degree * weights[factor][label[factor]]
            for factor, degree in enumerate(multidegree)
        )

    valence = {vertex: 0 for vertex in vertices}
    answer = Fraction(1)
    for left, right, factor in edges:
        valence[left] += 1
        valence[right] += 1
        degree = multidegree[factor]
        left_weight = fibre_weight(left)
        right_weight = fibre_weight(right)
        if degree == 0:
            answer *= left_weight
        else:
            for index in range(degree + 1):
                answer *= Fraction(
                    index * left_weight + (degree - index) * right_weight,
                    degree,
                )
    for vertex, value in valence.items():
        answer /= fibre_weight(vertex) ** (value - 1)
    return answer


def graph_contribution(
    conf: Sequence[Sequence[int]],
    graph: GraphKey,
    weights: Sequence[Sequence[int]],
) -> Fraction:
    columns = tuple(zip(*conf))
    answer = _target_localization_factor(graph, weights)
    for column in columns:
        answer *= _line_bundle_euler(graph, column, weights)
    return answer


def stable_map_euler_number(
    conf: Sequence[Sequence[int]], beta: Sequence[int]
) -> tuple[Fraction, int]:
    """Return (N_beta, number_of_fixed_graphs) using exact arithmetic."""

    dimensions = ambient_dimensions(conf)
    if len(beta) != len(dimensions):
        raise ValueError("degree vector has the wrong rank")
    if sum(dimensions) - len(conf[0]) != 3:
        raise ValueError("configuration is not a threefold")
    weights = torus_weights(dimensions)
    graphs = fixed_graphs_squarefree(dimensions, beta)
    answer = sum(
        (graph_contribution(conf, graph, weights) for graph in graphs),
        Fraction(0),
    )
    return answer, len(graphs)

