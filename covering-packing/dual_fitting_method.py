# Problem: https://www.youtube.com/watch?v=XmW3xR-0CSE

import numpy as np


class DualFitting(object):
    def __init__(self, inputs):
        self.inputs = self.input_aij_update(inputs)
        self.n_dv = inputs['n_dv']
        self.c_i = inputs['c_i']

    def input_aij_update(self, inputs):
        a_ij = []
        for c in range(len(inputs['a_ij'])):
            a_ij.append([a*b for a, b in zip(inputs['a_ij'][c], list(range(1, 17)))])

        del inputs["a_ij"]
        inputs["a_ij"] = a_ij
        return inputs


inputs = {
    'n_dv': 16,
    'n_cn': 16,
    'c_i': [1] * inputs['n_dv'],
    'a_ij':
        [
            [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
    ],
    'b_j': [1] * inputs['n_dv'],
}

# Initialization
xs = set()
uncovered = set(list(range(1, 17)))
pbm = DualFitting(inputs)

a_ij = pbm.inputs['a_ij']

a_ij_original = a_ij.copy()

while len(uncovered) != 0:
    decision_factor = []
    for rem_sets in a_ij:
        decision_factor.append(len(set([c for c in rem_sets if c > 0]) & uncovered))
    decision_factor = [c for c in decision_factor if c > 0]
    decision = [a/b for a, b in zip(pbm.c_i, decision_factor)]
    decision_min = min(decision)
    index_min_decision = decision.index(decision_min)
    covered = covered | set(a_ij[index_min_decision])
    covered.remove(0)
    uncovered = uncovered.difference(set(a_ij[index_min_decision]))
    xs.add(index_min_decision+1)
    print("uncovered= ", uncovered)
    print("covered= ", covered)
    print("decision_min=", decision_min)
    print("index_min_decision=", index_min_decision)

    a_ij_coverd_nozero = [c for c in a_ij[index_min_decision] if c > 0]
    for cov_nodes in a_ij_coverd_nozero:
        if a_ij_original[cov_nodes-1] in a_ij:
            a_ij.remove(a_ij_original[cov_nodes-1])
    print("a_ij= ", a_ij)
    print("-----------------")

    #del a_ij[index_min_decision]
