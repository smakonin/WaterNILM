"""Reference functions from WaterNILM Untitled6.ipynb cells 7–8.
Source commit a76be1e82920732e0cd259741b11a375cccfce10.
Research: Bradley Ellert, Stephen Makonin, Fred Popowich.
Comments stripped with ast.unparse; algorithm unchanged. Test oracle only.
See docs/PROVENANCE.md. No new license is asserted for this upstream code.
"""
from collections import defaultdict, Counter
from itertools import product

def train(hidden, observed, num, samples, add=1, order=2):
    transition = defaultdict(Counter)
    emission = defaultdict(Counter)
    for start, end in samples:
        hid = [0] * order + hidden[start:end]
        obs = [0] * order + observed[start:end]
        for t in range(order, len(hid)):
            transition[tuple(hid[t - order:t])][hid[t]] += 1
            emission[tuple(hid[t - order + 1:t + 1])][tuple(obs[t - order + 1:t + 1])] += 1
            for x in range(1, order):
                h = tuple(hid[t - x:t]) + (0,) * (order - x)
                emission[h]
                transition[h]
    for i in transition:
        if 0 not in transition[i]:
            transition[i][0] = 1
    for h in emission:
        os = [o for o in product(*[range(num) for x in range(order)]) if emission[h][o] == 0 or emission[h][o] == 1]
        if len(os) == 0:
            continue
        add = sum([emission[h][o] for o in os]) / len(os)
        if add == 0:
            add = 1
        for o in os:
            emission[h][o] = add
    for i in transition:
        total = sum(transition[i].values())
        if total != 0:
            for j in transition[i]:
                transition[i][j] /= total
    for h in emission:
        total = sum(emission[h].values())
        if total != 0:
            for o in emission[h]:
                emission[h][o] /= total
    return (transition, emission)

def viterbi(observed, cap, transition, emission, order=2):
    observed = [0] * order + observed
    cap = [0] * order + cap
    probability = {(0,) * order: 1}
    path = {(0,) * order: []}
    for t in range(order, len(observed)):
        newprobability = {}
        newpath = {}
        for j in range(cap[t] + 1):
            o = tuple(observed[t - order + 1:t + 1])
            p, state = max(((probability[i] * transition[i][j] * emission[i[1:] + (j,)][o], i) for i in probability))
            if p > 0:
                h = state[1:] + (j,)
                newprobability[h] = p
                newpath[h] = path[state] + [j]
        if len(newprobability) == 0:
            for j in range(cap[t] + 1):
                o = tuple(observed[t - order + 1:t + 1])
                print([(t, probability[i], transition[i][j], emission[i[1:] + (j,)][o], i, j, o) for i in probability])
        probability = newprobability
        path = newpath
    prob, state = max(((probability[y], y) for y in probability))
    return path[state]
