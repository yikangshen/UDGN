import numpy as np

def single_root_msa(weights):
    idxs = np.arange(weights.shape[0])
    best_headlist = None
    best_score = float('-inf')
    for j in range(weights.shape[0]):
        headlist = parentdict2parentlist(rooted_msa(weights, idxs, j))
        tree_score = tree_weight(weights, headlist)
        if tree_score > best_score:
            best_score = tree_score
            best_headlist = headlist
    
    # root = -1
    # for i, v in enumerate(best_headlist):
    #     if v == -1:
    #         root = i
    # best_headlist[root] = root
    return best_headlist

def rooted_msa(weights, idxs, root):
    best_parent = np.argmax(weights, axis=1)
    edges = {i:best_parent[i] for i in idxs if i != root}

    def check_cycle():
        cycle = []
        v_ = edges.get(v, None)
        while True:
            v_ = edges.get(v_, None)
            if v_ is None:
                return None
            elif v_ in cycle:
                return np.array(cycle[cycle.index(v_):])
            cycle.append(v_)


    for v in edges.keys():
        C = check_cycle()
        if C is not None:
            # Merge nodes
            new_v = weights.shape[0]
            new_idxs = [i for i in idxs if i not in C]
            new_weights = np.full((weights.shape[0] + 1, weights.shape[1] + 1), float("-inf"))
            new_weights[:-1, :-1] = weights
            new_weights[C] = float("-inf")
            new_weights[:, C] = float("-inf")

            # New parents of contracted vertex
            best_parent_score = weights[C, best_parent[C]]
            cycle_out_weights = weights[C] - best_parent_score[:, None]
            cycle_max_edges = np.argmax(cycle_out_weights, axis=0)
            new_weights[new_v, new_idxs] = cycle_out_weights[cycle_max_edges[new_idxs],
                                                             new_idxs]
            # New children of contracted vertex
            new_weights[new_idxs, new_v] = np.max(weights[new_idxs][:, C], axis=1)
            new_edges = rooted_msa(new_weights, new_idxs + [new_v], root)
            # Replace cycle parent with actual node
            cycle_parent = new_edges[new_v]
            del new_edges[new_v]
            new_edges[C[cycle_max_edges[cycle_parent]]] = cycle_parent
            # Replace original edges in the cycle
            for v in edges:
                if v in new_edges and new_edges[v] == new_v:  # that means parent is in cycle
                    orig_parent = edges[v]
                    del new_edges[v]
                    new_edges[v] = orig_parent
            for v in edges:
                if v not in new_edges:
                    new_edges[v] = edges[v]
            return new_edges
    return edges

def tree_weight(weights, heads):
    idx = np.arange(weights.shape[0])
    w = weights[idx, heads]
    return w[heads != idx].sum()

def parentdict2parentlist(d):
    # headlist = np.full((len(d) + 1), -1)
    headlist = np.arange(0, len(d) + 1)
    for i in d.keys():
        headlist[i] = d[i]
    return headlist


if __name__ == "__main__":
    weights = np.zeros((6, 6)) + float('inf')
    weights[0, 1] = 1
    weights[0, 2] = 10
    weights[0, 5] = 5
    weights[1, 2] = 8
    weights[1, 5] = 2
    weights[2, 4] = 6
    weights[3, 1] = 7
    weights[3, 2] = 4
    weights[4, 3] = 3
    weights[5, 3] = 9
    weights[5, 4] = 11
    weights = -weights.T

    root_weight = np.zeros(6)


    idx = np.arange(weights.shape[0])
    weights[idx, idx] = float('-inf')
    print(single_root_msa(root_weight, weights))