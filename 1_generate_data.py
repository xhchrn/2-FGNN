# This script generates random MILP instances for training and testing

import numpy as np
import random as rd
import os
import argparse
from pandas import read_csv
import pyscipopt as scip
from pyscipopt import Model


## ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=100, help="number of samples to generate")
parser.add_argument("--m", type=int, default=6, help='number of constraints')
parser.add_argument("--n", type=int, default=20, help='number of variables')
parser.add_argument("--nnz", type=int, default=60, help='number of nonzero elements in A')
parser.add_argument("--seed", type=int, default=42, help='random seed')
parser.add_argument("--data-dir", type=str, default='data/2fgnn', help='location to save generated data')


class VanillaFullstrongBranchingDataCollector(scip.Branchrule):
    """Branching policy for collecting vanilla strong branching score.
    """
    def __init__(self):
        self.cands = None
        self.scores = None

    def branchinit(self):
        pass

    def branchexeclp(self, allowaddcons):
        result = self.model.executeBranchRule('vanillafullstrong', allowaddcons)
        cands_, scores, _, bestcand = self.model.getVanillafullstrongData()
        self.cands = cands_
        self.scores = scores

        best_var = cands_[bestcand]
        self.model.branchVar(best_var)
        result = scip.SCIP_RESULT.BRANCHED

        return {'result':result}


def generate_unfoldable(m, n, nnz):
    """This function generates and saves unfoldable MILP instances.

    Arguments:
        - m: int, number of constraints
        - n: int, number of variables
        - nnz: int, number of nonzero elements in A
    """
    c = np.random.normal(size=(n,))
    b = np.random.normal(size=(m,))
    circ = np.random.randint(0, 3, size=(m,))
    vtype = np.random.randint(0, 2, size=(n,))
    bounds = np.random.normal(0, 10, size=(n,2))
    bounds.sort(axis=1)
    lb, ub = bounds[:,0], bounds[:,1]

    A = np.zeros((m, n))
    edge_inds = np.zeros((nnz, 2))
    edge_inds_1d = rd.sample(range(m * n), nnz)
    edge_feats = np.random.normal(size=(nnz,))

    for l in range(nnz):
        i = edge_inds_1d[l] // n
        j = edge_inds_1d[l] % n
        edge_inds[l, 0] = i
        edge_inds[l, 1] = j
        A[i, j] = edge_feats[l]

    cons_feats = np.hstack((b.reshape(m, 1),
                            circ.reshape(m, 1)))
    var_feats = np.hstack((c.reshape(n, 1),
                           vtype.reshape(n, 1),
                           lb.reshape(n, 1),
                           ub.reshape(n, 1)))

    opt_model = scip.Model()
    opt_model.hideOutput()

    x_vars = []
    for j in range(n):
        if vtype[j] == 1:
            x_vars.append(opt_model.addVar(vtype="INTEGER", lb=lb[j], ub=ub[j], name=str(j)))
        else:
            x_vars.append(opt_model.addVar(vtype="CONTINUOUS", lb=lb[j], ub=ub[j], name=str(j)))

    for i in range(m):
        # circ[i] is in {0,1,2} for any i
        if circ[i] == 0:
            opt_model.addCons(scip.quicksum(A[i,j] * x_vars[j] for j in range(n)) <= b[i], name="c{0}".format(i))
        elif circ[i] == 1:
            opt_model.addCons(scip.quicksum(A[i,j] * x_vars[j] for j in range(n)) == b[i], name="c{0}".format(i))
        else:
            opt_model.addCons(scip.quicksum(A[i,j] * x_vars[j] for j in range(n)) >= b[i], name="c{0}".format(i))

    opt_model.setObjective(scip.quicksum(x_vars[j] * c[j] for j in range(n)), "minimize")

    return opt_model, x_vars, cons_feats, edge_feats, edge_inds, var_feats


def get_root_strong_branch_scores(model, seed=1812, tol=1e-10):
    set_model_params(model, seed=seed)

    branchrule = VanillaFullstrongBranchingDataCollector()
    # Set `maxdepth` to 0 so that only root branching is executed
    model.includeBranchrule(
        branchrule=branchrule,
        name="Sampling branching rule", desc="",
        priority=666666, maxdepth=0, maxbounddist=1)

    model.setBoolParam('branching/vanillafullstrong/integralcands', True)
    model.setBoolParam('branching/vanillafullstrong/scoreall', True)
    model.setBoolParam('branching/vanillafullstrong/collectscores', True)
    model.setBoolParam('branching/vanillafullstrong/donotbranch', True)
    model.setBoolParam('branching/vanillafullstrong/idempotent', True)

    model.optimize()

    if branchrule.scores is None or branchrule.cands is None:
        if model.getStatus() == "optimal":
            return np.zeros(shape=(model.getNVars(),1))
        else:
            return None

    bs_scores = np.zeros(shape=(model.getNVars(),1))
    for v, s in zip(branchrule.cands, branchrule.scores):
        j = int(v.name.split('_')[-1])
        bs_scores[j] = s if s >= tol else 0.0
    return bs_scores


def set_model_params(model, seed):
    seed = seed % 2147483648  # SCIP seed range

    # set up tolerance
    model.setRealParam('numerics/sumepsilon', 1e-16)

    # set up randomization
    model.setBoolParam('randomization/permutevars', False)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    # separation only at root node
    model.setIntParam('separating/maxrounds', 0)

    # no restart
    model.setIntParam('presolving/maxrestarts', 0)

    # disable presolving
    model.setIntParam('presolving/maxrounds', 0)
    model.setIntParam('presolving/maxrestarts', 0)

    # disable separating (cuts)
    model.setIntParam('separating/maxroundsroot', 0)

    # disable conflict analysis (more cuts)\
    model.setBoolParam('conflict/enable', False)

    # disable primal heuristics
    model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)


if __name__ == "__main__":
    args = parser.parse_args()
    m = args.m
    n = args.n
    nnz = args.nnz
    rd.seed(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)

    total = 0
    adopted = 0
    unexpected = 0
    while adopted < args.num_samples:
        total += 1
        model, x_vars, cons_feats, edge_feats, edge_inds, var_feats = \
            generate_unfoldable(m, n, nnz)

        sb_scores = get_root_strong_branch_scores(model, seed=args.seed+total)

        status = model.getStatus()
        if status not in ['optimal', 'infeasible']:
            print(f'WARNING - unexpected status: {status}')
            unexpected += 1
            continue

        if sb_scores is None:
            continue

        outpath = os.outpath.join(args.data_dir, f'unfoldable_{adopted}')
        os.makedirs(outpath, exist_ok=True)
        model.writeProblem(filename=os.outpath.join(outpath, "model.mps"))
        np.savetxt(os.path.join(outpath, 'ConFeatures.csv'), cons_feats, delimiter = ',', fmt = '%10.5f')
        np.savetxt(os.path.join(outpath, 'EdgeFeatures.csv'), edge_feats, fmt = '%10.5f')
        np.savetxt(os.path.join(outpath, 'EdgeIndices.csv'), edge_inds, delimiter = ',', fmt = '%d')
        np.savetxt(os.path.join(outpath, 'VarFeatures.csv'), var_feats, delimiter = ',', fmt = '%10.5f')

        adopted += 1

    print(f"Ratio of adopted instances: {adopted}/{total}")
    print(f"Ratio of unexpected instances: {unexpected}/{total}")
