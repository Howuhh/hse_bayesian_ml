import numpy as np
from scipy.stats import binom, poisson

# There should be no main() in this file!!!
# Nothing should start running when you import this file somewhere.
# You may add other supporting functions to this file.
#
# Important rules:
# 1) Function pa_bc must return tensor which has dimensions (#a, #b, #c),
#    where #v is a number of different values of the variable v.
#    For input variables #v = how many input values of this variable you give to the function.
#    For output variables #v = number of all possible values of this variable.
#    Ex. for pb_a: #b = bmax-bmin+1,   #a is arbitrary.
# 2) Random variables in function names must be written in alphabetic order
#    e.g. pda_cb is an improper function name (pad_bc must be used instead)
# 3) Single dimension must be explicitly stated:
#    if you give only one value of a variable a to the function pb_a, i.e. #a=1, 
#    then the function pb_a must return tensor of shape (#b, 1), not (#b,).
#
# The format of all the functions for distributions is the following:
# Inputs:
# params - dictionary with keys 'amin', 'amax', 'bmin', 'bmax', 'p1', 'p2', 'p3'
# model - model number, number from 1 to 4
# all other parameters - values of the conditions (variables a, b, c, d).
#                        Numpy vectors of size (k,), where k is an arbitrary number.
#                        For variant 3: c and d must be numpy arrays of size (k,N),
#                        where N is a number of lectures.
# Outputs:
# prob, val
# prob - probabilities for different values of the output variable with different input conditions
#        prob[i,...] = p(v=val[i]|...)
# val - support of a distribution, numpy vector of size (#v,) for variable v
#
# Example 1:
#    Function pc_ab - distribution p(c|a,b)
#    Input: a of size (k_a,) and b of size (k_b,)
#    Result: prob of size (cmax-cmin+1,k_a,k_b), val of size (cmax-cmin+1,) 
#
# Example 2 (for variant 3):
#    Function pb_ad - distribution p(b|a,d_1,...,d_N)
#    Input: a of size (k_a,) and d of size (k_d,N)
#    Result: prob of size (bmax-bmin+1,k_a,k_d), val of size (bmax-bmin+1,)
#
# The format the generation function from variant 3 is the following:
# Inputs:
# N - how many points to generate
# all other inputs have the same format as earlier
# Outputs:
# d - generated values of d, numpy array of size (N,#a,#b)

def pa_const(params):
    return 1 / (params["amax"] - params["amin"] + 1)


def pb_const(params):
    return 1 / (params["bmax"] - params["bmin"] + 1)


def pc_ab(a, b, params, model):
    cmax = params["amax"] + params["bmax"]

    a, b = a.reshape(1, -1), b.reshape(1, -1)  # (1, a), (1, b)
    k = np.arange(cmax + 1).reshape(-1, 1)  # (cmax, 1)

    if model == 1 or model == 3:
        pmf_a = binom.pmf(k, a, params["p1"])  # (cmax, a)
        pmf_b = binom.pmf(k, b, params["p2"])  # (cmax, b)
        prob = np.empty((cmax + 1, a.size, b.size))

        for k in range(cmax + 1):
            prob[k] = pmf_a[:k + 1].T @ pmf_b[:k + 1][::-1]  # (a, cmax) @ (cmax, b) = (a, b)
    else:
        means = a.T * params["p1"] + b * params["p2"]  # (a, b)
        prob = poisson.pmf(k[..., None], mu=means[None]) # (cmax + 1, a, b)

    val = np.arange(cmax + 1)

    return prob, val


def pa(params, model):
    n = params["amax"] - params["amin"] + 1
    val = np.arange(params["amin"], params["amax"] + 1)
    prob = np.full(fill_value=1 / n, shape=(n,))
    return prob, val


def pb(params, model):
    n = params["bmax"] - params["bmin"] + 1
    val = np.arange(params["bmin"], params["bmax"] + 1)
    prob = np.full(fill_value=1 / n, shape=(n,))
    return prob, val


def pc(params, model, cache_pc_ab=None):
    a = np.arange(params["amin"], params["amax"] + 1)
    b = np.arange(params["bmin"], params["bmax"] + 1)

    if cache_pc_ab is None:
        dist, val = pc_ab(a, b, params, model)
    else:
        dist, val = cache_pc_ab, np.arange(params["amax"] + params["bmax"] + 1)

    prob = dist.sum((1, 2)) * pa_const(params) * pb_const(params)  # p(c) = sum_{a, b} p(c | a, b) * p(a) * p(b)

    return prob, val


def pc_a(a, params, model, cache_pc_ab=None):
    b = np.arange(params["bmin"], params["bmax"] + 1)

    if cache_pc_ab is None:
        dist, val = pc_ab(a, b, params, model)
    else:
        dist, val = cache_pc_ab, np.arange(params["amax"] + params["bmax"] + 1)

    prob = dist.sum(2) * pb_const(params)  # p(c | a) = sum p(c | a, b) * p(b)

    return prob, val


def pc_b(b, params, model, cache_pc_ab=None):
    a = np.arange(params["amin"], params["amax"] + 1)

    if cache_pc_ab is None:
        dist, val = pc_ab(a, b, params, model)
    else:
        dist, val = cache_pc_ab, np.arange(params["amax"] + params["bmax"] + 1)

    prob = dist.sum(1) * pa_const(params)  # p(c | b) = sum p(c | a, b) * p(a)

    return prob, val


def pd_c(c, params, model):
    cmax = params["amax"] + params["bmax"]
    dmax = 2 * cmax
    val = np.arange(dmax + 1)

    k = np.arange(dmax + 1).reshape(-1, 1) - c.reshape(1, -1) # (dmax, c)
    prob = binom.pmf(k, c, params["p3"])  # (dmax, c)

    return prob, val


def pd_b(b, params, model, cache_pc_ab=None):
    c = np.arange(params["amax"] + params["bmax"] + 1)
    dist_pd_c = pd_c(c, params, model)[0]
    dist_pc_b = pc_b(b, params, model, cache_pc_ab=cache_pc_ab)[0]

    prob = dist_pd_c.dot(dist_pc_b)
    val = np.arange(2 * (params["amax"] + params["bmax"]) + 1)

    return prob, val


def pd(params, model, cache_pc_ab=None):
    c = np.arange(params["amax"] + params["bmax"] + 1)

    dist_pc, _ = pc(params, model, cache_pc_ab=cache_pc_ab)
    dist_pd_c, val = pd_c(c, params, model)

    prob = dist_pd_c.dot(dist_pc)  # p(d) = sum p(d | c) p(c)

    return prob, val


def pb_a(a, params, model):
    b_size = params["bmax"] - params["bmin"] + 1
    prob = np.full(shape=(b_size, a.size), fill_value=pb_const(params))
    val = np.arange(params["bmin"], params["bmax"] + 1)
    return prob, val


def pb_d(d, params, model):
    a = np.arange(params["amin"], params["amax"] + 1)
    b = np.arange(params["bmin"], params["bmax"] + 1)
    cache_pc_ab = pc_ab(a, b, params, model)[0]

    # dist_pd = pd(params, model, cache_pc_ab=cache_pc_ab)[0][d]
    dist_pd_b = pd_b(b, params, model, cache_pc_ab=cache_pc_ab)[0][d]

    prob = (dist_pd_b.T * pb_const(params)) # / dist_pd[None]
    prob = prob / prob.sum(axis=0)  # магия чисел

    val = np.arange(params["bmin"], params["bmax"] + 1)

    return prob, val


def pb_ad(a, d, params, model):
    b = np.arange(params["bmin"], params["bmax"] + 1)
    c = np.arange(params["amax"] + params["bmax"] + 1)

    dist_pd_c = pd_c(c, params, model)[0][d]
    dist_pc_ab = pc_ab(a, b, params, model)[0]
    dist_pc_a = pc_a(a, params, model, cache_pc_ab=dist_pc_ab)[0]

    numerator = dist_pd_c.dot(dist_pc_ab.swapaxes(0, 1)) * pb_const(params)
    denominator = dist_pd_c.dot(dist_pc_a)

    prob = (numerator / denominator[..., None]).transpose(2, 1, 0)  # dab -> bad

    return prob, b