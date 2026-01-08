import numpy as np

def softmax(x, temp=1.0):
    x = np.asarray(x, dtype=float) / max(1e-9, float(temp))
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))

def option_entropy_score(values, temp=1.0, normalize=True):
    """
    values: list/array of utilities (higher is better)
    returns entropy (optionality). If normalize=True -> divide by log(K).
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    p = softmax(values, temp=temp)
    H = entropy(p)
    if normalize and values.size > 1:
        H = H / np.log(values.size + 1e-12)
    return float(H)

def nbv_score_with_optionality(ig, ret, opt_entropy,
                              w_ig=1.0, w_ret=1.0, w_opt=0.5):
    """
    Simple linear fusion:
      score = w_ig*IG + w_ret*R + w_opt*Hopt
    Assumes IG, R, Hopt are already comparable/scaled.
    """
    return float(w_ig*ig + w_ret*ret + w_opt*opt_entropy)
