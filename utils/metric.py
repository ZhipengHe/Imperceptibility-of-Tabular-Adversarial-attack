import pandas as pd
import numpy as np

from scipy import stats
from scipy.spatial.distance import mahalanobis

def get_lp_norm(arr, p=2):
    # use the numpy linalg.norm function to calculate the lp_norm of the given point
    return np.linalg.norm(arr, ord=p, axis=1)

def get_sparsity(arr):

    return get_lp_norm(arr, p=0)

def get_sparsity_rate(arr):
    
    return (get_sparsity(arr) / arr.shape[-1].astype(np.float32)).astype(np.float32)

def get_proximity_l1(arr):

    return get_lp_norm(arr, p=1)

def get_proximity_l2(arr):

    return get_lp_norm(arr, p=2)

def get_proximity_l2(arr):

    return get_lp_norm(arr, p=np.inf)

def get_proximity_mad(arr):
    '''
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html
    '''

    return stats.median_abs_deviation(arr, axis=1)

def get_proximity_md(arr, arr_adv, V):

    md_list = []

    for index, elem in np.ndenumerate(arr):
        md_list.append(mahalanobis(elem, arr_adv[index], V))

    return np.array(md_list)

def get_perturbation_sensitivity(arr):

    return (1.0 / np.std(arr)).astype(np.float32)

def metric_generator(result, arr, arr_adv):
    pass