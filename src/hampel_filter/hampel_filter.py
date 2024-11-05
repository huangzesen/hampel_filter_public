from numba import jit, njit, prange
import numpy as np
import pandas as pd

@jit(nopython=True)
def calc_medians(window_size, arr, medians): 
    for i in range(window_size, len(arr)-window_size, 1):
        id0 = i - window_size
        id1 = i + window_size
        median = np.median(arr[id0:id1])
        medians[i] = median

@jit(nopython=True)
def calc_medians_std(window_size, arr, medians, medians_diff): 
    k = 1.4826
    for i in range(window_size, len(arr)-window_size, 1):
        id0 = i - window_size
        id1 = i + window_size
        x = arr[id0:id1]
        medians_diff[i] = k * np.median(np.abs(x - np.median(x)))
        
        
@njit(parallel=True) 
def calc_medians_parallel(window_size, arr, medians): 
    for i in prange(window_size, len(arr)-window_size, 1):
        id0 = i - window_size
        id1 = i + window_size
        median = np.median(arr[id0:id1])
        medians[i] = median

@njit(parallel=True) 
def calc_medians_std_parallel(window_size, arr, medians, medians_diff): 
    k = 1.4826
    for i in prange(window_size, len(arr)-window_size, 1):
        id0 = i - window_size
        id1 = i + window_size
        x = arr[id0:id1]
        medians_diff[i] = k * np.median(np.abs(x - np.median(x)))
        
        
def hampel(arr, window_size=5, n=3, parallel=False, return_indices=True):

    if isinstance(arr, np.ndarray):
        pass
    elif isinstance(arr, pd.Series):
        arr = arr.values
    elif isinstance(arr, pd.DataFrame):
        arr = arr.values
    else:
        raise ValueError("arr must be a numpy array or pandas Series or DataFrame!")
    
    medians = np.ones_like(arr, dtype=float)*np.nan
    medians_diff = np.ones_like(arr, dtype=float)*np.nan
    if parallel:
        calc_medians_parallel(window_size, arr, medians)
        calc_medians_std_parallel(window_size, arr, medians, medians_diff)
    else:
        calc_medians(window_size, arr, medians)
        calc_medians_std(window_size, arr, medians, medians_diff)
    
    ind = np.abs(arr - medians) > n*(medians_diff)
    outlier_indices = np.where(ind)
    if return_indices:
        return outlier_indices
    else:
        return np.squeeze(ind)