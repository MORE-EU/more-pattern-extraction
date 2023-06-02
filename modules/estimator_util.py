# import pandas as pd
# import numpy as np
from scipy import signal
from modules.learning import *

def compute_residual(index, y_true, y_pred):
        out = pd.DataFrame(index = index)
        out['out'] = y_true - y_pred
        
        return out['out']

def detect_changepoints(index, res, width, t=0.999):
    slopes = pd.DataFrame(index = index)
    slopes['slopes'] = res.rolling(width, center = True).apply(get_slope)

    peak_indexes = signal.find_peaks(slopes['slopes'])
    valley_indexes = signal.find_peaks(-slopes['slopes'])
    
    q = np.quantile(np.abs(slopes['slopes'][np.union1d(peak_indexes[0], valley_indexes[0])]), t)
    array = (np.abs(slopes['slopes'][np.union1d(peak_indexes[0], valley_indexes[0])]))
    return (array[array.values>q].index)

def score_segments(index, res, chps): 
    out = pd.DataFrame(index = index)
    out['score'] = np.nan
    scores = []
    a = min(index)
    b = max(index)
    for i, t in enumerate(chps):
        if i == 0:
            try:
                out['score'].loc[a:t] = my_mce(res.loc[a:t])
            except:
                out['score'].loc[a:t] = np.nan
            scores.append(my_mce(res.loc[a:t]))
        if i == len(chps)-1:   
            try:
                out['score'].loc[t:b] = my_mce(res.loc[t:b])
            except:
                out['score'].loc[t:b] = np.nan    
            scores.append(my_mce(res.loc[t:b]))
        if i < len(chps)-1:   
            
            try:
                out['score'].loc[t:chps[i+1]] = my_mce(res.loc[t:chps[i+1]])
            except:
                out['score'].loc[t:chps[i+1]] = np.nan    
            scores.append(my_mce(res.loc[t:chps[i+1]])) 
    return out, scores


def my_mce(x):
    return np.median(np.power(x,3))

def get_slope(x):
    try: 
        _, slope, _ = get_line_and_slope(np.asarray(x))
    except Exception as e:
        slope = np.nan
    return slope