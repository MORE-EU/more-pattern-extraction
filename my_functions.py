import os, sys
import glob
import pathlib
from pathlib import Path
import time
from stumpy import stump, fluss, gpu_stump, mstumped, mstump, subspace
import gc
import pyscamp
import pandas as pd
import numpy as np
import numba
from numba import cuda, njit, set_num_threads
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import h5py
import pandas as pd
from sklearn import preprocessing
from numba import cuda, njit, set_num_threads
from matrixprofile import core
import matrixprofile as mpf
from matrixprofile.algorithms import top_k_motifs
from matrixprofile.visualize import visualize_md
import warnings
from matrixprofile.algorithms.mass2 import mass2

def save_mdmp_as_h5(dir_path, name, mps, idx, k=0):
    """Save a multidimensional matrix profile as a pair of hdf5 files. Input is based on the output of (https://stumpy.readthedocs.io/en/latest/api.html#mstump)
    :param dir_path: Path of the directory where the file will be saved.
    :param name: Name that will be appended to the file after a default prefix. (i.e. mp_multivariate_<name>.h5)
    :param mps: The multi-dimensional matrix profile. Each row of the array corresponds to each matrix profile for a given dimension 
                (i.e., the first row is the 1-D matrix profile and the second row is the 2-D matrix profile).
    :param idx: The multi-dimensional matrix profile index where each row of the array corresponds to each matrix profile index for a given dimension.
    :param k: If mps and idx are one-dimensional k can be used to specify the given dimension of the matrix profile. The default value specifies the 1-D matrix profile.
              If mps and idx are multi-dimensional, k is ignored.
    """
    if mps.ndim != idx.ndim:
        err = 'Dimensions of mps and idx should match'
        raise ValueError(f"{err}")
    if mps.ndim == 1:
        mps = mps[None, :]
        idx = idx[None, :]
        h5f = h5py.File(dir_path + 'mp_multivariate_' + name + '.h5','w')
        h5f.create_dataset(f'mp{k}', data=mps[0])
        h5f.close()

        h5f = h5py.File(dir_path + 'index_multivariate_' + name + '.h5','w')
        h5f.create_dataset(f'idx{k}', data=idx[0])
        h5f.close()
        return
    
    h5f = h5py.File(dir_path + 'mp_multivariate_' + name + '.h5','w')
    for i in range(mps.shape[0]):
        h5f.create_dataset(f'mp{i}', data=mps[i])
    h5f.close()

    h5f = h5py.File(dir_path + 'index_multivariate_' + name + '.h5','w')
    for i in range(mps.shape[0]):
        h5f.create_dataset(f'idx{i}', data=idx[i])
    h5f.close()
    return

def load_mdmp_from_h5(dir_path, name, k):
    """Load a multidimensional matrix profile that has been saved as a pair of hdf5 files.
    :param dir_path: Path of the directory where the file is located.
    :param name: Name that follows the default prefix. (i.e. mp_multivariate_<name>.h5)
    :param k: Specifies which K-dimensional matrix profile to load. 
              (i.e. k=2 loads the 2-D matrix profile)
    """
    # Load MP from disk
    
    h5f = h5py.File(dir_path + 'mp_multivariate_' + name + '.h5','r')
    mp= h5f[f'mp{k}'][:]
    h5f.close()

    h5f = h5py.File(dir_path + 'index_multivariate_' + name + '.h5','r')
    index = h5f[f'idx{k}'][:]
    h5f.close()
    return mp, index

def add_noise_to_series(series, noise_max=0.00009):
    
    """ Add uniform noise to series.
    :param series: The time series to be added noise.
    :param noise_max: The upper limit of the amount of noise that can be added to a time series point
    """
    
    if not core.is_array_like(series):
        raise ValueError('series is not array like!')

    temp = np.copy(core.to_np_array(series))
    noise = np.random.uniform(0, noise_max, size=len(temp))
    temp = temp + noise

    return temp

def add_noise_to_series_md(df, noise_max=0.00009):
    
    """ Add uniform noise to a multidimensional time series that is given as a pandas DataFrame.
    :param df: The DataFrame that contains the multidimensional time series.
    :param noise_max: The upper limit of the amount of noise that can be added to a time series point.
    """
    
    for col in df.columns:
        df[col] = add_noise_to_series(df[col].values, noise_max)
    return df
    
def filter_dates(df, start, end):
    """ Remove rows of the dataframe that are not in the [start, end] interval.
    :param df:DataFrame that has a datetime index.
    :param start: Date that signifies the start of the interval.
    :param end: Date that signifies the end of the interval.
    """
    date_range = (df.index >= start) & (df.index <= end)
    df = df[date_range]
    return df

def plot_knee(mps, save_plot=False, filename='knee.png'):
    
    """ Plot the minimum value of the matrix profile for each dimension. This plot is used to visually look for a 'knee' or 'elbow' that
    can be used to find the optimal number of dimensions to use.
    :param mps: The multi-dimensional matrix profile. Each row of the array corresponds to each matrix profile for a given dimension 
                (i.e., the first row is the 1-D matrix profile and the second row is the 2-D matrix profile).
    :param save_plot: If save_plot is True then the figure will be saved. Otherwise it will just be shown.
    :param filename: Used if save_plot=True, the name of the file to be saved.
    """
    
    motifs_idx = np.argsort(mps, axis=1)[:, :2]
    mp_len = mps.shape[0]
    plt.figure(figsize=(15, 5), dpi=80)
    plt.xlabel('k (Number of dimensions, zero-indexed)', fontsize='20')
    plt.ylabel('Matrix Profile Min Value', fontsize='20')
    plt.xticks(range(mp_len))
    plt.plot(mps[range(mp_len), motifs_idx[:mp_len, 0]], c='red', linewidth='4');
    if save_plot:
        plt.savefig(filename)
    else:
        plt.show()
    return

def pick_subspace_columns(df, mps, idx, k, m, include):
    
    """ Given a multi-dimensional time series as a pandas Dataframe, keep only the columns that have been used for the creation of the k-dimensional matrix profile.
    :param df: The DataFrame that contains the multidimensional time series.
    :param mps: The multi-dimensional matrix profile. Each row of the array corresponds to each matrix profile for a given dimension 
                (i.e., the first row is the 1-D matrix profile and the second row is the 2-D matrix profile).
    :param idx: The multi-dimensional matrix profile index where each row of the array corresponds to each matrix profile index for a given dimension.
    :param k: If mps and idx are one-dimensional k can be used to specify the given dimension of the matrix profile. The default value specifies the 1-D matrix profile.
              If mps and idx are multi-dimensional, k is ignored.
    :param m: The subsequence window size. Should be the same as the one used to create the multidimensional matrix profile that is the input.
    :param include: A list of the column names that must be included in the constrained multidimensional motif search.
    """
    
    motifs_idx = np.argsort(mps, axis=1)[:, :2]
    col_indexes = []
    for n in include:
        col_indexes.append(df.columns.get_loc(n))
    
    print(f'Include dimensions: {include}, indexes in df = {col_indexes}')
    S = subspace(df, m, motifs_idx[k][0], idx[k][motifs_idx[k][0]], k, include=col_indexes)
    print(f"For k = {k}, the {k + 1}-dimensional subspace includes subsequences from {df.columns[S].values}")
    subspace_cols = list(df.columns[S].values)
    df = df[subspace_cols]
    return df


def to_mpf(mp, index, window, ts):
    """ Using a matrix profile, a matrix profile index, the window size and the timeseries used to calculate the previous, create a matrix profile object that
        is compatible with the matrix profile foundation library (https://github.com/matrix-profile-foundation/matrixprofile). This is useful for cases where another               library was used to generate the matrix profile.
    :param mp: A matrix profile.
    :param index: The matrix profile index that accompanies the matrix profile.
    :param window: The subsequence window size.
    :param ts: The timeseries that was used to calculate the matrix profile.
    """
    
    
    mp_mpf = mpf.utils.empty_mp()
    mp_mpf['mp'] = np.array(mp)
    mp_mpf['pi'] = np.array(index)
    mp_mpf['metric'] = 'euclidean'
    mp_mpf['w'] = window
    mp_mpf['ez'] = 0
    mp_mpf['join'] = False
    mp_mpf['sample_pct'] = 1
    mp_mpf['data']['ts'] = np.array(ts).astype('d')
    mp_mpf['algorithm']='mpx'
    
    return mp_mpf

def compute_mp_av(mp, index, m, df, k):
    
    """ Given a matrix profile, a matrix profile index, the window size and the DataFrame that contains the timeseries.
        Create a matrix profile object and add the corrected matrix profile after applying the complexity av.
        Uses an extended version of the apply_av function from matrixprofile foundation that is compatible with multi-dimensional timeseries.
        The implementation can be found here (https://github.com/MORE-EU/matrixprofile/blob/master/matrixprofile/transform.py)
    :param mp: A matrix profile.
    :param index: The matrix profile index that accompanies the matrix profile.
    :param window: The subsequence window size.
    :param ts: The timeseries that was used to calculate the matrix profile.
    """
    
    # Apply the annotation vector
    m  = m # window size
    mp = np.nan_to_num(mp, np.nanmax(mp)) # remove nan values
    profile = to_mpf(mp, index, m, df)
    av_type = 'complexity'
    profile = mpf.transform.apply_av(profile, av_type)
    
    return profile


def pattern_loc(start, end, mask, segment_labels):
    
    """ Considering that a time series is characterized by regions belonging to two different labels.
        Return the label name of the region that the pattern is contained in.
    :param start: The starting index of the pattern.
    :param end: The ending index of the pattern. 
    :param mask: Binary mask used to annotate the time series.
    :param segment_labels: List of the two labels that characterize the time series.
    """
    
    if len(segment_labels) != 2:
        raise ValueError('segment_labels must contain exactly 2 labels')
    
    start = start
    end = end
    
    # the first label in the list will be assigned to for the True regions in the mask
    true_label = segment_labels[0]
    
    # the second label in the list will be assigned to for the False regions in the mask
    false_label = segment_labels[1]
    
    if mask[start] == mask[end]:
        if mask[start] == True:
            loc = true_label
        else:
            loc = false_label
    else:
        # if a pattern spans both regions return the label 'both'
        loc = 'both'
        
    return loc

def calc_cost(cl1_len, cl2_len, num_cl1, num_cl2):
    
    """ Assign a cost to a pattern based on if the majority of its occurances are observed
        in regions of a time series that are annotated with the same binary label.
        The cost calculation takes into account a possible difference in the total lengths of the segments.
        Return the label name of the region that the pattern is contained in, as well as the normalized number of occurences.
    :param cl1_len: Total length of the time series that belong to the class 1.
    :param cl2_len: Total length of the time series that belong to the class 2.
    :param num_cl1: Number of occurances of the pattern in regions that belong to cl1.
    :param num_cl2: Number of occurances of the pattern in regions that belong to cl2.
    """
    
    if (num_cl1 + num_cl2 <= 2):
        return 1.0, None, None
    if (cl1_len == 0 or cl2_len == 0):
        return 1.0, None, None
    f = cl1_len / cl2_len
    norm_cl1 = num_cl1 / f
    norm_cl2 = num_cl2
    cost = 1 - (abs(norm_cl1 - norm_cl2 ) / (norm_cl1 + norm_cl2))
    return cost, norm_cl1, norm_cl2

def calculate_motif_stats(p, mask, k, m, ez, radius):
    
    output_list = []
    
    a_len = np.count_nonzero(mask)
    m_len = abs(mask.shape[0] - a_len)
    
    for i in range(0, len(p['motifs'])):
        idx, nn1 = p['motifs'][i]['motifs']
        neighbors = p['motifs'][i]['neighbors']
        motif_pair = p['motifs'][i]['motifs']
        start = idx
        end = idx + m
        nn_idx_start = []
        nn_idx_end = []
        for neighbor in neighbors:
            nn_idx_start.append(neighbor + 1)
            nn_idx_end.append(neighbor + m)
        aligned_count = 0
        misaligned_count  = 0
        spanning_both = 0
        for nn_start, nn_end in zip(nn_idx_start, nn_idx_end):
            location_in_ts = motif_loc(nn_start, nn_end, mask)
            if location_in_ts == 'aligned':
                aligned_count += 1
            elif location_in_ts == 'misaligned':
                misaligned_count += 1
            else:
                spanning_both += 1
                
        motif_location = motif_loc(start, end, mask)
        if motif_location == 'aligned':
            aligned_count += 1
        elif motif_location == 'misaligned':
            misaligned_count += 1
            
        nearest_neighbor_location = motif_loc(nn1, nn1+m, mask)
        if motif_location == 'aligned':
            aligned_count += 1
        elif motif_location == 'misaligned':
            misaligned_count += 1
            
        cost, norm_a, norm_m = calc_cost(m_len, a_len, misaligned_count, aligned_count)
        
        maj = ''
        if norm_a == norm_m:
            maj = 'None'
        elif norm_a is None and norm_m is None:
            maj = 'None'
        elif norm_a > norm_m:
            maj = 'aligned'
        elif norm_a < norm_m:
            maj = 'misaligned'
            
        output_list.append([i+1, motif_location, nearest_neighbor_location, aligned_count, misaligned_count, cost, m, ez, radius, motif_pair, maj])
        
    return output_list

def get_top_k_motifs(df, mp, index, m, ez, radius, k, max_neighbors=50):
    
    np_df = df.to_numpy()

    mp = np.nan_to_num(mp, nan=np.nanmax(mp)) # remove nan values

    profile = to_mpf(mp, index, m, np_df)
    
    exclusion_zone = int(np.floor(m * ez))
    p = top_k_motifs.top_k_motifs(profile, k=k, radius=radius, exclusion_zone=exclusion_zone,  max_neighbors=max_neighbors)
    return p

def save_results(results_dir, sub_dir_name, p, df_stats, m, radius, ez, k, max_neighbors):

    path = os.path.join(results_dir, sub_dir_name)
    
    print(path)

    pathlib.Path(path).mkdir(parents=True, exist_ok=True) 

    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        figs = visualize_md(p)
    
        for i,f in enumerate(figs):
            f.savefig(path + f'/fig{i}.png' , facecolor='white', transparent=False, bbox_inches="tight")
            f.clf()
        
    # remove figures from memory
    plt.close('all')
    gc.collect() 
    
    df_stats.to_csv(path + '/counts.csv')

    lines = [f'Window size (m): {m}',
             f'Radius: {radius} (radius * min_dist)',
             f'Exclusion zone: {ez} * window_size',
             f'Top k motifs: {k}',
             f'Max neighbors: {max_neighbors}']

    with open(path+'/info.txt', 'w') as f:
        for ln in lines:
            f.write(ln + '\n')
            
            
def find_neighbors(query, ts, w, min_dist, exclusion_zone=None, max_neighbors=100, radius=3):
    
    # find subsequences of ts that are similar to query
    
    window_size = w
    ts = ts.T
    query = query.T
    dims = ts.shape[0]
    data_len = ts.shape[1]
    dp_len = data_len - window_size + 1
    
    if exclusion_zone is None:
        exclusion_zone = 0
        #print('No exculsion zone given.')


    # compute distance profile using mass2 for first appearance
    # create the multi dimensional distance profile

    md_distance_profile = np.zeros((dims, dp_len), dtype='complex128')
    for i in range(0, dims):
        ts_i = ts[i, :]
        query_i = query[i, :]
        md_distance_profile[i, :] = mass2(ts_i, query_i)

    D = md_distance_profile
    D.sort(axis=0, kind="mergesort")
    D_prime = np.zeros(dp_len)
    for i in range(dims):
        D_prime = D_prime + D[i]
        D[i, :] = D_prime / (i + 1)

    # reassign to keep compatibility with the rest of the code
    distance_profile = D[dims - 1, :]

    # find up to max_neighbors taking into account the radius and exclusion zone
    neighbors = []
    n_dists = []
    for j in range(max_neighbors):
        neighbor_idx = np.argmin(distance_profile)
        neighbor_dist = distance_profile[neighbor_idx]
        not_in_radius = not ((radius * min_dist) >= neighbor_dist)

        # no more neighbors exist based on radius
        if core.is_nan_inf(neighbor_dist) or not_in_radius:
            break

        # add neighbor and apply exclusion zone
        neighbors.append(neighbor_idx)
        n_dists.append(np.real(neighbor_dist))
        distance_profile = core.apply_exclusion_zone(
            exclusion_zone,
            False,
            window_size,
            data_len,
            neighbor_idx,
            distance_profile
        )
        
    # return the list of neighbor indices and the respective distances
    return neighbors, n_dists

def pairwise_dist(q1, q2):
    min_dist = float('inf')
    m = len(q1)
    _, nn_dist = find_neighbors(q1, q2, m, exclusion_zone=None, min_dist = min_dist, max_neighbors=1)
    pair_dist = nn_dist[0]
    return pair_dist


def calculate_nn_stats(nn, mask, m, ez, maj_other):
    a_len = np.count_nonzero(mask)
    m_len = abs(mask.shape[0] - a_len)
    neighbors = nn
    nn_idx_start = []
    nn_idx_end = []
    for neighbor in neighbors:
        nn_idx_start.append(neighbor)
        nn_idx_end.append(neighbor + m)
        
    aligned_count = 0
    misaligned_count  = 0
    spanning_both = 0
    
    for nn_start, nn_end in zip(nn_idx_start, nn_idx_end):
        location_in_ts = motif_loc(nn_start, nn_end, mask)
        if location_in_ts == 'aligned':
            aligned_count += 1
        elif location_in_ts == 'misaligned':
            misaligned_count += 1
        else:
            spanning_both += 1
    cost, norm_a, norm_m = calc_cost(m_len, a_len, misaligned_count, aligned_count)

    maj = ''
    if norm_a == norm_m:
        maj = 'None'
    elif norm_a is None and norm_m is None:
        maj = 'None'
    elif norm_a > norm_m:
        maj = 'aligned'
    elif norm_a < norm_m:
        maj = 'misaligned'
    matching_maj = (maj_other == maj)
    return [nn, aligned_count, misaligned_count, ez, cost, matching_maj]