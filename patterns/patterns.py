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


def create_mp(df, motif_len, column,path,dask=True):
   """Create and Save a univariate/multidimensional matrix profile as a pair of npz files. Input is based on the output of (https://stumpy.readthedocs.io/en/latest/api.html#mstump)
    :param df: The DataFrame that contains the multidimensional time series. 
    :param motif_len: The subsequence window size. 
    :param columns: A list of the column indexes that are included in the comptutation univariate/multidimensional profile.
    :param path: Path of the directory where the file will be saved.
    :param dask: A Dask Distributed client that is connected to a Dask scheduler and Dask workers
    """
    column1=str(column)
    if len(column1)<2:
        if dask==True:
            from dask.distributed import Client, LocalCluster
            with Client(scheduler_port=8782, dashboard_address=None, processes=False, n_workers=4, threads_per_worker=2, memory_limit='50GB') as dask_client:
                mps=stumped(dask_client, df.iloc[:,column], motif_len)# Note that a dask client is needed
                if(path):
                    np.savez_compressed(path,mp=mps[:,0],mpi=mps[:,1] )
                print('Univariate with Dask')
                return mps[:,0],mps[:,1]
        
        mps = stump(df.iloc[:,column], motif_len)
        if(path):
            np.savez_compressed(path, mp=mps[:,0],mpi=mps[:,1])
        print('Uvivariate without Dask')
        return mps[:,0],mps[:,1]

    else:
        if dask==True:
            from dask.distributed import Client, LocalCluster
            with Client(scheduler_port=8782, dashboard_address=None, processes=False, n_workers=4, threads_per_worker=2, memory_limit='50GB') as dask_client:
                mps,indices = mstumped(dask_client, df.iloc[:,column], motif_len)  # Note that a dask client is needed
                if(path):
                    np.savez_compressed(path, mp=mps, mpi=indices)
            print('Multivariate with Dask')
            return mps, indices
        
        mps,indices = mstump(df.iloc[:,column], motif_len) 
        if(path):
            np.savez_compressed(path, mp=mps, mpi=indices)
        print('Multivariate without Dask')
        return mps, indices

      
def load_mp(path):
  Load the Univariate/Multivariate Matrix profile which was saved from Create_mp in a .npz file.
  :param path: Path of the directory where the file is saved.
  
    mp={}
    mpi={}
    loaded = np.load(path + ".npz", allow_pickle=True)
    mp = loaded['mp']
    mpi = loaded['mpi']
    return mp, mpi

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
            
def pairwise_dist(q1, q2):
    min_dist = float('inf')
    m = len(q1)
    _, nn_dist = find_neighbors(q1, q2, m, exclusion_zone=None, min_dist = min_dist, max_neighbors=1)
    pair_dist = nn_dist[0]
    return pair_dist
  
  
def compute_change_points(mpi, L,change_points=4,path):
    """ Calculation of total change points we want to divide our region. 
        In return we provide the locations(indexes) of change_points and the arc-curve which are contained in a specific L.
        
    :param mpi: The one-dimensional matrix profile index where the array corresponds to the matrix profile index for a given dimension.
    :param L: The subsequence length that is set roughly to be one period length. This is likely to be the same value as the motif_len, 
              used to compute the matrix profile and matrix profile index.
    :param change_points: Number of segments that our space is going to be divided.
    :param path: Path of the directory where the file will be saved.
    """
    #global files
    regimes = [change_points]
    output = dict()
    print("Computing regimes..")
    for l in tqdm(L):
        output[l] = [fluss(mpi, L=int(l), n_regimes=int(r), excl_factor=5) for r in regimes] 
    if(path):
        np.save(path, output)
        
    print("Done")
    return output
  
 
def change_points_md(mpi,k_optimal,L=[100,200],change_points=4,paths): ##TODO
  """ Calculation of total change points we want to divide our region and build it with respect up to optimal dimensions given from elbow_method.
      In return we provide the locations(indexes) of change_points and the arc-curve which are contained in a specific L.
        
    :param mpi: The one-dimensional matrix profile index where the array corresponds to the matrix profile index for a given dimension.
    :param k_optimal: Choose optimal dimension(s) given from the elbow method
    :param L: The subsequence length that is set roughly to be one period length. This is likely to be the same value as the motif_len, 
              used to compute the matrix profile and matrix profile index.
    :param change_points: Number of segments that our space is going to be divided.
    :param path: Path of the directory where the file will be saved.
    """
    no_cols = np.arange(1, k_optimal + 1, 1)
    if(L == None):
        L = np.arange(1000,50000, 1000).astype(int)
    regimes = [change_points]
    output = dict()
    for c in tqdm(no_cols):
        output[c] = [fluss(mpi[c - 1], L=int(l), n_regimes=int(r), excl_factor=5) for r in regimes for l in L]
    if(path):
        np.save(path, output)
    return output
  
  def compute_regimes_for_optimal_dim(mpi,k_optimal, L=[100,200], change_points=4,path):
    """ Calculation of total change points we want to divide our region and build it with respect up to *the* optimal dimension given from elbow_method.
      In return we provide the locations(indexes) of change_points and the arc-curve which are contained in a specific L.
        
    :param mpi: The one-dimensional matrix profile index where the array corresponds to the matrix profile index for a given dimension.
    :param k_optimal: Choose *the* optimal dimension from the elbow method
    :param L: The subsequence length that is set roughly to be one period length. This is likely to be the same value as the motif_len, 
              used to compute the matrix profile and matrix profile index.
    :param change_points: Number of segments that our space is going to be divided.
    :param path: Path of the directory where the file will be saved.
    """
   
    if(L == None):
        L = np.arange(1000,50000, 1000).astype(int)
        
    regimes = [change_points]
    output = dict()
    for l in tqdm(L):
        output[l] = [fluss(mpi[k_optimal - 1], L=int(l), n_regimes=int(r), excl_factor=5) for r in regimes]
    if(path):
        np.save(path, output)
    return output
