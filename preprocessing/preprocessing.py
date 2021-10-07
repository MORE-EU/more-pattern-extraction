import sys
import os
import pickle
import argparse
paths = ['', '..', '../..']
import numpy as np
import pandas as pd
import timeit
import collections
import multiprocessing
import math
import glob
import itertools
from pyts.preprocessing import PowerTransformer
import re
import time
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


def load_df(path): 
    """
        :param path: Path of the under loading DataFrame.
        
        """
   
    df = pd.DataFrame()
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df.set_index(df.index, inplace=True)
    if path.endswith('.csv'):
        df = pd.read_csv(path)
        df.index = pd.to_datetime(df.index)
        df.set_index(df.index, inplace=True)
    return df


def change_granularity(df,granularity='30s',size=10**7,chunk=True): ##TODO Look for one Nan in the dataset and then perform interpolate
    """
        :param df: Date/Time DataFrame. 
        :param size: The size/chunks we want to divide our /DataFrame according to the global index of the set. The Default price is 10 million.       .
        :param granularity: The offset user wants to resample the Time Series                  
        :param chunk: If set True, It applies the chunk_interpolation
        """

    df = df.resample(granularity).mean()
    print('Resample Complete')
    if chunk==True: #Getting rid of NaN occurances.
        df=chunk_interpolate(df,size=size,interpolate=True, method="linear", axis=0,limit_direction="both", limit=1)
        print('Interpolate Complete')
    return df


def filter_col(df, col, less_than=None, bigger_than=None): 
    """
        :param df: Date/Time DataFrame. 
        :param col: The desired column to work on our DataFrame. 
        :param less_than: Filtering the column dropping values below that price.
        :param bigger_than: Filtering the column dropping values above that price.
        """


    if(less_than is not None):
        df=df.drop(df[df.iloc[:,col] < less_than].index)
    if(bigger_than is not None):
        df=df.drop(df[df.iloc[:,col] > bigger_than].index)
    print('Filter Complete')
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


def normalize(df):
    """ 
        :param df: Date/Time DataFrame or any DataFrame given with a specific column to Normalize. 
        """
    values=[]
        # prepare data for normalization
    values = df.values
    values = values.reshape((len(values), 1))
        # train the normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
        # normalize the dataset and print the first 5 rows
    normalized = scaler.transform(values)
    return normalized


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


def filter_df(df, filter_dict):
    """
    Creates a filtered DataFrame with multiple columns.
        
        :param df: Date/Time DataFrame or any Given DataFrame.
        :param filter_dict: A dictionary of columns user wants to filter
        
        """

    mask = np.ones(df.shape[0]).astype(bool)
    for name, item in filter_dict.items():
        val, perc = item
        if val is not None:
            mask = mask & (np.abs(df[name].values - val) < val * perc)
            
    df.loc[~mask, df.columns != df.index] = np.NaN
    f_df = df
    print(f_df.shape)
    return f_df


def multi_corr(df, dep_column):
    """
    Computation of the coefficient of multiple correlation. The input consists of a dataframe and the column corresponding to the dependent variable.
        
        :param df: Date/Time DataFrame or any Given DataFrame.
        :param dep_column: The corresponding the column to the dependent variable.
        
        """
    df_str_corr = df.corr(method='pearson')
    df_str_corr_ind_temp = df_str_corr.drop(index = dep_column)
    df_str_corr_ind = df_str_corr_ind_temp.drop(columns = dep_column)
    df_str_corr_ind_inv = inv(df_str_corr_ind.values)
    df_str_corr_dep = df_str_corr_ind_temp.loc[:,dep_column]
    return np.matmul(np.matmul(np.transpose(df_str_corr_dep.values), df_str_corr_ind_inv),df_str_corr_dep.values)


def chunker(seq, size):
    """
    Dividing a file/DataFrame etc into pieces for better hadling of RAM. 
        
        :param seq: Sequence, Folder, Date/Time DataFrame or any Given DataFrame.
        :param size: The size/chunks we want to divide our Seq/Folder/DataFrame.
        
        """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def chunk_interpolate(df,size=10**6,interpolate=True, method="linear", axis=0,limit_direction="both", limit=1):

    """
    After Chunker makes the pieces according to index, we Interpolate them with *args* of pandas.interpolate() and then we Merge them back together.
    This step is crucial for the complete data interpolation without RAM problems especially in large DataSets.
    
        :param df: Date/Time DataFrame or any Given DataFrame.
        :param size: The size/chunks we want to divide our /DataFrame according to the global index of the set. The Default price is 10 million.
        
        """
    
    group=[]
    for g in chunker(df,size):
        group.append(g)
    print('Groupping Complete')
    for i in range(len(group)):
            group[i].interpolate(method=method,axis=axis,limit_direction = limit_direction, limit = limit, inplace=True)
            df_int=pd.concat(group[i] for i in range(len(group)))
            df_int=pd.concat(group[i] for i in range(len(group)))
    print('Chunk Interpolate Done')
    return df_int


def is_stable(*args, epsilon):
   """
    Returns a boolean vector from the division of variance with mean of a column.
    
        :param epsilon: A small value in order to avoid dividing with Zero.
      
        """
    #implemented using the index of dispersion (or Fano factor)
    dis = np.var(np.array(args),axis = 1)/np.mean(np.array(args),axis = 1)
    return np.all(np.logical_or((dis < epsilon),np.isnan(dis)))


def filter_dispersed(df, window, eps):
   """
    We are looking at windows of consecutive row and calculate the mean and variance. For each window if the index of disperse or given column is in the given threshhold
    then the last row will remain in the data frame.
    
        :param df: Date/Time DataFrame or any Given DataFrame.
        :param window: A small value in order to avoid dividing with Zero.
        :param eps: A small value in order to avoid dividing with Zero (See is_stable)
      
        """
   df_tmp = df[rolling_apply(is_stable, window, *df.transpose().values, epsilon= eps)]
   return df_tmp[window:]
