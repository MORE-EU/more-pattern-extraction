import sys
import os
import pickle
import argparse
paths = ['', '..', '../..']
import numpy as np
import pandas as pd
import timeit
import numba
import pathlib
import collections
from numba import cuda
import multiprocessing
import math
import glob
import itertools
from pyts.preprocessing import PowerTransformer
import re
import time
from pathlib import Path
import matrixprofile
from datetime import datetime
from stumpy import stump, fluss, gpu_stump, mstumped, mstump, subspace, stumped
import seaborn as sns
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


def change_granularity(df, granularity='30s',interpolate=False,method='linear',noise=False): 

    """
        :param df: Date/Time DataFrame. 
        :param granularity: The desired time to resample our DataFrame.
        :param interpolate: If set True, the DataFrame interpolates with a 'method'.
                            in order to fill the NaN dates.
        :param method: Param of pandas.interpolate.
        :param noise: If set True, It applies a power transform to make data Gaussian with the method 'yeo-johnson'.
        """



    df = df.resample(granularity).mean()
    if interpolate==True: #Getting rid of NaN occurances.
        if method=='linear':
            df= df.interpolate(method=method)
        else:
            df=df.interpolate(method=method,order=2)
            
    if noise==True:  #Apply a power transform sample-wise to make data more Gaussian-like.
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        df = pd.DataFrame (pt.transform(df), columns= df.columns)
    
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

