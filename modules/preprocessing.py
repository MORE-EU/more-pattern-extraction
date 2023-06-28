import pandas as pd
import numpy as np
from matrixprofile import core
from sklearn.preprocessing import MinMaxScaler

def enumerate2(start, end, step=1):
    """ 
    Args:
        start: starting point
        end: ending point    .
        step: step of the process                   
        
        
    Return: 
        The interpolated DataFrame/TimeSeries
     """
    i=0
    while start < pd.to_datetime(end):
        yield (i, start)
        start = pd.to_datetime(start) + pd.Timedelta(days=step)
        i += 1

def change_granularity(df,granularity='30s',size=10**7,chunk=True): 
    """ 
    Changing the offset of a TimeSeries. 
    We do this procedure by using chunk_interpolate. 
    We divide our TimeSeries into pieces in order to interpolate them.
        
    Args:
        df: Date/Time DataFrame. 
        size: The size/chunks we want to divide our /DataFrame according to the global index of the set. The Default price is 10 million.       .
        granularity: The offset user wants to resample the Time Series                  
        chunk: If set True, It applies the chunk_interpolation
    
    Return: 
        The interpolated DataFrame/TimeSeries
     """

    df = df.resample(granularity).mean()
    print('Resample Complete')
    if chunk==True: #Getting rid of NaN occurances.
        df=chunk_interpolate(df,size=size,interpolate=True, method="linear", axis=0,limit_direction="both", limit=1)
        print('Interpolate Complete')
    return df
  
def filter_col(df, col, less_than=None, bigger_than=None): 
    """ 
    Remove rows of the dataframe that they are under, over/both from a specific/two different input price/prices.
        
    Args:
        df: Date/Time DataFrame. 
        col: The desired column to work on our DataFrame. 
        less_than: Filtering the column dropping values below that price.
        bigger_than: Filtering the column dropping values above that price.
    
    Return: 
        The Filtrated TimeSeries/DataFrame
    """
    if(less_than is not None):
        df=df.drop(df[df.iloc[:,col] < less_than].index)
    if(bigger_than is not None):
        df=df.drop(df[df.iloc[:,col] > bigger_than].index)
    print('Filter Complete')
    return df


def filter_dates(df, start, end):
    """ 
    Remove rows of the dataframe that are not in the [start, end] interval.
    
    Args:
        df:DataFrame that has a datetime index.
        start: Date that signifies the start of the interval.
        end: Date that signifies the end of the interval.
   
   Returns:
        The Filtrared TimeSeries/DataFrame
    """
    date_range = (df.index >= start) & (df.index <= end)
    df = df[date_range]
    return df

def normalize(df):
    """ 
    This function transforms an input dataframe by rescaling values to the range [0,1]. 
    
    Args:
        df: Date/Time DataFrame or any DataFrame given with a specific column to Normalize. 
   
    Return:
        Normalized Array
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
    
    """ 
    Add uniform noise to series.
    
    Args:
        series: The time series to be added noise.
        noise_max: The upper limit of the amount of noise that can be added to a time series point
    
    Return: 
        DataFrame with noise
    """
    
    if not core.is_array_like(series):
        raise ValueError('series is not array like!')

    temp = np.copy(core.to_np_array(series))
    noise = np.random.uniform(0, noise_max, size=len(temp))
    temp = temp + noise

    return temp


def add_noise_to_series_md(df, noise_max=0.00009):
    
    """ 
    Add uniform noise to a multidimensional time series that is given as a pandas DataFrame.
    
    Args:
        df: The DataFrame that contains the multidimensional time series.
        noise_max: The upper limit of the amount of noise that can be added to a time series point.
   
    Return:
        The DataFrame with noise to all the columns
    """
    
    for col in df.columns:
        df[col] = add_noise_to_series(df[col].values, noise_max)
    return df


def filter_df(df, filter_dict):
    """ 
    Creates a filtered DataFrame with multiple columns.
        
    Args:
        df: Date/Time DataFrame or any Given DataFrame.
        filter_dict: A dictionary of columns user wants to filter
    
    Return: 
        Filtered DataFrame
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

def chunker(seq, size):
    """
    Dividing a file/DataFrame etc into pieces for better hadling of RAM. 
    
    Args:
        seq: Sequence, Folder, Date/Time DataFrame or any Given DataFrame.
        size: The size/chunks we want to divide our Seq/Folder/DataFrame.
    
    Return:
        The divided groups
        
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def chunk_interpolate(df,size=10**6,interpolate=True, method="linear", axis=0,limit_direction="both", limit=1):

    """
    After Chunker makes the pieces according to index, we Interpolate them with *args* of pandas.interpolate() and then we Merge them back together.
    This step is crucial for the complete data interpolation without RAM problems especially in large DataSets.
    
    Args:
        df: Date/Time DataFrame or any Given DataFrame.
        size: The size/chunks we want to divide our /DataFrame according to the global index of the set. The Default price is 10 million.
    
    Return:
        The Interpolated DataFrame
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
    Args:
        epsilon: A small value in order to avoid dividing with Zero.
    
    Return: 
        A boolean vector from the division of variance with mean of a column.
    """
    #implemented using the index of dispersion (or Fano factor)
    dis = np.var(np.array(args),axis = 1)/np.mean(np.array(args),axis = 1)
    return np.all(np.logical_or((dis < epsilon),np.isnan(dis)))


def filter_dispersed(df, window, eps):
    """
    We are looking at windows of consecutive row and calculate the mean and variance. For each window if the index of disperse or given column is in the given threshhold
    then the last row will remain in the data frame.
    
    Args:
        df: Date/Time DataFrame or any Given DataFrame.
        window: A small value in order to avoid dividing with Zero.
        eps: A small value in order to avoid dividing with Zero (See is_stable)
    
    Return: The Filtered DataFrame
    """
    df_tmp = df[rolling_apply(is_stable, window, *df.transpose().values, epsilon= eps)]
    return df_tmp[window:]
  
def scale_df(df):
    """ 
    Scale each column of a dataframe to the [0, 1] range performing the min max scaling
    
    Args:
        df: The DataFrame to be scaled.
    
    Return: Scaled DataFrame
    """
    min_max_scaler = MinMaxScaler()
    df[df.columns] = min_max_scaler.fit_transform(df)
    return df

def soiling_dates(df,y=0.992,plot=True):
    """
    df:pandas dataframe with soiling column
    y: the depth of soiling period we are seeking
    plot:True/False to plot the derate
    Returns:a dataframe of dates of soiling start and soiling period ends
    """
    soil = pd.concat([pd.Series({f'{df.index[0]}': 1}),df.soiling_derate])
    soil.index = pd.to_datetime(soil.index)
    df_dates = pd.DataFrame(index = soil.index)
    df_dates["soil_start"] = soil[(soil == 1) & (soil.shift(-1) < 1)] # compare current to next
    df_dates["soil_stop"] = soil[(soil == 1) & (soil.shift(1) < 1)] # compare current to prev
    dates_soil_start = pd.Series(df_dates.soil_start.index[df_dates.soil_start.notna()])
    dates_soil_stop = pd.Series(df_dates.soil_stop.index[df_dates.soil_stop.notna()])

    #Filter significant rains with more than 'x' percipitation
    ids = []
    x=y
    for idx in range(dates_soil_start.size):
        d1 = dates_soil_start[idx]
        d2 = dates_soil_stop[idx]
        if np.min(soil.loc[d1:d2]) <= x:
            ids.append(idx)
    dates_soil_start_filtered = dates_soil_start[ids]
    dates_soil_stop_filtered = dates_soil_stop[ids]

    #df forsignificant rains.
    df_soil_output = pd.DataFrame.from_dict({"SoilStart": dates_soil_start_filtered, "SoilStop": dates_soil_stop_filtered})
    df_soil_output=df_soil_output.reset_index(drop='index')
    df_soil_output.reset_index(drop='index',inplace=True)
    print(f"We found {df_soil_output.shape[0]} Soiling Events with decay less than {x} ")

    if plot:
        print('The indication of the start of a Soil is presented with Bold line')
        print('The indication of the end of a Soil is presented with Uncontinious line')
        ax=df.soiling_derate.plot(figsize=(20,10),label='Soil Derate',color='green')
        for d in df_soil_output.SoilStart:
            ax.axvline(x=d, color='grey', linestyle='-')
        for d in df_soil_output.SoilStop:
            ax.axvline(x=d, color='grey', linestyle=':') 
        ax.set_title('Power Output', fontsize=8)
        plt.legend(fontsize=8)
        plt.show()
        
    return df_soil_output




def list_of_soil_index(df,df_soil_output,days):
    """
    Creates a list with discrete indexes from soiling events
    df: pandas dataframe
    df_soil_output: pandas dataframe of soiling events
    days: integer. shift in the index of soiling events by days
    """
    temp=df.reset_index()
    list_soil_index=[]
    for i in range(len(df_soil_output)):
        list_soil_index.append(list(range(temp[temp.timestamp==df_soil_output.SoilStart[i]].index[0]-days,
                                          temp[temp.timestamp==df_soil_output.SoilStop[i]].index[0])))
    lista_me_ta_index_apo_soil=[]
    for i in range(len(list_soil_index)):
        for j in range(len(list_soil_index[i])):
            lista_me_ta_index_apo_soil.append(list_soil_index[i][j])
    return lista_me_ta_index_apo_soil
    
def list_of_all_motifs_indexes(mi,new_population,row):
    """
    Creates a list with discrete indexes from our found motifs
    mi: motif indexes
    new_population: population of individuals
    row: the index of each individual
    """
    lista_listwn=[]
    for mtyp in range(len(mi)):
        try1=[]
        for i in mi[mtyp]:
            try1.append(list(range(i,i+int(new_population[row,5]))))
        listamot=[]
        for i in range(len(try1)):
            for j in range(len(try1[i])):
                listamot.append(try1[i][j])

        lista_listwn.append(listamot)
    return lista_listwn

def list_of_soil_index_start(df,df_soil_output,days):
    """
    Creates a list with discrete indexes from soiling events
    df: pandas dataframe
    df_soil_output: pandas dataframe of soiling events
    days: integer. shift in the index of soiling events by days
    """
    temp=df.reset_index()
    list_soil_index=[]
    for i in range(len(df_soil_output)):
        list_soil_index.append(temp[temp.timestamp==df_soil_output.SoilStart[i]].index[0]-days)
    return list_soil_index
     
def list_of_all_motifs_indexes_start(mi):
    """
    Creates a list with discrete indexes from our found motifs
    mi: motif indexes
    new_population: population of individuals
    row: the index of each individual
    """
    lista_listwn=[]
    for mtyp in range(len(mi)):
        try1=[]
        for i in mi[mtyp]:
            try1.append(i)
       
        lista_listwn.append(try1)

    return lista_listwn
