import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
from dtw import dtw

import stumpy
from tqdm import tqdm
import random
import matplotlib
matplotlib.rcParams.update({'font.size': 12,'figure.figsize': [20, 4],'lines.markeredgewidth': 0,
                            'lines.markersize': 2})

from modules.learning import clean_motifs,scorer

def plot_knee(mps, save_plot=False, filename='knee.png'):
    
    """ 
    Plot the minimum value of the matrix profile for each dimension. This plot is used to visually look for a 'knee' or 'elbow' that
    can be used to find the optimal number of dimensions to use.
    
    Args:
        mps: The multi-dimensional matrix profile. Each row of the array corresponds to each matrix profile for a given dimension 
                (i.e., the first row is the 1-D matrix profile and the second row is the 2-D matrix profile).
        save_plot: If save_plot is True then the figure will be saved. Otherwise it will just be shown.
        filename: Used if save_plot=True, the name of the file to be saved.
    
    Return:
        Figure
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

  
def predefined_labels(axs, df, fixed_dates):
    """
    Given a df and a list of dates we find the index of those dates in a dataset
    
    Args:
        axs: axes
        df: DateTime DataFrame
        fixed_dates: a list of dates (sorted)
    Return:
        ids: indexes of the dates 
    
    """

    ids = []
    for d in fixed_dates:
        axs.axvline(x=df.index.get_loc(d), ls="dashed", color='darkgreen')
        ids.append(df.index.get_loc(d))
    return ids


def convert_labels_to_dt(labels, df): 
    """
    
    Args:
    
    Returns:
    """
    l = []
    for label in labels:
        if(len(df) >= round(label)):
            l.append(df.index[round(label) ].strftime("%Y-%m-%d"))
        else:
            l.append(None)
            #l = [ df.index[round(label)].strftime("%Y-%m-%d") for label in labels]
    return l


def get_fixed_dates(df,fixed_dates):
    """
    For sorted data we get the array of indexes of those dates
    
    Args:
        df: DateTime dataframe
        fixed_dates: a list of dates (sorted)
         
    Return:
        array of indexes of the dates 
    
    """
    dates = []
    for date_point in fixed_dates:
        date_point = datetime.strptime(date_point, '%Y-%m-%d %H:%M:%S')
        start_date_d = df.index[0]
        end_date_d = df.index[-1]
        if (date_point >= start_date_d) and (date_point <= end_date_d):
            dates.append(df.index.get_loc(date_point))
    return np.array(dates)

  
def plot_profile(df, mp, col_name): 
    """
    Plot the 'Column_name' graph and the corresponding MatrixProfile. We denote with the black arrow to top motif in our set
    
    Args:
        df : A pandas DataFrame/Time Series
        mp : matrix profile distances
        col_name: name of a column
    
    Returns:
        list : figures
        A list of matplotlib figures.
    """
    motifs_idx = np.argsort(mp)[:2]
    mp_len = mp.shape[0]
    with sns.axes_style('ticks'):
        fig1, axs1 = plt.subplots(1, sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 6))
        df.iloc['col_name'].plot(ax=axs1, figsize=(20, 6))

        fig2, axs2 = plt.subplots(1, sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 6))


        axs2.set_ylabel(f'MP-', fontsize='20')
        axs2.plot(mp, c='orange')
        axs2.set_xlabel('Time', fontsize ='20')

        axs2.plot(motifs_idx, mp[motifs_idx] + 1, marker="v", markersize=10, color='black')
        axs2.plot(motifs_idx, mp[motifs_idx] + 1, marker="v", markersize=10, color='black')
        
        
def plot_profile_md(df,column_name,mp): 
    """
    Plot the 'Column_name' graph and the corresponding Profile. We denote with the black arrow to top motif in our set.
    
    Args:
        df : A pandas DataFrame/Time Series
        mp : matrix profile distances
        column_name: name of the column
        
    Returns:
        list : figures
        A list of matplotlib figures.
    """
    
    motifs_idx = np.argsort(mp, axis=1)[:, :2]
    with sns.axes_style('ticks'):
        fig1, axs1 = plt.subplots(1, sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 12))
        df['col_name'].plot(ax=axs1, figsize=(20, 2))

        fig2, axs2 = plt.subplots(mp.shape[0], sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 12))
        for k, dim_name in enumerate(cols):
            if np.all(np.isinf(mp[k])): continue

            axs2[k].set_ylabel(f'MP-{dim_name}', fontsize='10')
            axs2[k].plot(mp[k], c='orange')
            axs2[k].set_xlabel('Time', fontsize ='20')

        axs2[k].plot(motifs_idx[k, 0], mp[k, motifs_idx[k, 0]] + 1, marker="v", markersize=10, color='black')
        axs2[k].plot(motifs_idx[k, 1], mp[k, motifs_idx[k, 1]] + 1, marker="v", markersize=10, color='black')
        
        
def plot_segmentation(df, path, output, fixed_dates, file_name, top_seg=3):
    """
    Plotting the ChangePoints/ Regimes that we precomputed from change_points. 
    The result would be multiple graphs up to the number of L ( subsquence list).  
    Later we use the dtw in order to find a generalized distance between a list of dates and the
    precomputed change points/regimes.
    In the end we save the ones that have the minimum distance up to a number the user wishes.
    
    Args:
        df: DateTime DataFrame
        path: path to be saved the figures
        output: output from change_points
        top_seg: number of the best plots we want to save
        fixed_dates: list of sorted dates
    
    Return:
        List of figures, we are saving up to top_seg
    """

    
    best = []
    
    diff1=[]
    dloc=[]
    regi=[]
    model=[]
    lam=[]
    with sns.axes_style('ticks'): 
        figs = []
        
        for l, v in output.items():
            lam.append(l)
            dloc.append(convert_labels_to_dt(v[0][1],df))
            
            sz = len(df)
            threshold = sz / 1000
            ids = get_fixed_dates(df,fixed_dates)
            fig, axs = plt.subplots(1, sharex=True, gridspec_kw={'hspace': 0.1}, figsize=(20, 3))
            diff = 0
        
            for idx, (cac, regime_locations) in enumerate(v):
                axs.plot(np.arange(0, cac.shape[0]), cac, color='C1')
                filtered_regimes = regime_locations[(regime_locations > ids[0] - threshold) & (regime_locations < ids[- 1] + threshold)]
                if len(filtered_regimes) == 0:
                    filtered_regimes = regime_locations
                manhattan_distance = lambda x, y: np.abs(x - y)
                diff = dtw( regime_locations, ids, dist=manhattan_distance)[0]
                axs.set_ylabel(f'L:{str(l)}', fontsize=18)
                axs.set_title(f'{file_name}-{diff}')
                for regime in regime_locations:
                    
                    axs.axvline(x=regime, linestyle=":")
                plt.minorticks_on()
                diff1.append(diff)
                regi.append(regime_locations)
              
            ids = predefined_labels(axs, df,fixed_dates)
            labels = [item for item in axs.get_xticks()]
            visible = convert_labels_to_dt(labels[-1:1], df)
            locs, _ = plt.xticks()
            plt.xticks(ticks=locs[1:-1], rotation=30)
            name = f'{path}segmentation-{str(l)}-'
            config = {"L": l, "regime": regime, 'diff': diff}
            figs.append([fig, diff, name, config])
        sorted_figs = sorted(figs, key=lambda tup: tup[1])
       
        if(len(sorted_figs) < top_seg):
            top_seg = len(sorted_figs)
        for i in range(0, top_seg, 1):
            fig, diff, name, config = sorted_figs[i]
            best.append(config)
            fig.savefig(name + "_" + str(i))
    model = pd.DataFrame.from_dict({"L": lam,'Changepoints indexes': regi, "Changepoints Dates": dloc, "Normalized Distance": (diff1-np.min(diff1))/(np.max(diff1) - np.min(diff1))})
    return best, model







def motif_graph_multi_dim(col,df,df_soil_output,alles_df,n,plot=True):
    """
    Plots the best results returned by fitness function. Motifs are being plotted in a view a barplots
    df: pandas dataframe
    df_soil_output: pandas dataframe of soiling events
    col: columns of the dataframe to perform the routine
    alles_df: the dataframe returned by fitness fucntion
    n: the number of outputs to plot
    """
    import sklearn.metrics
    for p in range(alles_df.shape[0])[:n]:
        


        stumpy.config.STUMPY_EXCL_ZONE_DENOM= alles_df.iloc[p]['exclusion_zone']
        mp,mpi=stumpy.mstump(df[col], m=int(alles_df.iloc[p]['profile_wind']),normalize=True)
        md,mi,_,_=stumpy.mmotifs(df[col], mp, mpi,min_neighbors=alles_df.iloc[p]['min_nei'],
                                 max_distance=alles_df.iloc[p]['max_d'],cutoffs=alles_df.iloc[p]['cutoff'],
                                 max_matches=alles_df.iloc[p]['max_m'],max_motifs=alles_df.iloc[p]['max_motif'])  
       
        md,mi=clean_motifs(md,mi)
        m=int(alles_df.iloc[p]['profile_wind'])
        dates = pd.DataFrame(index=range(len(df)))
        dates = dates.set_index(df.index)
        dates['power']=df['power']        
        dates['soil']=df.soiling_derate
        for i,mtype in enumerate(mi):
#             print(i,mtype)
            if i==alles_df.iloc[p]['motif_type']-1:
                test=pd.DataFrame()
                test.index=df.index
                test['pred']=np.nan
                for j in mtype:
                    test['pred'].iloc[j:j+m]=1
                test['pred'] = test['pred'].fillna(0)
                test["actual"] = np.nan
                for start, end in zip(df_soil_output.SoilStart, df_soil_output.SoilStop):
                    test.loc[start:end, 'actual'] = 1
                test['actual'] = test['actual'].fillna(0)
                f1,ps,recal,hamming,jaccard,cohen,roc=scorer(test)
                if plot==True:
                
                    ax=df.soiling_derate.plot(color="green",label='Soil Derate',figsize=(20,10))
                    ax.set_xlabel('Periods defined by best motif type',fontsize=24)
                    ax.set_ylabel("Soil Derate",color="black",fontsize=24)
                    for d in df_soil_output.SoilStart:
                        ax.axvline(x=d, color='black', linestyle='-')
                    ax.axvline(x=d, color='black', linestyle='-',label='Soil Start')
                    for d in df_soil_output.SoilStop:
                        ax.axvline(x=d, color='black', linestyle=':') 
                    ax.axvline(x=d, color='black', linestyle=':',label='Soil Ends') 
                    plt.legend(fontsize=14,loc='lower left')   
                    for idx in mtype:
                        ax.set_title(f'{m}-Days length,{i+1}-Motif Type, {len(mtype)}-Neighbors, F1:{np.round(f1,3)}',
                                     fontsize=20)
                        ax.axvspan(dates.index[idx],dates.index[idx+m], color = 'orange',ymax=0.96, alpha = 0.9)
                    ax.axvspan(dates.index[idx],dates.index[idx+m], color = 'orange',ymax=0.96, alpha = 0.9,label='motifs')


                    plt.legend(fontsize=14,loc='lower left')   
                    plt.show()
                print(sklearn.metrics.classification_report(test.actual,test.pred))
                

def motif_graph_multi_dim_eval(col,df,df_soil_output,alles_df,n,plot=False):
    """
    Plots the best results returned by fitness function. Motifs are being plotted in a view a barplots
    df: pandas dataframe
    df_soil_output: pandas dataframe of soiling events
    col: columns of the dataframe to perform the routine
    alles_df: the dataframe returned by fitness fucntion
    n: the number of outputs to plot
    """
    import sklearn.metrics
    for p in range(alles_df.shape[0])[:n]:
        stumpy.config.STUMPY_EXCL_ZONE_DENOM=alles_df.iloc[p]['exclusion_zone']

        


       
        mp,mpi=stumpy.mstump(df[col], m=int(alles_df.iloc[p]['profile_wind']),normalize=True)
        md,mi,_,_=stumpy.mmotifs(df[col], mp, mpi,min_neighbors=alles_df.iloc[p]['min_nei'],
                                 max_distance=alles_df.iloc[p]['max_d'],cutoffs=alles_df.iloc[p]['cutoff'],
                                 max_matches=alles_df.iloc[p]['max_m'],max_motifs=alles_df.iloc[p]['max_motif'])  
       
        md,mi=clean_motifs(md,mi)
        m=int(alles_df.iloc[p]['profile_wind'])
        dates = pd.DataFrame(index=range(len(df)))
        dates = dates.set_index(df.index)
        dates['power']=df['power']        
        dates['soil']=df.soiling_derate
        for i,mtype in enumerate(mi):
#             print(i,mtype)
            if i==alles_df.iloc[p]['motif_type']-1:
                test=pd.DataFrame()
                test.index=df.index
                test['pred']=np.nan
                for j in mtype:
                    test['pred'].iloc[j:j+m]=1
                test['pred'] = test['pred'].fillna(0)
                test["actual"] = np.nan
                for start, end in zip(df_soil_output.SoilStart, df_soil_output.SoilStop):
                    test.loc[start:end, 'actual'] = 1
                test['actual'] = test['actual'].fillna(0)
                f1,ps,recal,hamming,jaccard,cohen,roc=scorer(test)
                if plot==True:
                    ax=df.soiling_derate.plot(color="green",label='Soil Derate',figsize=(20,10))
                    ax.set_xlabel('Periods defined by best motif type',fontsize=24)
                    ax.set_ylabel("Soil Derate",color="black",fontsize=24)
                    for d in df_soil_output.SoilStart:
                        ax.axvline(x=d, color='black', linestyle='-')
                    ax.axvline(x=d, color='black', linestyle='-',label='Soil Start')
                    for di in df_soil_output.SoilStop:
                        ax.axvline(x=di, color='black', linestyle=':') 
                    ax.axvline(x=di, color='black', linestyle=':',label='Soil Ends') 
                    plt.legend(fontsize=14,loc='lower left')   
                    for idx in mtype:
                        ax.set_title(f'{m}-Days length, {i+1}-Motif Type, {len(mtype)}-Neighbors, F1:{np.round(f1,3)}',
                                     fontsize=20)
                        ax.axvspan(dates.index[idx],dates.index[idx+m], color = 'orange',ymax=0.96, alpha = 0.9)
                    ax.axvspan(dates.index[idx],dates.index[idx+m], color = 'orange',ymax=0.96, alpha = 0.9,label='motifs')


                    plt.legend(fontsize=14,loc='lower left')

                    plt.show()
                print(sklearn.metrics.classification_report(test.actual,test.pred))







def evaluate_motifs(col,df,df_soil_output,alles_df):
    import sklearn.metrics
    eval_df=pd.DataFrame()
    eval_df.index=range(alles_df.shape[0])
    score_list=[]
    score_list_jac=[]
    score_list_recall=[]
    score_list_hamming=[]
    score_list_pres=[]

    type_list=[]
    for p in range(alles_df.shape[0]):
        stumpy.config.STUMPY_EXCL_ZONE_DENOM=alles_df.iloc[p]['exclusion_zone']
#         print(alles_df.iloc[p])
        mp,mpi=stumpy.mstump(df[col], m=int(alles_df.iloc[p]['profile_wind']),normalize=True)
        md,mi,_,_=stumpy.mmotifs(df[col], mp, mpi,min_neighbors=alles_df.iloc[p]['min_nei'],
                                 max_distance=alles_df.iloc[p]['max_d'],cutoffs=alles_df.iloc[p]['cutoff'],
                                 max_matches=alles_df.iloc[p]['max_m'],max_motifs=alles_df.iloc[p]['max_motif'])  
        md,mi=clean_motifs(md,mi)
        m=int(alles_df.iloc[p]['profile_wind'])
        
        maxi=0
        lista=[]
        listam=[]
        listjac=[]
        listham=[]
        listpre=[]
        listrec=[]
        for i,mtype in enumerate(mi):
            test=pd.DataFrame()
            test.index=df.index
            test['actual']=np.nan
            for start, end in zip(df_soil_output.SoilStart, df_soil_output.SoilStop):
                test.loc[start:end, 'actual'] = 1
            test['actual'] = test['actual'].fillna(0)
            test['pred']=np.nan
            for j in mtype:
                test['pred'].iloc[j:j+m]=1
            test['pred'] = test['pred'].fillna(0)
            f1,ps,recal,hamming,jaccard,cohen,roc=scorer(test)
            lista.append(f1)
            listjac.append(jaccard)
            listham.append(hamming)
            listpre.append(ps)
            listrec.append(recal)
            listam.append(i)
#             print(recal)
            
        score_list.append(np.max(lista))
        
        score_list_jac.append(listjac[lista.index(max(lista))])
        score_list_recall.append(listrec[lista.index(max(lista))])
        score_list_hamming.append(listham[lista.index(max(lista))])
        score_list_pres.append(listpre[lista.index(max(lista))])
        
        type_list.append(listam[np.argmax(lista)]+1)
        
    
    eval_df['f1']=score_list
    eval_df['jaccard']=score_list_jac
    eval_df['recall']=score_list_recall
    eval_df['hamming']=score_list_hamming
    eval_df['pres']=score_list_pres



    eval_df['motif_type']=type_list
    
    return eval_df,score_list,type_list
            
    
def matching_eval(col,df_old,df_new,alles_df,events,df_soil_output,n,plot=True):
    
    teliko=[]
    telos=pd.DataFrame()
    for p in tqdm(range(alles_df.shape[0])[:n]):
#         print(f'Grammh apo frame{p}')
        dates = pd.DataFrame(index=range(len(df_new)))
        dates = dates.set_index(df_new.index)
        dates['power']=df_new['power']        
        dates['soil']=df_new.soiling_derate
        stumpy.config.STUMPY_EXCL_ZONE_DENOM=alles_df.iloc[p]['exclusion_zone']
        mp,mpi=stumpy.mstump(df_old[col], m=int(alles_df.iloc[p]['profile_wind']),normalize=True)
        md,mi,_,_=stumpy.mmotifs(df_old[col], mp, mpi,min_neighbors=alles_df.iloc[p]['min_nei'],
                                 max_distance=alles_df.iloc[p]['max_d'],cutoffs=alles_df.iloc[p]['cutoff'],
                                 max_matches=alles_df.iloc[p]['max_m'],max_motifs=alles_df.iloc[p]['max_motif'])  
        md,mi=clean_motifs(md,mi)
        m=int(alles_df.iloc[p]['profile_wind'])
#         print(f'Window: {m}')
#         print(f'Motif index:{mi}')
#         print(f'Motif Distance:{md}')
        oliko_tupou=[]
        df_ok=pd.DataFrame()
        for typ in range(len(mi)):
#             print(f'Type of Motif {typ}')
            Oliko_query_score=[]
            df_preall=pd.DataFrame()
            
            for d,i in enumerate(mi[typ]):
#                 print(f'The index that motif start:{i}')
                if md[typ][d]<1000.0:
#                     print('pame')
#                     print(md[typ][d])
#                 print(f'Index:{mi[typ][i]}')
                    query=df_old.power[i:i+m].values
#                     print(f'Query: {query}')
                    x=random.choice([np.inf,1,2,3,4,5,6,7,8,9,10])
                    stumpy.config.STUMPY_EXCL_ZONE_DENOM = x
                    out=stumpy.match(query, df_new.power,max_matches=events)
#                     print("OUT")
                    test=pd.DataFrame()
                    test.index=df_new.index
                    test['pred']=np.nan
                    for k in out[:,1]:
                        test['pred'].iloc[k:k+m]=1
                    test['pred'] = test['pred'].fillna(0)
                    test["actual"] = np.nan
                    for start, end in zip(df_soil_output.SoilStart, df_soil_output.SoilStop):
                        test.loc[start:end, 'actual'] = 1
                    test['actual'] = test['actual'].fillna(0)
                    f1,ps,recal,hamming,jaccard,cohen,roc=scorer(test)
                    Oliko_query_score.append(f1)
                    data={'profile_wind':m,
                          'exclusion_zone':x,
                          'motif_index': i,
                          'motif_type': typ+1,
                          'actual_nei':len(out[:,1]),
                          'actual_motif':len(md),
                          'recall':np.round(recal,5),
                          'f1':np.round(f1,5),
                          'precision':np.round(ps,5),
                          'hamming':np.round(hamming,5),
                          'jaccard':np.round(jaccard,5),
                         'cohen':np.round(cohen,5),
                         'roc':np.round(roc,5)}
            
                    df_for_each = pd.DataFrame(data,index=[0])
                    df_preall=pd.concat([df_for_each,df_preall],axis=0)

                    if plot==True:
                        ax=df_new.soiling_derate.plot(color="green",label='Soil Derate',figsize=(20,10))
                        for d in df_soil_output.SoilStart:
                            ax.axvline(x=d, color='black', linestyle='-')
    #                 ax.axvline(x=d, color='black', linestyle='-',label='Soil Start')
                        for d in df_soil_output.SoilStop:
                            ax.axvline(x=d, color='black', linestyle=':') 
    #                 ax.axvline(x=d, color='black', linestyle=':',label='Soil Ends') 
                        print(f'Neigh:{len(out[:,1])}')
                        print(f'Score:{f1}')
                        for k in out[:,1]:
                            ax.axvspan(dates.index[k],dates.index[k+m], color = 'orange',ymax=0.96, alpha = 0.2)
                        plt.show()

#                 df_typ=pd.concat([df_typ,df_for_each],axis=0)
            oliko_tupou.append(np.max(Oliko_query_score))

            df_ok=pd.concat([df_ok,df_preall],axis=0)
        teliko.append(np.max(oliko_tupou))

        telos=pd.concat([telos,df_ok],axis=0)
    telos=telos.reset_index(drop=True)
    telos=telos.loc[telos[['f1']].drop_duplicates(['f1']).index]
    telos=telos.sort_values(by=['f1'], ascending=False).reset_index(drop=True)
                    
                    
            
    return teliko,telos

    
    
def match_graph_multi_dim_eval(col,df_new,df_old,df_soil_output,alles_df,n,plot=False):
  
    import sklearn.metrics
    for p in range(alles_df.shape[0])[:n]:
        index=int(alles_df.iloc[p]['motif_index'])
        stumpy.config.STUMPY_EXCL_ZONE_DENOM=alles_df.iloc[p]['exclusion_zone']
        m=int(alles_df.iloc[p]['profile_wind'])
        dates = pd.DataFrame(index=range(len(df_new)))
        dates = dates.set_index(df_new.index)
        dates['power']=df_new['power']        
        dates['soil']=df_new.soiling_derate
        query=df_old.power[index:index+m]
        out=stumpy.match(query, df_new.power,max_matches=len(df_soil_output))
        test=pd.DataFrame()
        test.index=df_new.index
        test['pred']=np.nan
        for k in out[:,1]:
            test['pred'].iloc[k:k+m]=1
        test['pred'] = test['pred'].fillna(0)
        test["actual"] = np.nan
        for start, end in zip(df_soil_output.SoilStart, df_soil_output.SoilStop):
            test.loc[start:end, 'actual'] = 1
        test['actual'] = test['actual'].fillna(0)
        f1,ps,recal,hamming,jaccard,cohen,roc=scorer(test)
        
        if plot==True:
            ax=df_new.soiling_derate.plot(color="green",label='Soil Derate',figsize=(20,10))
            ax.set_xlabel('Periods defined by best matching ',fontsize=24)
            ax.set_ylabel("Soil Derate",color="black",fontsize=24)
            ax.set_title(f'{m}-Days Length, {len(out[:,1])}-Neighbors, F1:{np.round(f1,3)}',
                                      fontsize=20)
            for d in df_soil_output.SoilStart:
                ax.axvline(x=d, color='black', linestyle='-')
            ax.axvline(x=d, color='black', linestyle='-',label='Soil Start')
            for d in df_soil_output.SoilStop:
                ax.axvline(x=d, color='black', linestyle=':') 
            ax.axvline(x=d, color='black', linestyle=':',label='Soil Ends') 
            for k in out[:,1]:
                ax.axvspan(dates.index[k],dates.index[k+m], color = 'orange',ymax=0.96, alpha = 0.9)
            ax.axvspan(dates.index[k],dates.index[k+m], color = 'orange',ymax=0.96,label='Match' ,alpha = 0.9)

            plt.legend(fontsize=14,loc='lower left')
            plt.show()
            print(sklearn.metrics.classification_report(test.actual,test.pred))
