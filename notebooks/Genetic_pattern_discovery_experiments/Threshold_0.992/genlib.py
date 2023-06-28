import pandas as pd
import numpy as np
from scipy.stats.mstats import theilslopes
from ast import literal_eval
#from stumpy import stump, motifs, mass, match, core
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams.update({'font.size': 12,'figure.figsize': [20, 4],'lines.markeredgewidth': 0,
#                             'lines.markersize': 2})
def summary_motifs(mot,wholedict,Ranking,show_best=True):
    
    dflist=[]
    for m in mot:
        for typem in range(0,len(wholedict[m])):
            for stats in wholedict[m][typem]:
                dflist.append([m,typem+1,Ranking[m][typem],wholedict[m][typem][0][0],wholedict[m][typem][0][1],wholedict[m][typem][0][2],wholedict[m][typem][0][11],wholedict[m][typem][0][12],wholedict[m][typem][0][13],wholedict[m][typem][0][6],wholedict[m][typem][0][7],wholedict[m][typem][0][8],wholedict[m][typem][0][9],wholedict[m][typem][0][10]])
    df_output = pd.DataFrame(dflist, columns=['Pattern Lenght in Days', 'Motif type','Motif Instances passing from Wash/Rain','Motif Neighbors','Soil Derate Before ','Soil Derate After','Slope at motif %','Slope Before Motif %','Slope ratio','Mean Power Before Motif','Gain of Power %','Mean Power of Neighbors','Mean Precititation of Neighbors','Precipitation Percentage %'])
    if show_best==False:
        return df_output
    else:
        return df_output[df_output['Slope ratio']<0]

# def motif_plot(df,mot,miclean,dfdaypower):
#     dates = pd.DataFrame(index=range(len(df)))
#     dates = dates.set_index(df.index)
#     dates['power']=df['power']
#     dates['soil']=df.soiling_derate
#     dates['preci']=df.precipitation
#     dates['irradiance']=df.poa
    
#     for m in mot:
#         fig, axes = plt.subplots(nrows=1, ncols=int(len(miclean[m])), figsize=(20,6), constrained_layout=True)
#         plt.suptitle(f'Plots for {m}-Days')
#         for i,d in enumerate(miclean[m]):
#             if len(miclean[m])==1:
#                 for mtype in range(0,len(miclean[m])):
#                     plt.xticks(labels=df.index,ticks=np.arange(len(dfdaypower)))
#                     plt.suptitle(f'{mtype+1}-Motif Type with Length of {m}-Days, with {len(miclean[m][mtype])} Neighbors ', fontsize='20')
#                     plt.plot(dates.index,dates.power,lw=1,color='pink',label='Power')
#                     plt.plot(dates.index,dates.soil,lw=1,color='blue',label='SoilDerate')
#                     plt.plot(dates.index,dates.preci,lw=1,color='green',label='Precipitation')
#                     for idx in miclean[m][mtype]:
#                         plt.plot(dates.index[idx:idx+m], dates.power[idx:idx+m], lw=2,label='Motifs', )
#                         plt.legend(['Power','SoilDerate' ,'Precipitation', 'Motifs'], loc='upper left')
#                     plt.show()      
#             else:
#                 ax = axes[i]
#                 ax.set_title(f'{i+1}-Motif Type with Length of {m}-Days, with {len(miclean[m][i])} Neighbors ', fontsize='13')
#                 ax.plot(dates.power,color='pink',label='Raw Power')
#                 ax.plot(dates.soil,color='blue',label='Soil Derate')
#                 ax.plot(dates.preci,color='green',label='Precipitation')
#                 for idx in miclean[m][i]:
#                     ax.plot(dates.power[idx:idx+m])
#                     ax.set_ylabel('Power')
#                     ax.set_xlabel('One Year')

#                 ax.legend(['Power','SoilDerate' ,'Precipitation', 'Motifs'], fontsize='12',loc='upper left')
    
#     plt.show()

def motif_plot(df,mot,miclean,dfdaypower,wholedict,Ranking,show_best=True,):
    dates = pd.DataFrame(index=range(len(df)))
    dates = dates.set_index(df.index)
    dates['power']=df['power']
    dates['soil']=df.soiling_derate
    dates['preci']=df.precipitation
    dates['irradiance']=df.poa
    if show_best== False:
        for m in mot:
            fig, axes = plt.subplots(nrows=1, ncols=int(len(miclean[m])), figsize=(20,6), constrained_layout=True)
            plt.suptitle(f'Plots for {m}-Days')
            for i,d in enumerate(miclean[m]):
                if len(miclean[m])==1:
                    for mtype in range(0,len(miclean[m])):
                        plt.xticks(labels=df.index,ticks=np.arange(len(dfdaypower)))
                        plt.suptitle(f'{mtype+1}-Motif Type with Length of {m}-Days, with {len(miclean[m][mtype])} Neighbors ', fontsize='20')
                        plt.plot(dates.index,dates.power,lw=1,color='pink',label='Power')
                        plt.plot(dates.index,dates.soil,lw=1,color='blue',label='SoilDerate')
                        plt.plot(dates.index,dates.preci,lw=1,color='green',label='Precipitation')
                        for idx in miclean[m][mtype]:
                            plt.plot(dates.index[idx:idx+m], dates.power[idx:idx+m], lw=2,label='Motifs', )
                            plt.legend(['Power','SoilDerate' ,'Precipitation', 'Motifs'], loc='upper left')
                        plt.show()      
                else:
                    ax = axes[i]
                    ax.set_title(f'{i+1}-Motif Type with Length of {m}-Days, with {len(miclean[m][i])} Neighbors ', fontsize='13')
                    ax.plot(dates.power,color='pink',label='Raw Power')
                    ax.plot(dates.soil,color='blue',label='Soil Derate')
                    ax.plot(dates.preci,color='green',label='Precipitation')
                    for idx in miclean[m][i]:
                        ax.plot(dates.power[idx:idx+m])
                        ax.set_ylabel('Power')
                        ax.set_xlabel('One Year')

                    ax.legend(['Power','SoilDerate' ,'Precipitation', 'Motifs'], fontsize='12',loc='upper left')

        plt.show()
    else:
        summ=summary_motifs(mot,wholedict,Ranking)
        summ=summ[summ['Slope ratio']<0]
        for m in summ['Pattern Lenght in Days']:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,6), constrained_layout=True)
            plt.suptitle(f'Plots for {m}-Days')

            for mtype in (summ['Motif type'][summ['Pattern Lenght in Days']==m].values-1):
                plt.xticks(labels=df.index,ticks=np.arange(len(dfdaypower)))
                plt.suptitle(f'{mtype+1}-Motif Type with Length of {m}-Days, with {len(miclean[m][mtype])} Neighbors ', fontsize='20')
                plt.plot(dates.index,dates.power,lw=1,color='pink',label='Power')
                plt.plot(dates.index,dates.soil,lw=1,color='blue',label='SoilDerate')
                plt.plot(dates.index,dates.preci,lw=1,color='green',label='Precipitation')
                for idx in miclean[m][mtype]:
                    plt.plot(dates.index[idx:idx+m], dates.power[idx:idx+m], lw=2,label='Motifs', )
                    plt.legend(['Power','SoilDerate' ,'Precipitation', 'Motifs'],fontsize=7, loc='upper left')
            plt.show()      

def get_corrected_matrix_profile(matrix_profile, annotation_vector):
    corrected_matrix_profile = matrix_profile.copy()    
    corrected_matrix_profile[:, 0] = matrix_profile[:, 0] + ((1 - annotation_vector) * np.max(matrix_profile[:, 0]))
    return corrected_matrix_profile

def motif_graph(miclean,mot,df):
    dates = pd.DataFrame(index=range(len(df)))
    dates = dates.set_index(df.index)
    dates['power']=df['power']
    dates['soil']=df.soiling_derate
    dates['preci']=df.precipitation
    dates['irradiance']=df.poa
   
    for m in mot:
        for i,mtype in enumerate(miclean[m]):
                ax=dates.power.plot(lw=1,figsize=(20,10),color='pink',label='Power')
                ax=dates.preci.plot(lw=1,color='green',figsize=(20,10),label='Precipitation')
                for idx in mtype:
                    ax.set_title(f'{i+1}-Motif Type with length of {m}-Days, with {len(mtype)} Neighbors', fontsize=20)
                    ax=dates.power[idx:idx+m].plot( figsize=(20,10), lw=2,label='Motifs', color='red')
                    plt.legend(['Power','Precipitation', 'Motifs'], loc='upper left')
                     # set x-axis label
                ax.set_xlabel("year",fontsize=14)
                # set y-axis label
                ax2=ax.twinx()
                # make a plot with different y-axis using second axis object
                ax2=df.soiling_derate.plot(color="gold",label='Soil Derate')
                ax2.set_ylabel("Soil Derate",color="black",fontsize=14)
                plt.legend()
                plt.show()
                
                
def Ranker(df,mot,miclean,mdclean,df_rains_output,df_wash_output):
    Ranker={}
    wholedict={}
    dates = pd.DataFrame(index=range(len(df)))
    dates = dates.set_index(df.index)
    dates['power']=df['power']
    dates['soil']=df.soiling_derate
    dates['preci']=df.precipitation
    dates['irradiance']=df.poa
    
    for m in mot:
        mscores=[]
        for motif in miclean[m]:
            scores={}
            for index in motif:
                d=dates.index[index]
                scores[index]=0
                for idx,row in df_rains_output.iterrows():
                    if d<row.RainStop and d>row.RainStart:
                        if index in scores.keys():
                            scores[index]+=1         
                for idx,row in df_wash_output.iterrows():
                    if d<row.WashStop and d>row.WashStart:
                        if index in scores.keys():
                             scores[index]+=1
                for idx,row in df_rains_output.iterrows():
                    if d<row.RainStop and d>row.RainStart:
                        if index+m in scores.keys():
                            scores[index]+=1       
                for idx,row in df_wash_output.iterrows():
                    if d<row.WashStop and d>row.WashStart:
                        if index+m in scores.keys():
                            scores[index]+=1      
            score = 0
            for val in scores.values():
                score = score + val
            mscores.append(score)             
        Ranker[m]=mscores

    for m in mot:
        fortype={}
        for mtype in range(0,len(miclean[m])):

            lista=[]
            metritis=0
            soilb=0
            metritisprin=0
            metritismd=0
            metritisprmd=0
            metritisstd=0
            metritisprstd=0
            stoind=0
            soila=0
            meanper=0
            meanperpr=0
            metrslope=0
            metrslopeb=0
            len(miclean[m][mtype])
            for index in miclean[m][mtype][::]:
                
                temp=df.reset_index(drop=True)
                if temp.index[0:index].size == 0:
                    slopep=np.nan  
                else:
                    slope, intercept = np.polyfit(temp.index[index:index+m],temp.power[index:index+m],1)
                    metrslope=metrslope+slope
                    slopep, intercept = np.polyfit(temp.index[0:index],temp.power[0:index],1)
                    metrslopeb= metrslopeb + slopep                
                #mean       
                metritis=metritis + df.power[index+m:index+2*m].mean()  
                metritisprin=metritisprin+ df.power[index-2*m:index-m].mean()
                stoind=stoind+df.power[index:index+m]
                meanper=meanper+df.precipitation[index+m:index+2*m].mean()
                meanperpr=meanperpr+df.precipitation[index-2*m:index-m].mean()
                soilb=soilb+df.soiling_derate[index-2*m:index-m].mean()
                soila=soila+df.soiling_derate[index+m:index+2*m].mean()
            
            
            if len(miclean[m][mtype])==0:
                break;
            metritis=metritis/len(miclean[m][mtype])
            metrslope=metrslope/len(miclean[m][mtype])*100
            metrslopep=metrslopeb/len(miclean[m][mtype])*100
            soilb=soilb/len(miclean[m][mtype])
            soila=soila/len(miclean[m][mtype])
            meanper=meanper/len(miclean[m][mtype])
            meanperpr=meanperpr/len(miclean[m][mtype])
            metritisprin=metritisprin/len(miclean[m][mtype])
            stoind=stoind/len(miclean[m][mtype])
    
            percent_diff = ((metritis - metritisprin)/metritis)*100
            prec_diff=((meanper-meanperpr)/meanper)*100
            slope_ratio=(metrslopep/metrslope)
            lista.append([len(miclean[m][mtype]),soilb,soila,np.min(mdclean[m][mtype]),np.average(mdclean[m][mtype]),np.max(mdclean[m][mtype]),
                          metritisprin,percent_diff,metritis,meanper,prec_diff,metrslope,metrslopep,slope_ratio])
            fortype[mtype]=lista
        wholedict[m]=fortype
    return Ranker ,wholedict

def clean_motifs(corrected,mot,dailypower,max_motifs=25,min_nei=1,max_di=None,cut=None,max_matc=200):
    md={}
    mi={}
    mdtest={}
    mitest={}
    for m in mot:
        md[m],mi[m]= motifs(dailypower,corrected[m][:,0], min_neighbors=min_nei, max_distance=max_di,
                            cutoff=cut, max_matches=max_matc, max_motifs=max_motifs, normalize=True)
    miclean=dict()
    for m in mot:
        outp=[]
        for j in range(0,len(mi[m])):
            outp.append(np.delete(mi[m][j], np.where(mi[m][j] == -1)))
        miclean[m]=outp

    mdclean=dict()
    for m in mot:
        outp=[]
        for j in range(0,len(md[m])):
            outp.append(md[m][j][~np.isnan(md[m][j])])
        mdclean[m]=outp
    return miclean ,mdclean

def parse_csv(filename):
    df_meta = pd.read_csv(filename, nrows=1)
    df_temp = pd.read_csv(filename, sep='\n', header=None, names=["temp"])
    col_names = df_temp.temp.iloc[2].split(',')
    col_names.extend(['I-V Curve I Values', 'I-V Curve V Values'])
    rows = []
    num_r = df_temp.iloc[3:].shape[0]
    
    # handle columns
    for i in range(num_r):
        data = df_temp.temp.loc[3+i].split(',')
        N = int(data[41])
        rows.append(data[:42])
        rows[i].extend([np.array(data[42:42+N]), np.array(data[42+N:])])
    df = pd.DataFrame(columns=col_names, data=rows)
    
    # convert to datetime
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce') 
    
    # convert to datetime.time (99:99 will be replaced with nan)
    df.iloc[:, 38] = pd.to_datetime(df.iloc[:, 38], format= '%H:%M', errors='coerce').dt.time
    df.iloc[:, 39] = pd.to_datetime(df.iloc[:, 39], format= '%H:%M', errors='coerce').dt.time
    
    # infer and convert to appropriate types for each column
    idx = [0, 38, 39, 42, 43]
    for j in range(1, df.shape[1]):
        if j in idx:
            continue
        df.iloc[:, j] = df.iloc[:, j].apply(literal_eval)
        
    # replace -9999 precipitation with nan
    idx1 = [x for x in list(range(df.shape[1])) if x not in idx]
    df.iloc[:, idx1] = df.iloc[:, idx1].replace(-9999, np.nan)
    
    return df_meta, df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12,'figure.figsize': [20, 4],'lines.markeredgewidth': 0,
                            'lines.markersize': 2})


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




#########
#Lista me ola ta soil event indexes
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
    
#Lista dummy gia ta motif indx, v , ola ta newpop..     
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
#         print(temp[temp.timestamp==df_soil_output.SoilStart[i]].index[0]-days)
        list_soil_index.append(temp[temp.timestamp==df_soil_output.SoilStart[i]].index[0]-days)
    return list_soil_index
# print(list_of_soil_index_start(df,df_soil_output,days))
#Lista dummy gia ta motif indx, v , ola ta newpop..     
def list_of_all_motifs_indexes_start(mi):
    """
    Creates a list with discrete indexes from our found motifs
    mi: motif indexes
    new_population: population of individuals
    row: the index of each individual
    """
    lista_listwn=[]
    for mtyp in range(len(mi)):
#         print(mtyp)
        try1=[]
        for i in mi[mtyp]:
#             print(i)
            try1.append(i)
#         print(try1)
        
        lista_listwn.append(try1)


#         print(listamot)
# 
#         print(lista_listwn)
    return lista_listwn


import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from math import*
from decimal import Decimal
from numpy import mean, absolute
from scipy.spatial.distance import directed_hausdorff
def initilization_of_population_mp(pop_size,events):
    
    population=[]
    for k in range(pop_size):
        
        #min_neighbors
        m1=random.randint(1,2)
       
        #max_distance 
        m2=np.round(random.uniform(0, 0.9), 1)
        
        #cutoff
        m3=np.round(random.uniform(0, 0.9), 1)
    
        #max_matches
        m4=random.randint(events,events+2)
        
        #max_motifs
        m5=random.randint(1,10)
        
        #matrix_profile_windows
        m6=random.choice(range(4,10,1))
        
        m=[int(m1),m2,m3,int(m4),int(m5),int(m6)]
        population.append(m)
#     print(population)
    return np.array(population)

def select_mating_pool(pop, fitness, num_parents):
    """
    Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    pop: population of initilization_of_population_mp
    fitness: the fitness_fucntion
    num_parents: number of parents from population
    """
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -9999999999
    return parents

def crossover(parents, offspring_size):
    
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations):
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = np.round(random.uniform(0, 0.3),2)
#             random_value=np.random.choice([-random_value,random_value])
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + np.random.choice([-random_value,random_value])
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover


def steady_state_selection(new_population,fitness, num_parents):

    """
    Selects the parents using the steady-state selection technique. Later, these parents will mate to produce the offspring.
    It accepts 2 parameters:
        -fitness: The fitness values of the solutions in the current population.
        -num_parents: The number of parents to be selected.
    It returns an array of the selected parents.
    """

    fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
    fitness_sorted.reverse()
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, new_population.shape[1]))
    
    for parent_num in range(num_parents):
        parents[parent_num, :] = new_population[fitness_sorted[parent_num], :].copy()

    return parents, fitness_sorted[:num_parents]
def rank_selection(new_population, fitness, num_parents):

    """
    Selects the parents using the rank selection technique. Later, these parents will mate to produce the offspring.
    It accepts 2 parameters:
        -fitness: The fitness values of the solutions in the current population.
        -num_parents: The number of parents to be selected.
    It returns an array of the selected parents.
    """

    fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
    fitness_sorted.reverse()
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.

    parents = np.empty((num_parents, new_population.shape[1]))

    for parent_num in range(num_parents):
        parents[parent_num, :] = new_population[fitness_sorted[parent_num], :].copy()

    return parents, fitness_sorted[:num_parents]

def random_selection(new_population, fitness, num_parents):

    """
    Selects the parents randomly. Later, these parents will mate to produce the offspring.
    It accepts 2 parameters:
        -fitness: The fitness values of the solutions in the current population.
        -num_parents: The number of parents to be selected.
    It returns an array of the selected parents.
    """


    parents = np.empty((num_parents, new_population.shape[1]))

    rand_indices = np.random.randint(low=0.0, high=len(fitness), size=num_parents)

    for parent_num in range(num_parents):
        parents[parent_num, :] = new_population[rand_indices[parent_num], :].copy()

    return parents, rand_indices

def tournament_selection(new_population, fitness, num_parents,toursize=5):

    """
    Selects the parents using the tournament selection technique. Later, these parents will mate to produce the offspring.
    It accepts 2 parameters:
        -fitness: The fitness values of the solutions in the current population.
        -num_parents: The number of parents to be selected.
    It returns an array of the selected parents.
    """


    parents = np.empty((num_parents, new_population.shape[1]))

    parents_indices = []

    for parent_num in range(num_parents):
        rand_indices = np.random.randint(low=0.0, high=len(fitness), size=toursize)
        K_fitnesses=[]
        for rand in rand_indices:
            K_fitnesses.append(fitness[rand])
        selected_parent_idx = np.where(K_fitnesses == np.max(K_fitnesses))[0][0]
        parents_indices.append(rand_indices[selected_parent_idx])
        parents[parent_num, :] = new_population[rand_indices[selected_parent_idx], :].copy()

    return parents, parents_indices

def roulette_wheel_selection(new_population, fitness, num_parents):

    """
    Selects the parents using the roulette wheel selection technique. Later, these parents will mate to produce the offspring.
    It accepts 2 parameters:
        -fitness: The fitness values of the solutions in the current population.
        -num_parents: The number of parents to be selected.
    It returns an array of the selected parents.
    """

    fitness_sum = np.sum(fitness)
    if fitness_sum == 0:
        raise ZeroDivisionError("Cannot proceed because the sum of fitness values is zero. Cannot divide by zero.")
    probs = fitness / fitness_sum
    probs_start = np.zeros(probs.shape, dtype=np.float) # An array holding the start values of the ranges of probabilities.
    probs_end = np.zeros(probs.shape, dtype=np.float) # An array holding the end values of the ranges of probabilities.

    curr = 0.0

    # Calculating the probabilities of the solutions to form a roulette wheel.
    for _ in range(probs.shape[0]):
        min_probs_idx = np.where(probs == np.min(probs))[0][0]
        probs_start[min_probs_idx] = curr
        curr = curr + probs[min_probs_idx]
        probs_end[min_probs_idx] = curr
        probs[min_probs_idx] = 99999999999

    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, new_population.shape[1]))

    parents_indices = []

    for parent_num in range(num_parents):
        rand_prob = np.random.rand()
        for idx in range(probs.shape[0]):
            if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                parents[parent_num, :] = new_population[idx, :].copy()
                parents_indices.append(idx)
                break
    return parents, parents_indices

def stochastic_universal_selection(new_population, fitness, num_parents_mating):

    """
    Selects the parents using the stochastic universal selection technique. Later, these parents will mate to produce the offspring.
    It accepts 2 parameters:
        -fitness: The fitness values of the solutions in the current population.
        -num_parents: The number of parents to be selected.
    It returns an array of the selected parents.
    """

    fitness_sum = np.sum(fitness)
    if fitness_sum == 0:
        raise ZeroDivisionError("Cannot proceed because the sum of fitness values is zero. Cannot divide by zero.")
    probs = fitness / fitness_sum
    probs_start = np.zeros(probs.shape, dtype=np.float) # An array holding the start values of the ranges of probabilities.
    probs_end = np.zeros(probs.shape, dtype=np.float) # An array holding the end values of the ranges of probabilities.

    curr = 0.0

    # Calculating the probabilities of the solutions to form a roulette wheel.
    for _ in range(probs.shape[0]):
        min_probs_idx = np.where(probs == np.min(probs))[0][0]
        probs_start[min_probs_idx] = curr
        curr = curr + probs[min_probs_idx]
        probs_end[min_probs_idx] = curr
        probs[min_probs_idx] = 99999999999

    pointers_distance = 1.0 /(num_parents_mating) # Distance between different pointers.
    first_pointer = np.random.uniform(low=0.0, high=pointers_distance, size=1) # Location of the first pointer.

    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, new_population.shape[1]))
    

    parents_indices = []

    for parent_num in range(num_parents):
        rand_pointer = first_pointer + parent_num*pointers_distance
        for idx in range(probs.shape[0]):
            if (rand_pointer >= probs_start[idx] and rand_pointer < probs_end[idx]):
                parents[parent_num, :] = new_population[idx, :].copy()
                parents_indices.append(idx)
                break
    return parents, parents_indices

#########CROSSOVER############3
def single_point_crossover(parents, offspring_size,crossover_probability=None):
    """
    Applies the single-point crossover. It selects a point randomly at which crossover takes place between the pairs of parents.
    It accepts 2 parameters:
        -parents: The parents to mate for producing the offspring.
        -offspring_size: The size of the offspring to produce.
    It returns an array the produced offspring.
    """

    
    offspring = np.empty(offspring_size)

    for k in range(offspring_size[0]):
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = np.random.randint(low=0, high=parents.shape[1], size=1)[0]

        if not (crossover_probability is None):
            probs = np.random.random(size=parents.shape[0])
            indices = np.where(probs <= crossover_probability)[0]

            # If no parent satisfied the probability, no crossover is applied and a parent is selected.
            if len(indices) == 0:
                offspring[k, :] = parents[k % parents.shape[0], :]
                continue
            elif len(indices) == 1:
                parent1_idx = indices[0]
                parent2_idx = parent1_idx
            else:
                indices = random.sample(set(indices), 2)
                parent1_idx = indices[0]
                parent2_idx = indices[1]
        else:
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]

        # The new offspring has its first half of its genes from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring has its second half of its genes from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

#         if (self.mutation_type is None) and (self.allow_duplicate_genes == False):
#             if self.gene_space is None:
#                 offspring[k], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[k],
#                                                                                      min_val=self.random_mutation_min_val,
#                                                                                      max_val=self.random_mutation_max_val,
#                                                                                      mutation_by_replacement=self.mutation_by_replacement,
#                                                                                      gene_type=self.gene_type,
#                                                                                      num_trials=10)
#             else:
#                 offspring[k], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[k],
#                                                                                      gene_type=self.gene_type,
#                                                                                      num_trials=10)

    return offspring
def two_points_crossover(parents, offspring_size,crossover_probability=None):

    """
    Applies the 2 points crossover. It selects the 2 points randomly at which crossover takes place between the pairs of parents.
    It accepts 2 parameters:
        -parents: The parents to mate for producing the offspring.
        -offspring_size: The size of the offspring to produce.
    It returns an array the produced offspring.
    """

    
    offspring = np.empty(offspring_size)
    
    for k in range(offspring_size[0]):
        if (parents.shape[1] == 1): # If the chromosome has only a single gene. In this case, this gene is copied from the second parent.
            crossover_point1 = 0
        else:
            crossover_point1 = np.random.randint(low=0, high=np.ceil(parents.shape[1]/2 + 1), size=1)[0]

        crossover_point2 = crossover_point1 + int(parents.shape[1]/2) # The second point must always be greater than the first point.

        if not (crossover_probability is None):
            probs = np.random.random(size=parents.shape[0])
            indices = np.where(probs <= crossover_probability)[0]

            # If no parent satisfied the probability, no crossover is applied and a parent is selected.
            if len(indices) == 0:
                offspring[k, :] = parents[k % parents.shape[0], :]
                continue
            elif len(indices) == 1:
                parent1_idx = indices[0]
                parent2_idx = parent1_idx
            else:
                indices = random.sample(set(indices), 2)
                parent1_idx = indices[0]
                parent2_idx = indices[1]
        else:
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]

        # The genes from the beginning of the chromosome up to the first point are copied from the first parent.
        offspring[k, 0:crossover_point1] = parents[parent1_idx, 0:crossover_point1]
        # The genes from the second point up to the end of the chromosome are copied from the first parent.
        offspring[k, crossover_point2:] = parents[parent1_idx, crossover_point2:]
        # The genes between the 2 points are copied from the second parent.
        offspring[k, crossover_point1:crossover_point2] = parents[parent2_idx, crossover_point1:crossover_point2]

#         if (self.mutation_type is None) and (self.allow_duplicate_genes == False):
#             if self.gene_space is None:
#                 offspring[k], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[k],
#                                                                                      min_val=self.random_mutation_min_val,
#                                                                                      max_val=self.random_mutation_max_val,
#                                                                                      mutation_by_replacement=self.mutation_by_replacement,
#                                                                                      gene_type=self.gene_type,
#                                                                                      num_trials=10)
#             else:
#                 offspring[k], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[k],
#                                                                                      gene_type=self.gene_type,
#                                                                                      num_trials=10)
    return offspring

def uniform_crossover(parents, offspring_size,crossover_probability=None):

    """
    Applies the uniform crossover. For each gene, a parent out of the 2 mating parents is selected randomly and the gene is copied from it.
    It accepts 2 parameters:
        -parents: The parents to mate for producing the offspring.
        -offspring_size: The size of the offspring to produce.
    It returns an array the produced offspring.
    """

    
    offspring = np.empty(offspring_size)
    

    for k in range(offspring_size[0]):
        if not (crossover_probability is None):
            probs = np.random.random(size=parents.shape[0])
            indices = np.where(probs <= crossover_probability)[0]

            # If no parent satisfied the probability, no crossover is applied and a parent is selected.
            if len(indices) == 0:
                offspring[k, :] = parents[k % parents.shape[0], :]
                continue
            elif len(indices) == 1:
                parent1_idx = indices[0]
                parent2_idx = parent1_idx
            else:
                indices = random.sample(set(indices), 2)
                parent1_idx = indices[0]
                parent2_idx = indices[1]
        else:
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]

        genes_source = np.random.randint(low=0, high=2, size=offspring_size[1])
        for gene_idx in range(offspring_size[1]):
            if (genes_source[gene_idx] == 0):
                # The gene will be copied from the first parent if the current gene index is 0.
                offspring[k, gene_idx] = parents[parent1_idx, gene_idx]
            elif (genes_source[gene_idx] == 1):
                # The gene will be copied from the second parent if the current gene index is 1.
                offspring[k, gene_idx] = parents[parent2_idx, gene_idx]

#         if (self.mutation_type is None) and (self.allow_duplicate_genes == False):
#             if self.gene_space is None:
#                 offspring[k], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[k],
#                                                                                      min_val=self.random_mutation_min_val,
#                                                                                      max_val=self.random_mutation_max_val,
#                                                                                      mutation_by_replacement=self.mutation_by_replacement,
#                                                                                      gene_type=self.gene_type,
#                                                                                      num_trials=10)
#             else:
#                 offspring[k], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[k],
#                                                                                      gene_type=self.gene_type,
#                                                                                      num_trials=10)

    return offspring
def scattered_crossover(parents, offspring_size,num_genes,crossover_probability=None):

    """
    Applies the scattered crossover. It randomly selects the gene from one of the 2 parents. 
    It accepts 2 parameters:
        -parents: The parents to mate for producing the offspring.
        -offspring_size: The size of the offspring to produce.
    It returns an array the produced offspring.
    """

   
    offspring = np.empty(offspring_size)
    
    for k in range(offspring_size[0]):
        if not (crossover_probability is None):
            probs = np.random.random(size=parents.shape[0])
            indices = np.where(probs <= crossover_probability)[0]

            # If no parent satisfied the probability, no crossover is applied and a parent is selected.
            if len(indices) == 0:
                offspring[k, :] = parents[k % parents.shape[0], :]
                continue
            elif len(indices) == 1:
                parent1_idx = indices[0]
                parent2_idx = parent1_idx
            else:
                indices = random.sample(set(indices), 2)
                parent1_idx = indices[0]
                parent2_idx = indices[1]
        else:
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]

        # A 0/1 vector where 0 means the gene is taken from the first parent and 1 means the gene is taken from the second parent.
        gene_sources = np.random.randint(0, 2, size=num_genes)
        offspring[k, :] = np.where(gene_sources == 0, parents[parent1_idx, :], parents[parent2_idx, :])

#         if (self.mutation_type is None) and (self.allow_duplicate_genes == False):
#             if self.gene_space is None:
#                 offspring[k], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[k],
#                                                                                      min_val=self.random_mutation_min_val,
#                                                                                      max_val=self.random_mutation_max_val,
#                                                                                      mutation_by_replacement=self.mutation_by_replacement,
#                                                                                      gene_type=self.gene_type,
#                                                                                      num_trials=10)
#             else:
#                 offspring[k], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[k],
#                                                                                      gene_type=self.gene_type,
#                                                                                      num_trials=10)
    return offspring

import time
from tqdm import tqdm
import pandas as pd
import random
# from genmod.popu import *
# import stumpy


def procedure(df,df_soil_output,pop_size,days,
              num_generations,num_parents_mating,num_mutations,
              col,events,parenting,crossover,mix_up=True):
    """
    The whole genetic algortihm procedure. Returns best outputs of fitness,
    the last survived population,end_df: the frame created by all iterations,
    alles_df: the last df with the best results
    df: pandas dataframe
    df_soil_output: pandas dataframe of soiling events
    col: columns of the dataframe to perform the routine
    days: integer. shift in the index of soiling events by days
    popsize: integer. Creates individuals
    events: integer. Soiling periods
    num_generations: number of generations
    num_parents_mating: nubmer of parents to mate
    num_mutations: how many chromosomes will mutate after crossover    
    """
  #    print(f'pop_size:{pop_size}')
#     print(f'num_gen:{num_generations}')

    #Creating the initial population.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import sklearn.metrics
    start = time.time()

    new_population=initilization_of_population_mp(pop_size,events)
    # print(new_population)
    # print(f'new_population:{new_population}')
    # Number of the weights we are looking to optimize.
    num_weights = len(new_population[0,:])
    print(f'Features: {col}')
    print(f'Chromosomes: {len(new_population[0,:])}')
    print(f'Soiling Events: {events}')
    print(f'Generations: {num_generations}')
    print(f'Population :{len(new_population)}')
    print(f'Parents: {num_parents_mating}')
    
#     print(f'num_weights: {num_weights}')
    best_outputs = []
    end_df=pd.DataFrame()
    for generation in tqdm(range(num_generations)):
        # Measuring the fitness of each chromosome in the population.
        fitness,alles_df = fiteness_fun(df,df_soil_output,days,new_population,col)
        result = [] 
        for i in fitness: 
            if i not in result: 
                result.append(i)
            else: 
                result.append(0)
        fitness=result
#         print(generation,np.max(fitness))
#         print(alles_df.head(1))
        # Thei best result in the current iteration.  
#         print(np.max(fitness))
        if mix_up:
            parenting=random.choice(['sss','ranks','randoms','tournament','rws'])
#         print(parenting)
        best_outputs.append(np.max(fitness))
        # Selecting the best parents in the population for mating.
        if parenting=='smp':
            parents = select_mating_pool(new_population, fitness, 
                                          num_parents_mating)
        elif parenting=='sss':
            parents = steady_state_selection(new_population,fitness, num_parents_mating)[0]
        elif parenting=='ranks':
            parents = rank_selection(new_population,fitness, num_parents_mating)[0]
        elif parenting=='randoms':
            parents = random_selection(new_population,fitness, num_parents_mating)[0]
        elif parenting=='tournament':
            parents = tournament_selection(new_population,fitness, num_parents_mating,toursize=100)[0]
        elif parenting=='rws':
            parents = roulette_wheel_selection(new_population,fitness, num_parents_mating)[0]
        elif parenting=='sus':
            parents = stochastic_universal_selection(new_population,fitness, num_parents_mating)[0]
        else:
            raise TypeError('Undefined parent selection type')
        # Generating next generation using crossover.
        offspring_size=(len(new_population)-len(parents), num_weights)
        if mix_up:
            crossover=random.choice(['single','twopoint','uni','scatter'])

#         print(crossover)




        if crossover=='single':
            offspring_crossover=single_point_crossover(parents, offspring_size,crossover_probability=None)
        elif crossover=='twopoint':
            offspring_crossover=two_points_crossover(parents, offspring_size,crossover_probability=None)
        elif crossover=='uni':
            offspring_crossover=uniform_crossover(parents, offspring_size,crossover_probability=None)
        elif crossover=='scatter':
            offspring_crossover=scattered_crossover(parents, offspring_size,crossover_probability=None,num_genes=num_weights)
        elif crossover=='old':
            offspring_crossover = crossover(parents,
                                           offspring_size=(len(new_population)-len(parents), num_weights))
        else:
            raise TypeError('Undefined crossover selection type')

        
        # Adding some variations to the offspring using mutation.
        offspring_mutation = mutation(offspring_crossover, num_mutations)
        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
        end_df=pd.concat([alles_df,end_df],axis=0)

    end = time.time()
    print(f'Time to complete: {np.round(end - start,2)}seconds')
    return new_population,best_outputs,end_df,alles_df




def fiteness_fun(df,df_soil_output,days,new_population,col):
    fitness_each_itter=[]
    list_score_jac=[]
    alles_df=pd.DataFrame()
    
    for row in (range(len(new_population))):

        if new_population[:,5][row]<3:
            new_population[:,5][row]=3
        if new_population[:,4][row]<1:
            new_population[:,4][row]=1
            
       
            
        conc=pd.DataFrame()
        md,mi,excluzion_zone=pmc(df=df,new_population=new_population,row=row,col=col)
        lista_jac_mtype=[]
        for n,k in enumerate(range(len(mi))):
            test=tester(df,new_population,row,df_soil_output,mi,k)
            f1,ps,recal,hamming,jaccard,cohen,roc=scorer(test)
            lista_jac_mtype.append(f1)
            data={'min_nei':int(new_population[row,0]),
                 'max_d':new_population[row,1],
                 'cutoff':new_population[row,2],
                 'max_m':int(new_population[row,3]),
                 'max_motif':int(new_population[row,4]),
                 'profile_wind':int(new_population[row,5]),
                 'exclusion_zone':excluzion_zone,
                 'motif_type': n+1,
                 'actual_nei':len(mi[k]),
                 'actual_motif':len(md),
                 'recall':np.round(recal,5),
                 'f1':np.round(f1,5),
                  'precision':np.round(ps,5),
                  'hamming':np.round(hamming,5),
                  'jaccard':np.round(jaccard,5),
                 'cohen':np.round(cohen,5),
                 'roc':np.round(roc,5)}
            
             
            conc=pd.concat([conc,pd.DataFrame(data,index=[0])],axis=0)
        alles_df=pd.concat([conc,alles_df],axis=0)
        list_score_jac.append(np.max(lista_jac_mtype))
    alles_df=alles_df.loc[alles_df[['f1']].drop_duplicates(['f1']).index]
    alles_df=alles_df.sort_values(by=['f1'], ascending=False).reset_index(drop=True)
    return list_score_jac,alles_df

    


def clean_motifs(md,mi):
    """
    Cleaning found motifs from trivial motifs or dirty neighbors
    md: motif distance
    mi: motif indexes
    """
    outp=[]
    for j in range(0,len(mi)):
        outp.append(np.delete(mi[j], np.where(mi[j] == -1)))
    mi=outp    
    outp=[]
    for j in range(0,len(md)):
        outp.append(md[j][~np.isnan(md[j])])
    md=outp
    return md,mi

def pmc(df,new_population,row,col):


    """
    pmc:profile,motif,cleaning
    creates a pipeline calculation the profile the motifs and clean them for each individual
    df:df: pandas dataframe
    new_population: population of individuals
    row: the index of each individual
    col: columns of the dataframe to perform the routine
    """

    from stumpy import mstump
    from stumpy import mmotifs
    x=random.choice([np.inf,1,2,3,4,5,6,7,8])
    stumpy.config.STUMPY_EXCL_ZONE_DENOM = x

    mp,mpi=stumpy.mstump(df[col].to_numpy().transpose(), m=int(new_population[row][5]),discords=False,normalize=True)

    md,mi,sub,mdl=stumpy.mmotifs(df[col].to_numpy().transpose(),mp,mpi,
                             min_neighbors=int(new_population[row][0]),
                             max_distance=new_population[row][1],cutoffs=new_population[row][2],
                             max_matches=int(new_population[row][3]),max_motifs=int(new_population[row][4]))  
#     print(stumpy.config.STUMPY_EXCL_ZONE_DENOM)
    md,mi=clean_motifs(md,mi)
    return md,mi,stumpy.config.STUMPY_EXCL_ZONE_DENOM


def tester(df,new_population,row,df_soil_output,mi,k):
    test=pd.DataFrame()
    test.index=df.index
    test['pred']=np.nan
    for i in mi[k]:
        test['pred'].iloc[i:i+int(new_population[row,5])]=1
    test['pred'] = test['pred'].fillna(0)
    test["actual"] = np.nan
    for start, end in zip(df_soil_output.SoilStart, df_soil_output.SoilStop):
        test.loc[start:end, 'actual'] = 1
    test['actual'] = test['actual'].fillna(0)
    return test

def scorer(test):  
    import sklearn.metrics
    f1=sklearn.metrics.f1_score(test.actual, test.pred)
    ps=sklearn.metrics.precision_score(test.actual, test.pred,zero_division=0)
    recal=sklearn.metrics.recall_score(test.actual, test.pred)
    hamming=sklearn.metrics.hamming_loss(test.actual, test.pred)
    jaccard=sklearn.metrics.jaccard_score(test.actual, test.pred,zero_division=0)
    cohen=sklearn.metrics.cohen_kappa_score(test.actual, test.pred)
    roc=sklearn.metrics.roc_auc_score(test.actual, test.pred)
    return f1,ps,recal,hamming,jaccard,cohen,roc


import stumpy
from tqdm import tqdm
import random
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12,'figure.figsize': [20, 4],'lines.markeredgewidth': 0,
                            'lines.markersize': 2})





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
            
    
###KANE ONTWS PLOT
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