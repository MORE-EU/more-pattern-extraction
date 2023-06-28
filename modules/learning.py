import numpy as np
import pandas as pd
import modules.statistics as st
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from modules.preprocessing import enumerate2
from modules.io import *
from modules.preprocessing import *
from tqdm import tqdm
import random
import stumpy


def predict(df_test, model, feats, target):
    """
    Applies a regression model to predict values of a dependent variable for a given dataframe and 
    given features.

    Args:
        df_test: The input dataframe.
        model: The regression model. Instance of Pipeline.
        feats: List of strings: each string is the name of a column of df_test.
        target: The name of the column of df corresponding to the dependent variable.
    Returns:
        y_pred: Array of predicted values. 
    """

    df_x = df_test[feats]
    df_y = df_test[target] #is this needed?
    X = df_x.values
    y_true = df_y.values #is this needed?
    y_pred = model.predict(X)
    return y_pred

def fit_linear_model(df, feats, target, a=1e-4, deg=3):
    """
    Fits a regression model on a given dataframe, and returns the model, the predicted values and the associated 
    scores. Applies Ridge Regression with polynomial features. 

    Args:
        df: The input dataframe.
        feats: List of names of columns of df. These are the feature variables.
        target: The name of a column of df corresponding to the dependent variable.
        a: A positive float. Regularization strength parameter for the linear least squares function 
        (the loss function) where regularization is given by the l2-norm. 
        deg: The degree of the regression polynomial.

    Returns:    
        pipeline: The regression model. This is an instance of Pipeline.
        y_pred: An array with the predicted values.
        r_sq: The coefficient of determination “R squared”.
        mae: The mean absolute error.
        me: The mean error.
        mape: The mean absolute percentage error.
        mpe: The mean percentage error.
    """

    df_x = df[feats]
    df_y = df[target]
    X = df_x.values
    y = df_y.values

    polynomial_features = PolynomialFeatures(degree=deg)
    linear_regression = Ridge(alpha=a)

    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])

    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    r_sq, mae, me, mape, mpe = st.score(y, y_pred)
    return pipeline, y_pred, r_sq, mae, me, mape, mpe

def get_line_and_slope(values):
    """
    Fits a line on the 2-dimensional graph of a regular time series, defined by a sequence of real values. 

    Args:
        values: A list of real values.

    Returns: 
        line: The list of values as predicted by the linear model.
        slope: Slope of the line.
        intercept: Intercept of the line.   
    """

    ols = LinearRegression()
    X = np.arange(len(values)).reshape(-1,1)
    y = values.reshape(-1,1)
    ols.fit(X, y)
    line = ols.predict(X)
    slope = ols.coef_.item()
    intercept = ols.intercept_.item()
    return line, slope, intercept

def train_on_reference_points(df, w_train, ref_points, feats, target, random_state=0):
    """
    Trains a regression model on a training set defined by segments of a dataframe. 
    These segments are defined by a set of starting points and a parameter indicating their duration. 
    In each segment, one subset of points is randomly chosen as the training set and the remaining points 
    define the validation set.
    
    Args:
        df: Input dataframe. 
        w_train: The duration, given as a number of days, of the segments where the model is trained.
        ref_points: A list containing the starting date of each segment where the model is trained.
        feats: A list of names of columns of df corresponding to the feature variables.
        target: A name of a column of df corresponding to the dependent variable.
        random_state: Seed for a random number generator, which is used in randomly selecting the validation 
        set among the points in a fixed segment.

    Returns:
        model: The regression model. This is an instance of Pipeline.
        training_scores: An array containing scores for the training set. It contains the coefficient 
        of determination “R squared”, the mean absolute error, the mean error, the mean absolute percentage error.
        validation_scores: An array containing scores for the validation set. It contains the coefficient 
        of determination “R squared”, the mean absolute error, the mean error, the mean absolute percentage error.
    """

    df_train = pd.DataFrame([])
    df_val = pd.DataFrame([])
    for idx in range(ref_points.size):
        d_train_stop = pd.to_datetime(ref_points[idx]) + pd.Timedelta(days=w_train)
        df_tmp = df.loc[ref_points[idx]:str(d_train_stop)]
        df_tmp2 = df_tmp.sample(frac=1, random_state=random_state) # added random state for reproducibility during experiments
        size_train = int(len(df_tmp2) * 0.80)
        df_train = df_train.append(df_tmp2[:size_train])
        df_val = df_val.append(df_tmp2[size_train:])

    model, y_pred_train, r_sq_train, mae_train, me_train, mape_train, mpe_train = fit_linear_model(df_train, feats, target)
    y_pred_val = predict(df_val, model, feats, target)
    r_sq_val, mae_val, me_val, mape_val, mpe_val = st.score(df_val[target].values, y_pred_val)
    training_scores = np.array([r_sq_train, mae_train, me_train, mape_train])
    validation_scores = np.array([r_sq_val, mae_val, me_val, mape_val, mpe_val])

    print('Training Metrics:')
    print(f'MAE:{training_scores[1]:.3f} \nME(true-pred):{training_scores[2]:.3f} \nMAPE:{training_scores[3]:.3f} \nR2: {training_scores[0]:.3f}\n')
    print('Validation Metrics:')
    print(f'MAE:{validation_scores[1]:.3f} \nME(true-pred):{validation_scores[2]:.3f} \nMAPE:{validation_scores[3]:.3f} \nMPE:{validation_scores[4]:.3f} \nR2: {validation_scores[0]:.3f}\n')
    return model, training_scores, validation_scores

def predict_on_sliding_windows(df, win_size, step, model, feats, target):
    """
    Given a regression model, predicts values on a sliding window in a dataframe 
    and outputs scores, a list of predictions and a list of windows. 

    Args: 
        df: The input dataframe.
        win_size: The size of the sliding window, as a number of days.
        step: The sliding step.
        model: The regression model. 
        feats: A list of names of columns of df indicating the feature variables.
        target: The name of a column of df indicating the dependent variable.

    Returns:
        scores: An array of arrays of scores: one array for each window containing the coefficient of 
        determination “R squared”, the mean absolute error, the mean error, the mean absolute percentage error, 
        the mean percentage error.
        preds_test: a list of predictions: one list of predicted values for each window.
        windows: A list of starting/ending dates: one for each window.
    """

    windows = []
    preds_test = []
    scores_list = []
    for i, time in enumerate2(min(df.index), max(df.index), step=step):
        window = pd.to_datetime(time) + pd.Timedelta(days=win_size)
        df_test = df.loc[time:window]
        if df_test.shape[0]>0:
            y_pred = predict(df_test, model, feats, target)
            r_sq, mae, me, mape, mpe = st.score(df_test[target].values, y_pred)
            scores_list.append([r_sq, mae, me, mape, mpe])
            preds_test.append(y_pred)
            windows.append((time, window))
    scores = np.array(scores_list)
    return scores, preds_test, windows

def changepoint_scores(df, feats, target, d1, d2, w_train, w_val, w_test):
    """
    Given as input a dataframe and a reference interval where a changepoint may lie, trains a regression model in
    a window before the reference interval, validates the model in a window before the reference interval and tests 
    the model in a window after the reference interval. 

    Args:
        df: The input dataframe.
        feats: A list of names of columns of df indicating the feature variables.
        target: The name of a column of df indicating the dependent variable.
        d1: The first date in the reference interval.
        d2: The last date in the reference interval.
        w_train: The number of days defining the training set.
        w_val: The number of days defining the validation set.
        w_test: The number of days defining the test set.
    Returns:
        y_pred_train: The array of predicted values in the training set.
        score_train: An array containing scores for the training set: 
        the coefficient of determination “R squared”, the mean absolute error, the mean error, 
        the mean absolute percentage error, the mean percentage error.
        y_pred_val: The array of predicted values in the validation set.
        score_val: An array containing scores for the validation set: 
        the coefficient of determination “R squared”, the mean absolute error, the mean error, 
        the mean absolute percentage error, the mean percentage error.
        y_pred_test: The array of predicted values in the test set.
        score_test: An array containing scores for the test set: 
        the coefficient of determination “R squared”, the mean absolute error, the mean error, 
        the mean absolute percentage error, the mean percentage error.
    """

    d_train_start = pd.to_datetime(d1) - pd.Timedelta(days=w_train) - pd.Timedelta(days=w_val)
    d_train_stop = pd.to_datetime(d1) - pd.Timedelta(days=w_val)
    d_test_stop = pd.to_datetime(d2) + pd.Timedelta(days=w_test)
    df_train = df.loc[str(d_train_start):str(d_train_stop)]
    df_val = df.loc[str(d_train_stop):str(d1)]
    df_test = df.loc[str(d2):str(d_test_stop)]
    if len(df_train) > 0 and len(df_test) > 0:
        model, y_pred_train, r_sq_train, mae_train, me_train, mape_train, mpe_train = fit_linear_model(df_train, feats, target)
        y_pred_val = predict(df_val, model, feats, target)
        y_pred_test = predict(df_test, model, feats, target)
        
        r_sq_val, mae_val, me_val, mape_val, mpe_val = st.score(df_val[target].values, y_pred_val)
        r_sq_test, mae_test, me_test, mape_test, mpe_test = st.score(df_test[target].values, y_pred_test)
        score_train = np.array([-r_sq_train, mae_train, me_train, mape_train, mpe_train])
        score_val = np.array([-r_sq_val, mae_val, me_val, mape_val, mpe_val])
        score_test = np.array([-r_sq_test, mae_test, me_test, mape_test, mpe_test])
        return y_pred_train, score_train, y_pred_val, score_val, y_pred_test, score_test
    else:
        raise Exception("Either the training set is empty or the test set is empty")




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
