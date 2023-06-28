import pandas as pd
import numpy as np
import h5py
import os
import pathlib
import warnings
import gc
import matplotlib as plt
import random
from sklearn.preprocessing import MinMaxScaler
from math import*
from decimal import Decimal
from numpy import mean, absolute
from scipy.spatial.distance import directed_hausdorff


def load_df(path): 
    """ 
    Loading a parquet file to a pandas DataFrame. Return this pandas DataFrame.

    Args:
        path: Path of the under loading DataFrame.
    Return: 
        pandas DataFrame.
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
  
  
def load_mp(path):
    """
    Load the Univariate/Multivariate Matrix profile which was saved from Create_mp in a .npz file.

    Args:
      path: Path of the directory where the file is saved.

    Return:
        mp: Matrixprofile Distances
        mpi: Matrixprofile Indices
    """
    mp={}
    mpi={}
    loaded = np.load(path + ".npz", allow_pickle=True)
    mp = loaded['mp']
    mpi = loaded['mpi']
    return mp, mpi

  
def save_mdmp_as_h5(dir_path, name, mps, idx, k=0):

    """
    Save a multidimensional matrix profile as a pair of hdf5 files. Input is based on the output of (https://stumpy.readthedocs.io/en/latest/api.html#mstump).

    Args:
       dir_path: Path of the directory where the file will be saved.
       name: Name that will be appended to the file after a default prefix. (i.e. mp_multivariate_<name>.h5)
       mps: The multi-dimensional matrix profile. Each row of the array corresponds to each matrix profile for a given dimension 
                   (i.e., the first row is the 1-D matrix profile and the second row is the 2-D matrix profile).
       idx: The multi-dimensional matrix profile index where each row of the array corresponds to each matrix profile index for a given dimension.
       k: If mps and idx are one-dimensional k can be used to specify the given dimension of the matrix profile. The default value specifies the 1-D matrix profile.
                 If mps and idx are multi-dimensional, k is ignored.

    Return:

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
   
    """
    Load a multidimensional matrix profile that has been saved as a pair of hdf5 files.
    
    Args:
      dir_path: Path of the directory where the file is located.
     name: Name that follows the default prefix. (i.e. mp_multivariate_<name>.h5)
      k: Specifies which K-dimensional matrix profile to load. 
                 (i.e. k=2 loads the 2-D matrix profile
    
    Return:
        mp: matrixprofile/stumpy distances
        index: matrixprofile/stumpy indexes
            
          
        
    """
    # Load MP from disk
    
    h5f = h5py.File(dir_path + 'mp_multivariate_' + name + '.h5','r')
    mp= h5f[f'mp{k}'][:]
    h5f.close()

    h5f = h5py.File(dir_path + 'index_multivariate_' + name + '.h5','r')
    index = h5f[f'idx{k}'][:]
    h5f.close()
    return mp, index
  
  
def save_results(results_dir, sub_dir_name, p, df_stats, m, radius, ez, k, max_neighbors):
    """ 
    Save the results of a specific run in the directory specified by the results_dir and sub_dir_name.
    The results contain some figures that are created with an adaptation of the matrix profile foundation visualize() function.
    The adaptation works for multi dimensional timeseries and can be found at 
    (https://github.com/MORE-EU/matrixprofile/blob/master/matrixprofile/visualize.py) as visualize_md()

    Args:
        results: Path of the directory where the results will be saved.
        sub_directory: Path of the sub directory where the results will be saved.
        p: A profile object as it is defined in the matrixprofile foundation python library.
        df_stats: DataFrame with the desired statistics that need to be saved.
        m: The subsequence window size.
        ez: The exclusion zone to use.
        radius: The radius to use.
        k: The number of the top motifs that were calculated.
        max_neighbors: The maximum amount of neighbors to find for each of the top k motifs.

    Return:
        None

    """




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

    df_stats.to_csv(path + '/stats.csv')

    lines = [f'Window size (m): {m}',
             f'Radius: {radius} (radius * min_dist)',
             f'Exclusion zone: {ez} * window_size',
             f'Top k motifs: {k}',
             f'Max neighbors: {max_neighbors}']

    with open(path+'/info.txt', 'w') as f:
        for ln in lines:
            f.write(ln + '\n')


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

#########CROSSOVER############
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


    return offspring        
