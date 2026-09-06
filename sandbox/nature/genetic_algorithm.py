""" Simple Genetic Algorithm

NOTE: Current implementation defined only on the Iris dataset,
as a control/comparison to the deep model.


# The genetic algorithm is defined as follows:
#---------------------------------------------

#==== Genome
genome : ndarray.float32, (C, D)
    an array used to make predictions (affine trans.)
    on the iris class, from the observed input features
    where:
    C = |iris classes|  = 3
    D = |iris features| = 4

#==== Fitness function
fitness : (features, labels, genome) --> int
    fitness of a genome is defined as the number of
    correct predictions of class from a set of input
    features

#==== Selection routine
selection : (samples, population) ---> fittest genome
    tournament-style selection routine where
    TOURNAMENT_SIZE number of genomes are randomly
    selected and the fittest genome (singular) is returned

#==== Mutation
mutate : (genome,) ---> genome | mutated_genome
    with probability MUTATION_PROB, a genome has
    one of it's 3 weight vectors re-initialized

#==== Reproduction
reproduce : (genome1, genome2) ---> (genomeA, genomeB)
    reproduction in this GA is defined as a random,
    multi-point crossover between the weights of
    two different genomes, followed by mutation on the
    offspring of the two. Nothing fancy, just the
    selection of weights with masking algebra

# Population evolution
#---------------------
The process of evolution for this GA is as follows:

- Population of genomes is initialized

- Population is "evolved" through a fixed number of epochs,
  or generations following the standard GA process:

  * Selection(Population)  # Random selection of some percentage pop
  * Fitness(select)        # Evaluate fitness of selected genomes
  * Reproduction(fittest)  # Fittest genomes mate, have offspring
  * Mutation(offspring)    # Offspring have random chance of mutation
  * Population   <---------- Offspring of fittest to new population

- Using the (hopefully) fit or adapted population,
  predictions on the test dataset are made by
  taking the median label predictions across all genomes

"""
import os
import sys
import code

import numpy as np

# unfortunate relative pathing hack
#
fpath = os.path.abspath(os.path.dirname(__file__))
path_to_dataset = fpath.rstrip(fpath.split('/')[-1]) + 'data'
if not os.path.exists(path_to_dataset):
    print('ERROR: Unable to locate project data directory')
    print('Please restore the data directory to its original path at {}\n'.format(path_to_dataset),
          'or symlink it to {}\n'.format(fpath),
          'or specify the updated absolute path to the sandbox submodule scripts')
    sys.exit()

if path_to_dataset not in sys.path:
    sys.path.append(path_to_dataset)
from dataset import IrisDataset


#=============================================================================#


# Constants
#----------
POPULATION_SIZE = 128
TOURNAMENT_SIZE = 36
MUTATION_RATE   = 0.1
NUM_GENERATIONS = 200
_dtype = np.float16

#=============================================================================#
#                              GA Functions                                   #
#=============================================================================#


#-----------------------------------------------------------------------------#
#                           Initialization                                    #
#-----------------------------------------------------------------------------#

def init_genome(size=(3,4)):
    """ Genomes are:
         - (3,4) ndarray
         - sampled from Glorot random normal distribution
         - with low FP precision (to keep representation simple)
    # NOTE: Not sure this is the sensical distibution to sample from,
            nor even whether the genomes should be sampled.
            But it works, so consideration for another time.
    """
    scale = np.sqrt(2 / sum(size))
    return np.random.normal(scale=scale, size=size).astype(_dtype)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def init_population(population_size=POPULATION_SIZE):
    """ Initializes population of genomes """
    population = [init_genome() for _ in range(population_size)]
    return population

#-----------------------------------------------------------------------------#
#                             Selection                                       #
#-----------------------------------------------------------------------------#

def fitness(x, y, g):
    """ Genetic fitness function (objective function for GAs)
    Evaluates how well adapted a genome is to its env by simply
    measuring predictive accuracy
    * NB: The last step is non-differentiable, which is
          not a constraint on a GA as it would be with
          gradient-based policy

    Params
    ------
    x : ndarray.float16, (N, D)
        input features
    y : ndarray.int32, (N,), values in [0,2]
        ground truth labels (iris classes) for input x
    g : ndarray.float16, (3, D)
        'genome' represention

    Returns
    -------
    score : int
        number of correct class predictions made by genome
    """
    h     = np.matmul(x, g.T)    # (N, D).(D, 3) ---> (N, 3)
    yhat  = np.argmax(h, axis=1) # (N,)
    score = (y == yhat).sum()
    return score

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def selection(x, y, population, population_size=POPULATION_SIZE,
                               tournament_size=TOURNAMENT_SIZE):
    """ Tournament style selection routine
      A fixed number of genomes are selected from the population at random
      and the fittest genome from that selection goes on to reproduce.
    """
    #==== random selection
    idx = np.random.choice(population_size, tournament_size, replace=False)
    tournament = list(np.array(population)[idx])
    #==== evaluate fitness
    fitnesses = [fitness(x,y,g) for g in tournament]
    fittest = tournament[np.argmax(fitnesses)]
    return fittest


#-----------------------------------------------------------------------------#
#                            Crossover                                        #
#-----------------------------------------------------------------------------#

def reproduce(p1, p2):
    """ Crossover routine for two genomes

    Instead of more typical single-point transfer, a masking
    array is used to select multi-point transfer from
    parent genomes to child genomes
    """
    #==== crossover points
    gene_mask = np.random.randint(0, 2, p1.shape, dtype=np.bool)
    #==== offspring from parent genomes
    c1 = (p1 *  gene_mask) + (p2 * ~gene_mask)
    c2 = (p1 * ~gene_mask) + (p2 *  gene_mask)
    return c1.astype(_dtype), c2.astype(_dtype)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def mutate(g, mutation_rate=MUTATION_RATE):
    """ Mutation defined here as randomly resampling part of the genome"""
    if np.random.random() < MUTATION_RATE:
        mut_seq = init_genome((1,4)).squeeze()
        seq_idx = np.random.choice(3)
        g[seq_idx] = mut_seq
    return g


#=============================================================================#
#                          Genetic Algorithm                                  #
#=============================================================================#
class GeneticAlgorithm:
    """ GA function class """
    def __init__(self, dataset=None, batch_size=12,
                 num_gens=NUM_GENERATIONS,
                 population_size=POPULATION_SIZE,
                 tournament_size=TOURNAMENT_SIZE,
                 mutation_rate=MUTATION_RATE,
                 fp_precision=_dtype):
        #==== auto-attr
        for k, v in locals().items():
            if k == 'self': continue
            setattr(self, k, v)

        if self.dataset is None:
            self.dataset = IrisDataset()
        print('Initializing GA')
        self.population = init_population(population_size)

    def run(self):
        """ Run algorithm """
        #==== update status
        r_info = 'Running GA:\n  population_size: {}\n  num_generations: {}'
        status = r_info.format(self.population_size, self.num_gens)
        print(status)
        #==== var shortnames
        B = self.batch_size
        p_size = self.population_size
        t_size = self.tournament_size
        m_rate = self.mutation_rate
        #==== algo loop
        for step in range(self.num_gens):
            x, t = self.dataset.get_batch(step, B)
            next_gen = []
            for i in range(p_size // 2):
                #==== selection
                p1 = selection(x, t, self.population)
                p2 = selection(x, t, self.population)
                #==== crossover
                c1, c2 = reproduce(p1, p2)
                next_gen.extend([mutate(c1, m_rate), mutate(c2, m_rate)])
            #==== update generation
            self.population = next_gen

    def evaluate_population(self, X_test=None):
        #==== Split test-set into features and labels
        X_test = X_test if X_test is not None else np.copy(self.dataset.X_test)
        X, Y = X_test[...,:-1], X_test[...,-1]
        #print(f'Y: {Y}')

        #==== Get population
        population = self.population
        r_size = (len(population), X.shape[0])
        population_response = np.zeros(r_size, np.int32)

        #==== partial fitness func
        def _respond(x, g):
            h = np.matmul(np.copy(x), g.T) # (N,D).(D,K)
            return np.argmax(h, axis=1)    # (N,)

        #==== Iterate through population
        for idx, genome in enumerate(population):
            population_response[idx] = _respond(X, genome)

        #==== Summary population fitness
        population_fitness = np.median(population_response, axis=0)
        self.population_score = population_fitness == Y
        sum_correct = np.sum(population_fitness == Y)
        accuracy = sum_correct / len(Y)
        print('GA test accuracy: {:.4f}; {}/{} correct'.format(accuracy, sum_correct, len(Y)))
        return self.population_score


#dataset = IrisDataset()
#GA = GeneticAlgorithm(dataset)
#GA.run()
#GA.evaluate_population()


'''
#===============================================================
# Evolving a population

# Setup
#---------
dataset = IrisDataset()
B = 12 # batch-size
population = init_population()
num_gens = 50

for step in range(num_gens):
    x, t = dataset.get_batch(step, B)
    next_gen = []
    for i in range(POPULATION_SIZE // 2):
        #==== selection
        p1 = selection(x, t, population)
        p2 = selection(x, t, population)
        #==== crossover
        child1, child2 = reproduce(p1, p2)
        next_gen.extend([mutate(child1), mutate(child2)])  # this sounds bad, change var naem
    population = next_gen



#===============================================================
# Evaluating population "fitness"
#  (eg, making predictions on a test set)

def predict(x, g):
    h = np.matmul(np.copy(x), g.T) # (N, 3)
    yhat = np.argmax(h, axis=1) # (N,)
    return yhat

def predict_mat(x, genomes):
    # x.shape = (30, 4)
    # g.shape = (P, 3, 4)
    h = np.einsum('nd,dkp->nkp', np.copy(x), genomes.T) # (N,3,POPULATION_SIZE)
    yhats = np.argmax(h, axis=1) # (N,POPULATION_SIZE)
    return yhats

def query_gene_pool(x, pool):
    yhats = np.zeros((len(pool), x.shape[0])).astype(np.int32)
    for gidx, gene in enumerate(pool):
        yhats[gidx] = predict(x, gene)
    return yhats

#==== split test set
X_test = dataset.X_test
x_test, y_test = X_test[...,:-1], X_test[...,-1]

test_preds = query_gene_pool(x_test, population) #(num_test, POPULATION_SIZE)

Y_pred   = np.median(test_preds, axis=0).astype(np.int32)
accuracy = np.sum(Y_pred == y_test) / len(y_test)

print(f'Y_pred: {Y_pred}')
print(f'Accuracy: {accuracy}')
'''
