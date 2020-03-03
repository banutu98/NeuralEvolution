#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import gzip
import pickle
import random
import time
from sklearn.metrics import log_loss
from scipy.special import expit


# In[53]:


NR_EPOCHS = 20
POP_SIZE = 30
HIGHER_BOUND = 10
LOWER_BOUND = -10
# The final result is 18 bits. Couldn't we use 32 bit unsigned
# integers instead? It could be faster, but not sure...
INTERVALS_NR = (HIGHER_BOUND - LOWER_BOUND) * 10 ** 4
BITS_NR = math.ceil(np.log2(INTERVALS_NR))
MUTATION_PROB = 0.3
CROSSOVER_PROB = 0.6
BATCH_SIZE = 256
BIT_TABLE = np.array([0x1, 0x2, 0x4, 0x8,
                      0x10, 0x20, 0x40, 0x80,
                      0x100, 0x200, 0x400, 0x800,
                      0x1000, 0x2000, 0x4000, 0x8000,
                      0x10000, 0x20000, 0x40000, 0x80000,
                      0x100000, 0x200000, 0x400000, 0x800000,
                      0x1000000, 0x2000000, 0x4000000, 0x8000000,
                      0x10000000, 0x20000000, 0x40000000, 0x80000000])


# In[169]:


def convert(y):
    decimal = int(y, 2)
    rez_final = LOWER_BOUND + decimal * (HIGHER_BOUND - LOWER_BOUND) / ((2 ** BITS_NR) - 1)
    return rez_final


# In[170]:


def sigmoid(z):
    return np.divide(1, (1 + np.exp(-z)))


def soft_max(z):
    e_z = np.exp(z)
    return e_z / e_z.sum(axis=0)


# In[171]:


def test_network(data, computed_weights, computed_biases):
    first_layer_weights, second_layer_weights = computed_weights[0], computed_weights[1]
    first_layer_biases, second_layer_biases = computed_biases[0], computed_biases[1]
    third_layer_biases, third_layer_weights = computed_biases[2], computed_weights[2]
    good_predictions = 0
    for x, label in zip(data[0], data[1]):
        z1 = np.dot(x, first_layer_weights) + first_layer_biases
        z1 = sigmoid(z1)
        z2 = np.dot(z1, second_layer_weights) + second_layer_biases
        z2 = sigmoid(z2)
        z3 = np.dot(z2, third_layer_weights) + third_layer_biases
        y3 = sigmoid(z3)
        prediction = np.argmax(y3)
        if prediction == label:
            good_predictions += 1
    return good_predictions / len(data[0])


# In[172]:


def fitness_network(x, y, converted_parameters):
    first_layer_weights = converted_parameters[0]
    second_layer_weights = converted_parameters[1]
    third_layer_weights = converted_parameters[2]
    first_layer_biases = converted_parameters[3]
    second_layer_biases = converted_parameters[4]
    third_layer_biases = converted_parameters[5]

    log_loss_result = np.zeros(POP_SIZE)
    y_true = np.broadcast_to(y, (POP_SIZE, *y.shape))
    for start_idx in range(0, x.shape[0], BATCH_SIZE):
        x_batch = x[start_idx:start_idx + BATCH_SIZE]
        # No need to add a dimension to x, implicit broadcasting
        # rules are faster.
        z1 = np.matmul(x_batch, first_layer_weights) + first_layer_biases
        # expit may be better, although it's debatable.
        z1 = expit(z1)
        z2 = np.matmul(z1, second_layer_weights) + second_layer_biases
        z2 = expit(z2)
        z3 = np.matmul(z2, third_layer_weights) + third_layer_biases
        y3 = expit(z3)
        for i in range(len(y3)):
            log_loss_result[i] += log_loss(y_true[i][start_idx:start_idx + BATCH_SIZE], y3[i])
    # axis=1, 0 is population size.
    # Now it works even if the number of instances is not divisible
    # by the size of the minibatch.
    return 1 / (log_loss_result / y_true.shape[1])


# In[173]:


def prepare_individual_for_network(individual):
    """
    This function is called for the same individuals on every call
    of fitness_network. There is no need for this.
    """
    convert_vector = np.vectorize(convert)
    first_layer_weights = np.reshape(np.apply_along_axis(convert_vector, 0, individual[:78400]), (784, 100))
    second_layer_weights = np.reshape(np.apply_along_axis(convert_vector, 0, individual[78400:79400]), (100, 10))
    third_layer_weights = np.reshape(np.apply_along_axis(convert_vector, 0, individual[79400:79500]), (10, 10))
    first_layer_biases = np.apply_along_axis(convert_vector, 0, individual[79500:79600])
    second_layer_biases = np.apply_along_axis(convert_vector, 0, individual[79600:79610])
    third_layer_biases = np.apply_along_axis(convert_vector, 0, individual[79610:])
    return ([first_layer_weights, second_layer_weights, third_layer_weights],
            [first_layer_biases, second_layer_biases, third_layer_biases])


# In[174]:


def selection(population, x_train, y_train):
    new_population = list()
    individual_fitness = [fitness_network(x_train, y_train, *prepare_individual_for_network(ind)) for ind in population]
    total_fitness = sum(individual_fitness)
    individual_probabilities = [ind_fitness / total_fitness for ind_fitness in individual_fitness]
    cumulative_probabilities = [0]
    for i in range(POP_SIZE):
        cumulative_probabilities.append(cumulative_probabilities[i] + individual_probabilities[i])
    size = 0
    while size < POP_SIZE:
        r = random.uniform(0.0001, 1)
        for i in range(POP_SIZE):
            if cumulative_probabilities[i] < r <= cumulative_probabilities[i + 1]:
                if size == POP_SIZE:
                    break
                new_population.append(population[i])
                size += 1
    return new_population


# In[175]:


def upgrade(population):
    for individual in population:
        for row in individual:
            for b in range(len(row)):
                new_bit_array = ''
                for bit in individual[b]:
                    if random.uniform(0, 1) < MUTATION_PROB:
                        if bit == '1':
                            new_bit_array += '0'
                        else:
                            new_bit_array += '1'
                    else:
                        new_bit_array += bit
                individual[b] = new_bit_array

    cross_over_indexes = [i for i in range(POP_SIZE) if random.uniform(0, 1) < 0.6]
    if len(cross_over_indexes) % 2 != 0:
        cross_over_indexes = cross_over_indexes[:-1]
    for i in range(0, len(cross_over_indexes), 2):
        first_child = list()
        second_child = list()
        # TODO: Reimplementation with the new data structure
        for k in range(len(population[cross_over_indexes[i]])):
            bit_array_length = len(population[cross_over_indexes[i]][k])
            r = random.randint(0, bit_array_length - 1)
            new_first_array = population[cross_over_indexes[i]][k][:r] + population[cross_over_indexes[i + 1]][k][r:]
            new_second_array = population[cross_over_indexes[i]][k][r:] + population[cross_over_indexes[i + 1]][k][:r]
            first_child.append(new_first_array)
            second_child.append(new_second_array)
        population[cross_over_indexes[i]] = list(first_child)
        population[cross_over_indexes[i + 1]] = list(second_child)
    return population


# In[176]:


def evaluate_population(x, y, population):
    convert_vector = np.vectorize(convert)
    # best_individual = np.zeros(len(population[0]))
    best_individual = None
    fitness_values = fitness_network(x, y,
                                     tuple(convert_vector(layer)
                                           for params in population
                                           for layer in params))
    local_best = np.argmax(fitness_values)
    best = fitness_values[local_best]
    # Ugly, but faster than other ways I can think of.
    best_individual = (# weights
                       population[0][0][local_best],
                       population[0][1][local_best],
                       population[0][2][local_best],
                       # biases
                       population[1][0][local_best],
                       population[1][1][local_best],
                       population[1][2][local_best])
    return best, best_individual


# In[177]:


def generate_population():
    first_layer_weights = np.apply_along_axis(''.join,
                                              1,
                                              np.random.choice(['0', '1'], size=(POP_SIZE, BITS_NR, 784, 100)))
    second_layer_weights = np.apply_along_axis(''.join,
                                               1,
                                               np.random.choice(['0', '1'], size=(POP_SIZE, BITS_NR, 100, 10)))
    third_layer_weights = np.apply_along_axis(''.join,
                                              1,
                                              np.random.choice(['0', '1'], size=(POP_SIZE, BITS_NR, 10, 10)))
    
    first_layer_biases = np.apply_along_axis(''.join,
                                             2,
                                             np.random.choice(['0', '1'], size=(POP_SIZE, 100, BITS_NR)))[:, np.newaxis, :]
    second_layer_biases = np.apply_along_axis(''.join,
                                              2,
                                              np.random.choice(['0', '1'], size=(POP_SIZE, 10, BITS_NR)))[:, np.newaxis, :]
    third_layer_biases = np.apply_along_axis(''.join,
                                             2,
                                             np.random.choice(['0', '1'], size=(POP_SIZE, 10, BITS_NR))) [:, np.newaxis, :]
    weights = (first_layer_weights, second_layer_weights, third_layer_weights)
    biases = (first_layer_biases, second_layer_biases, third_layer_biases)
    return weights, biases


# In[178]:


def generate_all_individuals():
    population = []
    for _ in range(POP_SIZE):
        first_layer_weights = np.apply_along_axis(''.join, 1, np.random.choice(['0', '1'], size=(78400, BITS_NR)))
        second_layer_weights = np.apply_along_axis(''.join, 1, np.random.choice(['0', '1'], size=(1000, BITS_NR)))
        third_layer_weights = np.apply_along_axis(''.join, 1, np.random.choice(['0', '1'], size=(100, BITS_NR)))

        first_layer_biases = np.apply_along_axis(''.join, 1, np.random.choice(['0', '1'], size=(100, BITS_NR)))
        second_layer_biases = np.apply_along_axis(''.join, 1, np.random.choice(['0', '1'], size=(10, BITS_NR)))
        third_layer_biases = np.apply_along_axis(''.join, 1, np.random.choice(['0', '1'], size=(10, BITS_NR)))
        individual = np.concatenate([first_layer_weights, second_layer_weights, third_layer_weights,
                                     first_layer_biases, second_layer_biases, third_layer_biases])
        population.append(individual)
    return population


# In[179]:


def main():
    start_time = time.time()
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        x_train, y_train = train_set
        x_valid, y_valid = valid_set
        x_test, y_test = test_set
    #best = 0
    population = generate_population()
    best, best_individual = evaluate_population(x_train, y_train, population)
    for i in range(NR_EPOCHS):
        print(f'Current epoch: {i}')
        population = selection(population, x_train, y_train)
        population = list(upgrade(population))
        new_best, new_best_individual = evaluate_population(x_train, y_train, population)
        if new_best > best:
            best = new_best
            #best_individual = np.copy(temp_individual)
            best_individual = new_best_individual
    best_score = test_network(train_set, *prepare_individual_for_network(best_individual))
    print(f'The network achieved an accuracy of {best_score * 100} percent on training set!')
    best_score = test_network(test_set, *prepare_individual_for_network(best_individual))
    print(f'The network achieved an accuracy of {best_score * 100} percent on testing set!')
    print(f'Time taken: {time.time() - start_time} seconds!')


# In[180]:



# In[181]:


# get_ipython().run_line_magic('load_ext', 'line_profiler')


# In[182]:


#%lprun -f evaluate_population evaluate_population(x_train, y_train, population)
#%timeit evaluate_population(x_train, y_train, population)


# In[26]:


# A1 = np.random.randint(1, high=5, size=(30, 256, 768))
# B1 = np.random.randint(1, high=5, size=(30, 768, 100))
# A2 = np.random.randint(1, high=5, size=(30, 256, 768))
# B2 = np.random.randint(1, high=5, size=(30, 768, 100))
# #B2 = np.asfortranarray(B2)
#
# print('A1')
# print(A1.flags)
# print('B1')
# print(B1.flags)
# print('A2')
# print(A2.flags)
# print('B2')
# print(B2.flags)


# In[22]:


# get_ipython().run_line_magic('timeit', 'np.matmul(A1, B1)')


# In[27]:


# get_ipython().run_cell_magic('timeit', '', 'global B2, A2\nA2 = np.asfortranarray(A2)\n#B2 = np.asfortranarray(B2)\nnp.matmul(A2, B2)')


# In[30]:


# M1 = np.apply_along_axis(''.join,
#                          1,
#                          np.random.choice(['0', '1'], size=(POP_SIZE, BITS_NR, 784, 100)))
# uint32_limit = np.iinfo(np.uint32).max
# M2 = np.random.randint(0, high=uint32_limit, size=(POP_SIZE, 784, 100))


# In[33]:


def mutate_one(M):
    for individual in M:
        for row in individual:
            for b in range(len(row)):
                new_bit_array = ''
                for bit in individual[b]:
                    if random.uniform(0, 1) < MUTATION_PROB:
                        if bit == '1':
                            new_bit_array += '0'
                        else:
                            new_bit_array += '1'
                    else:
                        new_bit_array += bit
                individual[b] = new_bit_array


# In[61]:


def mutate_two(M):
    pass

# mat = np.array([0xF, 0xFF, 0X43])
# a = np.bitwise_and(mat[:, np.newaxis], BIT_TABLE) != 0
# p = np.random.rand(*mat.shape)
# print(a)


# In[60]:


# get_ipython().run_line_magic('timeit', 'mutate_one(M1)')


# In[ ]:


# TODO: Do some tests on mutation
# M1 = array of constant length strings
# M2 = array of uint32
# %timeit mutate(M1) (as written in upgrade function)
# %timeit mutate(M1) (with vectorization and other optimizations)
# %timeit mutate(M1) (use Cython?)
# %timeit mutate(M2) (uint32 and bit-twiddling)
# %timeit mutate(M2) (use Cython)

if __name__ == '__main__':
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        x_train, y_train = train_set
        x_valid, y_valid = valid_set
        x_test, y_test = test_set
    population = generate_population()
    evaluate_population(x_train, y_train, population)
