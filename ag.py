#!/usr/bin/env python
# coding: utf-8
import numpy as np
import math
import gzip
import pickle
import random
import time
from sklearn.metrics import log_loss
from scipy.special import expit, softmax

NR_EPOCHS = 20
POP_SIZE = 100
HIGHER_BOUND = 10
LOWER_BOUND = -10
INTERVALS_NR = (HIGHER_BOUND - LOWER_BOUND) * 10 ** 4
BITS_NR = math.ceil(np.log2(INTERVALS_NR))
MUTATION_PROB = 0.03
CROSSOVER_PROB = 0.6
BATCH_SIZE = 256
IDXS = 2 ** np.arange(BITS_NR)[::-1]


def convert_bits(bits, indices):
    return bits.dot(indices)


def convert(m):
    convert_bits_vect = np.vectorize(convert_bits,
                                     otypes=[np.uint32],
                                     signature='(m,n),(n)->(m)')
    result = convert_bits_vect(m, IDXS) / (2 ** BITS_NR - 1)
    return result * (HIGHER_BOUND - LOWER_BOUND) + LOWER_BOUND


# Activation functions
# TODO: See which is more efficient
def sigmoid(z):
    return np.divide(1, (1 + np.exp(-z)))


def soft_max(z):
    e_z = np.exp(z)
    return e_z / e_z.sum(axis=0)


def expit_approx(x):
    return 1.0 / (1 + np.abs(x))


def softplus(x):
    return np.log(1 + np.exp(x))


# expit imported from scipy.special


def fitness_network(x, y, params, testing=False):
    first_layer_weights = params[0]
    second_layer_weights = params[1]
    third_layer_weights = params[2]
    first_layer_biases = params[3]
    second_layer_biases = params[4]
    third_layer_biases = params[5]
    y_pred = list()
    for start_idx in range(0, x.shape[0], BATCH_SIZE):
        x_batch = x[start_idx:start_idx + BATCH_SIZE]
        z1 = np.matmul(x_batch, first_layer_weights) + first_layer_biases
        # expit may be better, although it's debatable.
        z1 = expit(z1)
        z2 = np.matmul(z1, second_layer_weights) + second_layer_biases
        z2 = expit(z2)
        z3 = np.matmul(z2, third_layer_weights) + third_layer_biases
        y3 = softmax(z3)
        y3 = y3.argmax(axis=2)
        y_pred.append(y3)
    if not testing:
        y_pred = np.concatenate(y_pred, axis=1)
        y_true = np.broadcast_to(y, (y_pred.shape[0], *y.shape))
        return [1 / log_loss(y_true[i], y_pred[i])
                for i in range(y_pred.shape[0])]
    else:
        y_pred = np.concatenate(y_pred)
        y_pred = np.apply_along_axis(np.argmax, 1, y_pred)
        return (y_pred == y).sum() / y.size


def mutate(m):
    return np.where(np.random.rand(*m.shape) < MUTATION_PROB,
                    1 - m,
                    m)


def crossover(m, cross_percentages):
    def swap_weights(m, i1, i2):
        n_pop = len(i1)
        i = np.random.randint(m.shape[1], size=(n_pop,))
        j = np.random.randint(m.shape[2], size=(n_pop,))
        for i1_idx, i2_idx in zip(i1, i2):
            temp = m[i1_idx, i, j].copy()
            m[i1_idx, i, j] = m[i2_idx, i, j]
            m[i2_idx, i, j] = temp

    def swap_neurons(m, i1, i2):
        n_pop = len(i1)
        i = np.random.randint(m.shape[1], size=(n_pop,))
        for i1_idx, i2_idx in zip(i1, i2):
            temp = m[i1_idx, i].copy()
            m[i1_idx, i] = m[i2_idx, i]
            m[i2_idx, i] = temp

    def swap_layers(m, i1, i2):
        for i1_idx, i2_idx in zip(i1, i2):
            temp = m[i1_idx].copy()
            m[i1_idx] = m[i2_idx]
            m[i2_idx] = temp

    def split_perc(indices, perc):
        # Turn percentages into values between 0 and 1
        splits = np.cumsum(perc)
        if splits[-1] != 1:
            raise ValueError("percents don't add up to 100")
        # Split doesn't need last percent, it will just take what is left
        splits = splits[:-1]
        # Turn values into indices
        splits *= len(indices)
        # Turn double indices into integers.
        # CAUTION: numpy rounds to closest EVEN number when a number is halfway
        # between two integers. So 0.5 will become 0 and 1.5 will become 2!
        # If you want to round up in all those cases, do
        # splits += 0.5 instead of round() before casting to int
        splits = splits.round().astype(np.int)
        splits = np.split(indices, splits)
        # Make arrays of even lengths
        for i in range(len(splits)):
            if len(splits[i]) % 2:
                splits[i] = np.append(splits[i],
                                      np.random.choice(splits[i],
                                                       size=(1,)))
        return splits

    # ACTUAL FUNCTION LOGIC STARTS HERE

    cross_indices = np.arange(POP_SIZE)[np.random.rand(POP_SIZE) < CROSSOVER_PROB]
    shuffled_indices = np.random.choice(cross_indices,
                                        size=cross_indices.size,
                                        replace=False)
    weights, neurons, layers = split_perc(shuffled_indices, cross_percentages)
    swap_weights(m, *np.split(weights, 2))
    swap_neurons(m, *np.split(neurons, 2))
    swap_layers(m, *np.split(layers, 2))


def upgrade(population, cross_percentages=(.3, .3, .4)):
    new_population = []
    for i in range(len(population)):
        layer = population[i]
        layer_new = mutate(layer)
        # This function modifies the matrix in-place
        crossover(layer_new, cross_percentages)
        new_population.append(layer_new)
    return tuple(new_population)


def selection(population, fitness_values):
    new_population = list()
    total_fitness = sum(fitness_values)
    individual_probabilities = [fitness_val / total_fitness for fitness_val in fitness_values]
    cumulative_probabilities = [0]
    for i in range(POP_SIZE):
        cumulative_probabilities.append(cumulative_probabilities[i] + individual_probabilities[i])
    # Do this for each layer.
    for layer in population:
        new_layer = []
        size = 0
        while size < POP_SIZE:
            r = random.uniform(0.0001, 1)
            for i in range(POP_SIZE):
                if cumulative_probabilities[i] < r <= cumulative_probabilities[i + 1]:
                    if size == POP_SIZE:
                        break
                    new_layer.append(layer[i])
                    size += 1
        new_population.append(np.array(new_layer))
    return new_population


def get_best_individual(population, fitness_values):
    # best_individual = np.zeros(len(population[0]))
    local_best = np.argmax(fitness_values)
    best = fitness_values[local_best]
    # Ugly, but faster than other ways I can think of.
    best_individual = (  # weights
        population[0][local_best],
        population[1][local_best],
        population[2][local_best],
        # biases
        population[3][local_best],
        population[4][local_best],
        population[5][local_best])
    return best, best_individual


def generate_population():
    first_layer_weights = np.random.randint(2,
                                            size=(POP_SIZE, 784, 100, BITS_NR),
                                            dtype=np.uint8)
    second_layer_weights = np.random.randint(2,
                                             size=(POP_SIZE, 100, 10, BITS_NR),
                                             dtype=np.uint8)
    third_layer_weights = np.random.randint(2,
                                            size=(POP_SIZE, 10, 10, BITS_NR),
                                            dtype=np.uint8)
    first_layer_biases = np.random.randint(2,
                                           size=(POP_SIZE, 100, BITS_NR),
                                           dtype=np.uint8)[:, np.newaxis, :, :]
    second_layer_biases = np.random.randint(2,
                                            size=(POP_SIZE, 10, BITS_NR),
                                            dtype=np.uint8)[:, np.newaxis, :, :]
    third_layer_biases = np.random.randint(2,
                                           size=(POP_SIZE, 10, BITS_NR),
                                           dtype=np.uint8)[:, np.newaxis, :, :]
    return (first_layer_weights, second_layer_weights,
            third_layer_weights, first_layer_biases,
            second_layer_biases, third_layer_biases)


def convert_population(population):
    return tuple(convert(layer)
                 for layer in population)


def main():
    start_time = time.time()
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, _, test_set = pickle.load(f, encoding='latin1')
        x_train, y_train = train_set
        x_test, y_test = test_set
    # best = 0
    population = generate_population()

    fitness_values = fitness_network(x_train, y_train, convert_population(population))
    best, best_individual = get_best_individual(population, fitness_values)
    for i in range(NR_EPOCHS):
        print(f'Current epoch: {i}')
        population = selection(population, fitness_values)
        population = upgrade(population, cross_percentages=[.3, .3, .4])
        fitness_values = fitness_network(x_train, y_train, convert_population(population))
        new_best, new_best_individual = get_best_individual(population, fitness_values)
        if new_best > best:
            best = new_best
            # best_individual = np.copy(temp_individual)
            best_individual = new_best_individual
        best_score = fitness_network(x_train, y_train, convert_population(best_individual), testing=True)
        print(f'The network achieved an accuracy of {best_score * 100} percent on training set!')
    best_score = fitness_network(x_test, y_test, convert_population(best_individual), testing=True)
    print(f'The network achieved an accuracy of {best_score * 100} percent on testing set!')
    print(f'Time taken: {time.time() - start_time} seconds!')


if __name__ == '__main__':
    main()
