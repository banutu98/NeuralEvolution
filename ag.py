#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import os
import gzip
import pickle
import random
import time

from sklearn.metrics import log_loss
from scipy.special import expit, softmax

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
import copy

NR_EPOCHS = 500
POP_SIZE = 100
ELITISM_NR = 10
HIGHER_BOUND = 1
LOWER_BOUND = -1
# 95% of values will be between LOWER_BOUND and HIGHER_BOUND
# if mean centered
SCALE = ((HIGHER_BOUND - LOWER_BOUND) / 2) / 2
INTERVALS_NR = (HIGHER_BOUND - LOWER_BOUND) * 10 ** 4
BITS_NR = math.ceil(np.log2(INTERVALS_NR))
MUTATION_PROB = 0.01
CROSSOVER_PROB = 0.6
BATCH_SIZE = 256
# 1 input, 1 hidden, 1 output = 3 layers
N_UNITS = (784, 16, 10)
N_WEIGHTS = len(N_UNITS) - 1
N_BIASES = N_WEIGHTS


# Activation functions
def sigmoid(z):
    return np.divide(1, (1 + np.exp(-z)))


def expit_approx(x):
    return 1.0 / (1 + np.abs(x))


def softplus(x):
    return np.log(1 + np.exp(x))


# expit imported from scipy.special


def fitness_network(population, x, y, metric='log_loss'):
    losses = []
    allowed_metrics = ['log_loss', 'acc']
    if not population:
        return []
    if metric not in allowed_metrics:
        raise NotImplemented(f'{metric} not implemented. Allowed metrics: {allowed_metrics}')
    pop = [population] if not isinstance(population, list) else population
    n_weights = len(pop[0]) // 2
    for individual in pop:
        weights = individual[:n_weights]
        biases = individual[n_weights:]
        y_pred = list()
        for start_idx in range(0, x.shape[0], BATCH_SIZE):
            x_batch = x[start_idx:start_idx + BATCH_SIZE]
            z = x_batch
            for i in range(n_weights - 1):
                z = np.dot(z, weights[i]) + biases[i]
                # expit may be better, although it's debatable.
                z = expit(z)
            z = np.dot(z, weights[n_weights - 1]) + biases[n_weights - 1]
            y_final = softmax(z)
            y_pred.append(y_final)
        y_pred = np.concatenate(y_pred)
        if metric == 'log_loss':
            losses.append(1 / np.exp(log_loss(y, y_pred)))
        else:
            y_pred = np.apply_along_axis(np.argmax, 1, y_pred)
            losses.append(np.sum(y_pred == y) / y.size)
    return losses


def test_network(individual, x, y):
    n_weights = len(individual) // 2
    weights = individual[:n_weights]
    biases = individual[n_weights:]
    y_pred = list()
    for start_idx in range(0, x.shape[0], BATCH_SIZE):
        x_batch = x[start_idx:start_idx + BATCH_SIZE]
        z = x_batch
        for i in range(n_weights - 1):
            z = np.dot(z, weights[i]) + biases[i]
            # expit may be better, although it's debatable.
            z = expit(z)
        z = np.dot(z, weights[n_weights - 1]) + biases[n_weights - 1]
        y_final = softmax(z)
        y_pred.append(y_final)
    y_pred = np.concatenate(y_pred)
    y_pred = np.apply_along_axis(np.argmax, 1, y_pred)
    return np.sum(y_pred == y) / y.size


def mutate(pop):
    new_pop = []
    for indiv in pop:
        new_indiv = []
        for layer in indiv:
            r = np.random.rand()
            if r < 0.5:
                new_indiv.append(np.where(np.random.rand(*layer.shape) < MUTATION_PROB,
                                          layer + np.random.normal(loc=0,
                                                                   scale=SCALE,
                                                                   size=layer.shape),
                                          layer))
            else:
                new_indiv.append(np.where(np.random.rand(*layer.shape) < MUTATION_PROB,
                                          layer - np.random.normal(loc=0,
                                                                   scale=SCALE,
                                                                   size=layer.shape),
                                          layer))
        new_pop.append(new_indiv)
    return new_pop


def crossover(pop, cross_percentages):
    def swap_weights(p, i1, i2):
        for i1_idx, i2_idx in zip(i1, i2):
            # choose a random layer (weights only)
            l = random.randint(0, N_WEIGHTS - 1)
            i = random.randint(0, p[i1_idx][l].shape[0] - 1)
            j = random.randint(0, p[i1_idx][l].shape[1] - 1)
            temp = p[i1_idx][l][i, j].copy()
            p[i1_idx][l][i, j] = p[i2_idx][l][i, j]
            p[i2_idx][l][i, j] = temp

    def swap_neurons(p, i1, i2):
        for i1_idx, i2_idx in zip(i1, i2):
            # choose a random layer (weights and biases)
            l = random.randint(0, N_WEIGHTS + N_BIASES - 1)
            i = random.randint(0, p[i1_idx][l].shape[0] - 1)
            temp = p[i1_idx][l][i].copy()
            p[i1_idx][l][i] = p[i2_idx][l][i]
            p[i2_idx][l][i] = temp

    def swap_layers(p, i1, i2):
        for i1_idx, i2_idx in zip(i1, i2):
            # choose a random layer (weights and biases)
            l = random.randint(0, N_WEIGHTS + N_BIASES - 1)
            temp = p[i1_idx][l].copy()
            p[i1_idx][l] = p[i2_idx][l]
            p[i2_idx][l] = temp

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
    swap_weights(pop, *np.split(weights, 2))
    swap_neurons(pop, *np.split(neurons, 2))
    swap_layers(pop, *np.split(layers, 2))


def upgrade(population, cross_percentages=(.3, .3, .4)):
    new_population = mutate(population)
    # This function modifies the matrix in-place
    crossover(new_population, cross_percentages)
    return new_population


def selection(population, fitness_values, elitism=False, strategy='roulette', ranking='proportional', base=0.95):
    new_population = []
    if ranking == 'proportional':
        new_fitness = fitness_values
    elif ranking == 'slots':
        # Get permutation that sorts array in decreasing order
        # (highest fitness first).
        sorted_indices = np.argsort(fitness_values)[::-1]
        # Get inverse permutation in order to bring array back
        # to original order.
        inverse_perm = np.argsort(sorted_indices)
        # New fitness values take the form a^i, where i = 0..n-1.
        # Fitness is assigned based on ranking.
        new_fitness = base ** np.arange(sorted_indices.size)
        # Sort back to original order using inverse permutation.
        new_fitness = new_fitness[inverse_perm]
        # print(fitness_values)
        # print(new_fitness)
    # Compute cumulative distribution.
    total_fitness = sum(new_fitness)
    individual_probabilities = [fitness_val / total_fitness for fitness_val in new_fitness]
    cummulative_probabilities = np.cumsum(individual_probabilities)
    if not elitism:
        # Generate probabilities for new population.
        r = np.random.rand(POP_SIZE)
        # Get insertion points through a left bisect algorithm.
        selected = np.searchsorted(cummulative_probabilities, r)
        for idx in selected:
            new_population.append(population[idx])
    else:
        best_fitness_values = sorted(fitness_values, reverse=True)[:ELITISM_NR]
        chosen_elitism_values = [np.where(fitness_values == i)[0][0] for i in best_fitness_values]
        # Generate probabilities for new population.
        r = np.random.rand(POP_SIZE - ELITISM_NR)
        # Get insertion points through a left bisect algorithm.
        selected = np.searchsorted(cummulative_probabilities, r)
        new_population.extend([population[idx] for idx in selected])
        new_population.extend([population[idx] for idx in chosen_elitism_values])
    return new_population


def get_best_individual(population, fitness_values):
    local_best = np.argmax(fitness_values)
    best = fitness_values[local_best]
    best_individual = population[local_best]
    return best, best_individual


def build_model():
    input_layer = Input(shape=(784,))
    dense_1 = Dense(16, activation='sigmoid')(input_layer)
    # dense_2 = Dense(10, activation='sigmoid')(dense_1)
    pred = Dense(10, activation='softmax')(dense_1)
    model = Model(inputs=input_layer, outputs=pred)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model


def generate_smart_population(x_train, y_train, load=False):
    if not load:
        model = build_model()
        model.fit(x_train, to_categorical(y_train, num_classes=10), batch_size=BATCH_SIZE, epochs=1)
        model.save('model.h5')
    else:
        if os.path.exists('model.h5'):
            model = load_model('model.h5')
        else:
            model = build_model()
            model.fit(x_train, to_categorical(y_train, num_classes=10), batch_size=BATCH_SIZE, epochs=1)
            model.save('model.h5')

    loss, acc = model.evaluate(x_train, to_categorical(y_train))
    print(f'Accuracy from the initial model: {acc}')

    first_layer_weights = model.layers[1].get_weights()[0]
    first_layer_biases = model.layers[1].get_weights()[1]
    second_layer_weights = model.layers[2].get_weights()[0]
    second_layer_biases = model.layers[2].get_weights()[1]
    # third_layer_weights = model.layers[3].get_weights()[0]
    # third_layer_biases = model.layers[3].get_weights()[1]

    return [[np.copy(first_layer_weights), np.copy(second_layer_weights),
             np.copy(first_layer_biases), np.copy(second_layer_biases)]
            for _ in range(POP_SIZE)]


def generate_population(units=N_UNITS):
    return [[np.random.uniform(low=LOWER_BOUND,
                               high=HIGHER_BOUND,
                               size=(units[i], units[i + 1])).astype('f')
             for i in range(len(units) - 1)]
            +
            [np.random.uniform(low=LOWER_BOUND,
                               high=HIGHER_BOUND,
                               size=(units[i + 1],)).astype('f')
             for i in range(len(units) - 1)]
            for _ in range(POP_SIZE)]


def print_parameters(**kwargs):
    print('PARAMETERS')
    print('######################################')
    print('Mutation rate:', MUTATION_PROB)
    print('Crossover rate:', CROSSOVER_PROB)
    print('Population size:', POP_SIZE)
    print('Epochs:', NR_EPOCHS)
    print('Layers:', N_UNITS)
    print('Batch size:', BATCH_SIZE)
    print('Scale:', SCALE)
    print('Init interval:', f'({LOWER_BOUND}, {HIGHER_BOUND})')
    print('Used elitism?', kwargs.get('elitism'))
    print('Elitism number:', ELITISM_NR)
    print('Backprop init?', kwargs.get('use_back_prop'))
    print('######################################')


def main(use_back_prop=True, load=True, elitism=True, ranking='proportional', mutation_prob=0.01):
    global MUTATION_PROB
    MUTATION_PROB = mutation_prob
    parameters = locals()
    print_parameters(**parameters)
    start_time = time.time()
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, _, test_set = pickle.load(f, encoding='latin1')
        x_train, y_train = train_set
        x_test, y_test = test_set
    if load:
        if os.path.exists('population.pkl'):
            with open('population.pkl', 'rb') as f:
                population = pickle.load(f)
        else:
            if not use_back_prop:
                population = generate_population()
            else:
                population = generate_smart_population(x_train, y_train, load=load)
    else:
        if not use_back_prop:
            population = generate_population()
        else:
            population = generate_smart_population(x_train, y_train, load=load)

    fitness_values = fitness_network(population, x_train, y_train)
    best, best_individual = get_best_individual(population, fitness_values)
    for i in range(NR_EPOCHS):
        if (i + 1) % 10 == 0:
            with open('population.pkl', 'wb') as f:
                pickle.dump(population, f)
        print(f'Current epoch: {i}')
        population = selection(population, fitness_values, elitism=elitism, ranking=ranking)
        population = upgrade(population, cross_percentages=[.40, .55, .05])
        fitness_values = fitness_network(population, x_train, y_train)
        new_best, new_best_individual = get_best_individual(population, fitness_values)
        print('Current best:', best)
        print('New best:', new_best)
        if new_best > best:
            best = new_best
            best_individual = new_best_individual
            best_score = test_network(best_individual, x_train, y_train)
            print(f'Training accuracy: {best_score * 100}')
            best_score = test_network(best_individual, x_test, y_test)
            print(f'Testing accuracy: {best_score * 100}')
    best_score = test_network(best_individual, x_test, y_test)
    print(f'The network achieved an accuracy of {best_score * 100} percent on testing set!')
    print(f'Time taken: {time.time() - start_time} seconds!')


if __name__ == '__main__':
    main(use_back_prop=True, load=False, ranking='slots', elitism=True, mutation_prob=0.01)
