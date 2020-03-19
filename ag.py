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
from scipy.linalg.blas import sgemm

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical


NR_EPOCHS = 200
POP_SIZE = 100
ELITISM_NR = 10
HIGHER_BOUND = 1
LOWER_BOUND = -1
INTERVALS_NR = (HIGHER_BOUND - LOWER_BOUND) * 10 ** 4
BITS_NR = math.ceil(np.log2(INTERVALS_NR))
MUTATION_PROB = 0.1
CROSSOVER_PROB = 0.6
BATCH_SIZE = 256


# Activation functions
# TODO: See which is more efficient
def sigmoid(z):
    return np.divide(1, (1 + np.exp(-z)))


def expit_approx(x):
    return 1.0 / (1 + np.abs(x))


def softplus(x):
    return np.log(1 + np.exp(x))


# expit imported from scipy.special


def fitness_network(population, x, y):
    losses = []
    for individual in population:
        first_layer_weights = individual[0]
        second_layer_weights = individual[1]
        third_layer_weights = individual[2]
        first_layer_biases = individual[3]
        second_layer_biases = individual[4]
        third_layer_biases = individual[5]
        y_pred = list()
        for start_idx in range(0, x.shape[0], BATCH_SIZE):
            x_batch = x[start_idx:start_idx + BATCH_SIZE]
            z1 = np.dot(x_batch, first_layer_weights) + first_layer_biases
            # expit may be better, although it's debatable.
            z1 = expit(z1)
            z2 = np.dot(z1, second_layer_weights) + second_layer_biases
            z2 = expit(z2)
            z3 = np.dot(z2, third_layer_weights) + third_layer_biases
            y3 = softmax(z3)
            y_pred.append(y3)
        y_pred = np.concatenate(y_pred)
        losses.append(1/np.exp(log_loss(y, y_pred)))
    return losses


def test_network(individual, x, y):
    first_layer_weights = individual[0]
    second_layer_weights = individual[1]
    third_layer_weights = individual[2]
    first_layer_biases = individual[3]
    second_layer_biases = individual[4]
    third_layer_biases = individual[5]
    y_pred = list()
    for start_idx in range(0, x.shape[0], BATCH_SIZE):
        x_batch = x[start_idx:start_idx + BATCH_SIZE]
        z1 = np.dot(x_batch, first_layer_weights) + first_layer_biases
        # expit may be better, although it's debatable.
        z1 = expit(z1)
        z2 = np.dot(z1, second_layer_weights) + second_layer_biases
        z2 = expit(z2)
        z3 = np.dot(z2, third_layer_weights) + third_layer_biases
        y3 = softmax(z3)
        y_pred.append(y3)
    y_pred = np.concatenate(y_pred)
    y_pred = np.apply_along_axis(np.argmax, 1, y_pred)
    return np.sum(y_pred == y) / y.size


def mutate(pop):
    new_pop = []
    for indiv in pop:
        new_indiv = []
        for layer in indiv:
            new_indiv.append(np.where(np.random.rand(*layer.shape) < MUTATION_PROB,
                                      np.random.uniform(low=LOWER_BOUND,
                                                        high=HIGHER_BOUND,
                                                        size=layer.shape),
                                      layer))
        new_pop.append(new_indiv)
    return new_pop


def crossover(pop, cross_percentages):
    def swap_weights(p, i1, i2):
        for i1_idx, i2_idx in zip(i1, i2):
            # choose a random layer (weights only)
            l = random.randint(0, 2)
            i = random.randint(0, p[i1_idx][l].shape[0]-1)
            j = random.randint(0, p[i1_idx][l].shape[1]-1)
            temp = p[i1_idx][l][i, j].copy()
            p[i1_idx][l][i, j] = p[i2_idx][l][i, j]
            p[i2_idx][l][i, j] = temp

    def swap_neurons(p, i1, i2):
        for i1_idx, i2_idx in zip(i1, i2):
            # choose a random layer (weights and biases)
            l = random.randint(0, 5)
            i = random.randint(0, p[i1_idx][l].shape[0]-1)
            temp = p[i1_idx][l][i].copy()
            p[i1_idx][l][i] = p[i2_idx][l][i]
            p[i2_idx][l][i] = temp

    def swap_layers(p, i1, i2):
        for i1_idx, i2_idx in zip(i1, i2):
            # choose a random layer (weights and biases)
            l = random.randint(0, 5)
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


def selection(population, fitness_values):
    new_population = []
    #best_fitness_values = sorted(fitness_values, reverse=True)[:ELITISM_NR]
    #chosen_elitism_values = [np.where(fitness_values == i)[0][0] for i in best_fitness_values]
    # Compute cumulative distribution.
    total_fitness = sum(fitness_values)
    individual_probabilities = [fitness_val / total_fitness for fitness_val in fitness_values]
    cummulative_probabilities = np.cumsum(individual_probabilities)
    # Generate probabilities for new population.
    r = np.random.rand(POP_SIZE)
    # Get insertion points through a left bisect algorithm.
    selected = np.searchsorted(cummulative_probabilities, r)
    for idx in selected:
        new_population.append(population[idx])
    return new_population


def get_best_individual(population, fitness_values):
    local_best = np.argmax(fitness_values)
    best = fitness_values[local_best]
    best_individual = population[local_best]
    return best, best_individual


def generate_population():
    return [[np.random.uniform(low=LOWER_BOUND,
                               high=HIGHER_BOUND,
                               size=(784, 100)).astype('f'),
             np.random.uniform(low=LOWER_BOUND,
                               high=HIGHER_BOUND,
                               size=(100, 10)).astype('f'),
             np.random.uniform(low=LOWER_BOUND,
                               high=HIGHER_BOUND,
                               size=(10, 10)).astype('f'),
             np.random.uniform(low=LOWER_BOUND,
                               high=HIGHER_BOUND,
                               size=(100,)).astype('f'),
             np.random.uniform(low=LOWER_BOUND,
                               high=HIGHER_BOUND,
                               size=(10,)).astype('f'),
             np.random.uniform(low=LOWER_BOUND,
                               high=HIGHER_BOUND,
                               size=(10,)).astype('f')]
            for _ in range(POP_SIZE)]


def main():
    start_time = time.time()
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, _, test_set = pickle.load(f, encoding='latin1')
        x_train, y_train = train_set
        x_test, y_test = test_set
    population = generate_population()
    print(population[0][0].shape[0])
    fitness_values = fitness_network(population, x_train, y_train)
    best, best_individual = get_best_individual(population, fitness_values)
    for i in range(NR_EPOCHS):
        print(f'Current epoch: {i}')
        old_population = population
        population = selection(population, fitness_values)
        population = upgrade(population, cross_percentages=[.40, .55, .05])
        #print('populations equal?')
        old_fitness_values = fitness_values
        fitness_values = fitness_network(population, x_train, y_train)
        print('fitness equal?',
              np.allclose(fitness_values,old_fitness_values))
        new_best, new_best_individual = get_best_individual(population, fitness_values)
        print('current best:', best)
        print('new best:', new_best)
        if new_best > best:
            best = new_best
            best_individual = new_best_individual
            best_score = test_network(best_individual, x_train, y_train)
            print(f'The network achieved an accuracy of {best_score * 100} percent on training set!')
    best_score = test_network(best_individual, x_test, y_test)
    print(f'The network achieved an accuracy of {best_score * 100} percent on testing set!')
    print(f'Time taken: {time.time() - start_time} seconds!')


if __name__ == '__main__':
    main()
