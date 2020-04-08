import pickle as pkl
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical

SWARM_SIZE = 30
MAX_LAYERS = 20
MAX_FEATURE_MAPS = 4
MAX_KERNEL_SIZE = 7
MAX_FC_NEURONS = 32
OUTPUT_NEURONS = 10
PSO_ITER = 100
NN_TRAIN_EPOCHS = 5
NN_TEST_EPOCHS = 10
NN_BATCH_SIZE = 100
G_BEST_PROB = 0.7


class ParticleModel:

    def __init__(self):
        self.fc_layers = list()
        self.conv_layers = list()
        self.loss = 9999

    def compute_loss(self, train_data, test_data):
        model = build_model(self.conv_layers, self.fc_layers)
        model.fit(train_data[0], train_data[1], batch_size=NN_BATCH_SIZE, epochs=NN_TRAIN_EPOCHS)
        loss, acc = model.evaluate(test_data[0], test_data[1], batch_size=NN_BATCH_SIZE)
        self.loss = loss


class ParticleValue:

    def __init__(self, particle: ParticleModel):
        self.particle = particle
        self.best_particle = particle
        self.velocity = [[], []]

    def update_velocity(self, g_best: ParticleModel):
        self.velocity = [[], []]
        g_best_diff = compute_particle_difference(g_best, self.particle)
        p_best_diff = compute_particle_difference(self.best_particle, self.particle)
        if check_for_similarity(g_best_diff, p_best_diff):
            self.velocity = [[0] * len(g_best_diff[0]), [0] * len(g_best_diff[1])]
        else:
            max_conv = max(len(g_best_diff[0]), len(p_best_diff[0]))
            max_fc = max(len(g_best_diff[1]), len(p_best_diff[1]))
            g_best_diff[1], p_best_diff[1] = pad_layer_sequences(g_best_diff[0], p_best_diff[0], g_best_diff[1],
                                                                 p_best_diff[1], max_conv, max_fc)
            for i in range(len(g_best_diff[0])):
                if np.random.uniform() < G_BEST_PROB:
                    self.velocity[0].append(g_best_diff[0][i])
                else:
                    self.velocity[0].append(p_best_diff[0][i])
            for i in range(len(g_best_diff[1])):
                if np.random.uniform() < G_BEST_PROB:
                    self.velocity[1].append(g_best_diff[1][i])
                else:
                    self.velocity[1].append(p_best_diff[1][i])

    def update_particle(self):
        max_conv = max(len(self.velocity[0]), len(self.particle.conv_layers))
        max_fc = max(len(self.velocity[1]), len(self.particle.fc_layers))
        self.velocity[1], self.particle.fc_layers = pad_layer_sequences(self.velocity[0], self.particle.conv_layers,
                                                                        self.velocity[1], self.particle.fc_layers,
                                                                        max_conv, max_fc)
        new_particle = ParticleModel()
        pool_count = 0
        for i in range(len(self.velocity[0])):
            if self.velocity[0][i] == 0:
                if type(self.particle.conv_layers[i]) in [MaxPooling2D, AveragePooling2D]:
                    if pool_count < 3:
                        new_particle.conv_layers.append(self.particle.conv_layers[i])
                        pool_count += 1
                    else:
                        continue
                elif self.particle.conv_layers[i] != 0:
                    new_particle.conv_layers.append(self.particle.conv_layers[i])
            elif self.velocity[0][i] == -1:
                continue
            else:
                if type(self.velocity[0][i]) in [MaxPooling2D, AveragePooling2D]:
                    if pool_count < 3:
                        new_particle.conv_layers.append(self.velocity[0][i])
                        pool_count += 1
                    else:
                        continue
                else:
                    new_particle.conv_layers.append(self.velocity[0][i])
        for i in range(len(self.velocity[1])):
            if self.velocity[1][i] == 0:
                if self.particle.fc_layers[i] != 0:
                    new_particle.fc_layers.append(self.particle.fc_layers[i])
                else:
                    continue
            elif self.velocity[1][i] == -1:
                continue
            else:
                new_particle.fc_layers.append(self.velocity[1][i])
        self.particle = new_particle


def pad_layer_sequences(p1_conv, p2_conv, p1_fc, p2_fc, max_conv, max_fc):
    p1_conv.extend([0] * (max_conv - len(p1_conv)))
    p2_conv.extend([0] * (max_conv - len(p2_conv)))
    p1_fc = [0] * (max_fc - len(p1_fc)) + p1_fc
    p2_fc = [0] * (max_fc - len(p2_fc)) + p2_fc
    return p1_fc, p2_fc


def compute_particle_difference(particle_1: ParticleModel, particle_2: ParticleModel):
    result = [[], []]
    particle_1_convs, particle_1_fc = list(particle_1.conv_layers), list(particle_1.fc_layers)
    particle_2_convs, particle_2_fc = list(particle_2.conv_layers), list(particle_2.fc_layers)

    convs_max = max(len(particle_1_convs), len(particle_2_convs))
    fc_max = max(len(particle_1_fc), len(particle_2_fc))
    particle_1_fc, particle_2_fc = pad_layer_sequences(particle_1_convs, particle_2_convs, particle_1_fc, particle_2_fc,
                                                       convs_max, fc_max)

    for i in range(convs_max):
        if type(particle_1_convs[i]) is type(particle_2_convs[i]):
            result[0].append(0)
        else:
            if particle_1_convs[i] != 0:
                result[0].append(particle_1_convs[i])
            else:
                result[0].append(-1)
    for i in range(fc_max):
        if type(particle_1_fc[i]) is type(particle_2_fc[i]):
            result[1].append(0)
        else:
            if particle_1_fc[i] != 0:
                result[1].append(particle_1_fc[i])
            else:
                result[1].append(-1)
    return result


def check_for_similarity(g_best_diff, p_best_diff):
    if len(g_best_diff[0]) == len(p_best_diff[0]) and len(g_best_diff[1]) == len(p_best_diff[1]):
        for i in range(len(g_best_diff[0])):
            if type(g_best_diff[0][i]) is not type(p_best_diff[0][i]):
                return False
            else:
                if type(g_best_diff[0][i]) is int and g_best_diff[0][i] != p_best_diff[0][i]:
                    return False
        for i in range(len(g_best_diff[1])):
            if type(g_best_diff[1][i]) is not type(p_best_diff[1][i]):
                return False
            else:
                if type(g_best_diff[1][i]) is int and g_best_diff[1][i] != p_best_diff[1][i]:
                    return False
    return True


def add_conv_layer(layers: list, input_layer=False):
    maps = np.random.randint(1, MAX_FEATURE_MAPS + 1)
    kernel_size = np.random.randint(3, MAX_KERNEL_SIZE + 1)
    if input_layer:
        layers.append(Conv2D(maps, (kernel_size, kernel_size), padding='same', input_shape=(28, 28, 1)))
    else:
        layers.append(Conv2D(maps, (kernel_size, kernel_size), padding='same'))


def add_pool_layer(layers: list):
    pool_type = np.random.randint(1, 3)
    if pool_type == 1:
        layers.append(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    else:
        layers.append(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))


def add_fc_layer(layers: list, count=1, final_layer=False):
    if final_layer:
        layers.append(Dense(OUTPUT_NEURONS, activation='softmax'))
    else:
        for _ in range(count):
            nr_neurons = np.random.randint(1, MAX_FC_NEURONS + 1)
            layers.append(Dense(nr_neurons, activation='relu'))


def init_swarm():
    swarm = list()
    for _ in range(SWARM_SIZE):
        depth = np.random.randint(3, MAX_LAYERS)
        current_particle = ParticleModel()
        added_pooling_layers = 0
        for i in range(depth):
            if i == 0:
                add_conv_layer(current_particle.conv_layers, input_layer=True)
            elif i == depth - 1:
                add_fc_layer(current_particle.fc_layers, final_layer=True)
            else:
                if added_pooling_layers < 3:
                    layer_type = np.random.randint(1, 4)
                else:
                    layer_type = np.random.randint(1, 3)
                if layer_type == 1:
                    add_conv_layer(current_particle.conv_layers)
                elif layer_type == 2:
                    add_fc_layer(current_particle.fc_layers, count=depth - i)
                    add_fc_layer(current_particle.fc_layers, final_layer=True)
                    break
                elif layer_type == 3:
                    added_pooling_layers += 1
                    add_pool_layer(current_particle.conv_layers)
        swarm.append(current_particle)
    return swarm


def build_model(conv_layers: list, fc_layers: list) -> Sequential:
    model = Sequential()
    for layer in conv_layers:
        layer_copy = type(layer).from_config(layer.get_config())
        model.add(layer_copy)
    model.add(Flatten())
    for layer in fc_layers:
        layer_copy = type(layer).from_config(layer.get_config())
        model.add(layer_copy)
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model


def compute_particle_loss(particle: ParticleModel, train_data, test_data):
    model = build_model(particle.conv_layers, particle.fc_layers)
    model.fit(train_data[0], train_data[1], batch_size=NN_BATCH_SIZE, epochs=NN_TRAIN_EPOCHS)
    loss, acc = model.evaluate(test_data[0], test_data[1], batch_size=NN_BATCH_SIZE)
    return loss


def validate_saved_particle(file_path):
    x_train, y_train, x_test, y_test = load_data()
    with open(file_path, 'rb') as f:
        particle = pkl.load(f)

    model = build_model(particle.conv_layers, particle.fc_layers)
    model.fit(x_train, y_train, batch_size=NN_BATCH_SIZE, epochs=NN_TEST_EPOCHS)

    train_loss, train_acc = model.evaluate(x_train, y_train)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Train loss: ' + str(train_loss) + '\n Train accuracy: ' + str(train_acc))
    print('Test loss: ' + str(test_loss) + '\n Test accuracy: ' + str(test_acc))


def load_data():
    train, test = mnist.load_data()
    x_train, y_train = train[0], train[1]
    x_test, y_test = test[0], test[1]
    x_train, x_test = x_train.reshape(x_train.shape[0], 28, 28, 1), x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def main(load=False):
    x_train, y_train, x_test, y_test = load_data()
    if not load:
        swarm = init_swarm()
        swarm[0].compute_loss((x_train, y_train), (x_test, y_test))
        particles = [ParticleValue(swarm[0])]
        g_best = particles[0].best_particle
        for i in range(1, SWARM_SIZE):
            swarm[i].compute_loss((x_train, y_train), (x_test, y_test))
            particles.append(ParticleValue(swarm[i]))
            if particles[-1].particle.loss < g_best.loss:
                g_best = particles[-1].best_particle
    else:
        with open('pso_checkpoint/particles.pkl', 'rb') as f, open('pso_checkpoint/best.pkl', 'rb') as g:
            particles = pkl.load(f)
            g_best = pkl.load(g)

    for it in range(PSO_ITER):
        with open('train_history.txt', 'a') as f:
            print(f'Iteration: {str(it)}', file=f)
            print(f'Best Loss: {str(g_best.loss)}', file=f)
        if (it + 1) % 10 == 0:
            with open('pso_checkpoint/particles.pkl', 'wb') as g, open('pso_checkpoint/best.pkl', 'wb') as h:
                pkl.dump(particles, g)
                pkl.dump(g_best, h)
        for value in particles:
            value.update_velocity(g_best)
            value.update_particle()
            value.particle.compute_loss((x_train, y_train), (x_test, y_test))
            if value.particle.loss < value.best_particle.loss:
                value.best_particle = value.particle
                if value.best_particle.loss < g_best.loss:
                    g_best = value.best_particle
    with open('best_particle.pkl', 'wb') as f:
        pkl.dump(g_best, f)


if __name__ == '__main__':
    main()
    # validate_saved_particle('best_particle.pkl')
