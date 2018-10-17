# Some potentially useful modules
# Whether or not you use these (or others) depends on your implementation!
# import random
import numpy as np
# import math
import matplotlib.pyplot as plt
# import time


class Layer(object):
    '''
        Layer class.
        Defines the individual layers in the network.
        Number of nodes can be different.
    '''

    def __init__(self, num_input, num_output, ):
        self.num_output = num_output
        self.num_input = num_input
        self.input = np.zeros([num_input])
        self.output = np.zeros([num_output])
        # self.num_neurons = num_neurons
        self.weights = np.random.randn(self.num_input, self.num_output)\
                       * 2 / (np.sqrt(self.num_input))   # [-1/sqrt(d) 1/sqrt(d)]
        self.thetas = np.random.randn(self.num_output) /5   # [-0.1 0.1]
        self.activation = NeuralMMAgent.sigmoid_af
        self.change_in_weights = self.weights.copy()
        self.change_in_thetas = self.thetas.copy()
        self.delta = np.zeros([self.num_output]) + 1             # initialize to 1


    def feed_forward(self):
        '''
            Feeds the input to activations.

        :return:
        '''
        self.output = self.activation(np.dot(self.input, self.weights) + self.thetas)

    def change_weights_thetas(self):
        '''
            Chanes weights and thetas according to the deltas calcultated.

        :return:
        '''
        self.weights += self.change_in_weights
        self.thetas += self.change_in_thetas

class NeuralMMAgent(object):
    '''
    Class to for Neural Net Agents
    '''

    def __init__(self, num_in_nodes, num_hid_nodes, num_hid_layers, num_out_nodes,
                 learning_rate=0.2, max_epoch=10000, max_sse=.01, momentum=0.2,
                 creation_function=None, activation_function=None, random_seed=1):
        '''
        Arguments:
            num_in_nodes -- total # of input layers for Neural Net
            num_hid_nodes -- total # of hidden nodes for each hidden layer
                in the Neural Net
            num_hid_layers -- total # of hidden layers for Neural Net
            num_out_nodes -- total # of output layers for Neural Net
            learning_rate -- learning rate to be used when propagating error
            creation_function -- function that will be used to create the
                neural network given the input
            activation_function -- list of two functions:
                1st function will be used by network to determine activation given a weighted summed input
                2nd function will be the derivative of the 1st function
            random_seed -- used to seed object random attribute.
                This ensures that we can reproduce results if wanted
        '''
        self.momentum = momentum
        self.max_sse = max_sse
        self.max_epoch = max_epoch
        assert num_in_nodes > 0 and num_hid_layers > 0 and num_hid_nodes and \
               num_out_nodes > 0, "Illegal number of input, hidden, or output layers!"

        self.x = np.zeros([num_in_nodes])
        self.is_x_set = False

        self.y = np.zeros([num_out_nodes])
        self.is_y_set = False
        self.num_in_nodes = num_in_nodes
        self.num_hid_nodes = num_hid_nodes
        self.num_hid_layers = num_hid_layers
        self.num_out_nodes = num_out_nodes
        self.layers = list()
        self.construct_net()
        self.learning_rate = learning_rate
        self.cost_function = lambda x: np.sqrt(np.sum(np.array(x)**2))
        self.error_list = list()

    def export_weights(self):
        weights = list()
        for layer in self.layers:
            weights.append(layer.weights)
        print(weights)
        return weights

    def export_thetas(self):
        thetas = list()
        for layer in self.layers:
            thetas.append(layer.thetas)
        return thetas

    def set_weights(self, weights):
        '''
        Set weights.
        :param weights:
        :return:
        '''

        for i, weight in enumerate(weights):
            weight = np.reshape(weight, np.shape(self.layers[i].weights))
            self.layers[i].weights = weight

    def construct_net(self):
        self.add_layer(self.num_in_nodes, self.num_hid_nodes)

        for i in range(self.num_hid_layers - 1):
            self.add_layer(self.num_hid_nodes, self.num_hid_nodes)

        self.add_layer(self.num_hid_nodes, self.num_out_nodes)

    def add_layer(self, num_in, num_out):
        '''
        Adds another layer to the network.
        :param num_in: the num of nodes of the previous layer
        :param num_out: the num of nodes of the current layer
        :return:
        '''
        self.layers.append(Layer(num_in, num_out))

    def get_x(self):
        assert self.is_x_set, "error, x not set!!"
        return self.x

    def get_y(self):
        assert self.is_y_set, "error, y not set!!"
        return self.y

    def set_x(self, x):
        assert np.shape(x) == np.shape(self.x), "shape does not match. {} vs {}"\
            .format(np.shape(x), np.shape(self.x))
        self.x = np.array(x)
        self.is_x_set = True

    def set_y(self, y):
        assert np.shape(y) == np.shape(self.y), "shape does not match. {} vs {}"\
            .format(np.shape(y), np.shape(self.y))
        self.y = np.array(y)
        self.is_y_set = True


    def feed_forward(self):
        activation = self.get_x()
        for layer in self.layers:
            layer.input = activation
            layer.feed_forward()
            activation = layer.output

        return self.layers[-1].output

    def accuracy(self):
        results = np.array([])
        for i, line in enumerate(self.input_list):
            self.set_x(self.input_list[i])
            self.set_y(self.output_list[i])
            logits = self.feed_forward()
            r = np.array([1 if x > 0.5 else 0 for x in logits])
            #print(r)
            results = np.append(results, r)
        # print("results: {}, desired: {}".format(results, self.output_list))
        return np.sum(np.equal(results, np.array(self.output_list).flatten())) / len(results)

    def plot_loss(self, title):
        '''
        Plots loss vs epoch
        :return: Null
        '''

        plt.plot(self.error_list, label = title)
        plt.ylabel("sse")
        plt.xlabel("num of epochs")
        plt.legend()
        plt.show()

    def train_net(self, input_list, output_list, max_num_epoch=100000,
                  max_sse=0.1):
        ''' Trains neural net using incremental learning
            (update once per input-output pair)
            Arguments:
                input_list -- 2D list of inputs
                output_list -- 2D list of outputs matching inputs
        '''

        '''
            all_err.append(total_err)
    
            if (total_err < max_sse):
                break
                #Show us how our error has changed
            plt.plot(all_err)
            plt.show()
        '''
        self.input_list = input_list
        self.output_list = output_list

        for epoch in range(self.max_epoch):
            sse = 0

            for i, line in enumerate(input_list):
                self.set_x(line)
                self.set_y(output_list[i])
                self.feed_forward()
                sse += self._calculate_deltas() ** 2
                self._adjust_weights_thetas()
                if epoch % 100 == 0:
                    print("Input: {}, Output: {}".format(self.get_x(), self.feed_forward()))
                #self.print_weights_thetas()
            print("Epoch: {}, Cost: {}, Accuracy: {}".format(epoch, np.sqrt(sse), self.accuracy()))

            if sse < self.max_sse:
                break
            #time.sleep(0.3)
            self.error_list.append(sse)




    def _calculate_deltas(self):
        '''Used to calculate all weight deltas for our neural net
            Arguments:
                out_error -- output error (typically SSE), obtained using target
                    output and actual output
        '''

        '''
            Find change in thetas first
        '''

        # Calculate error gradient for each output node & propgate error
        #   (calculate weight deltas going backward from output_nodes)
        assert self.is_y_set and self.is_x_set, "Please set input/ output."
        last_layer = self.layers[-1]
        e = self.get_y() - last_layer.output
        last_delta = NeuralMMAgent.sigmoid_af_deriv(last_layer.output) * e
        for i in range(len(self.layers))[::-1]:
            # momentum
            self.layers[i].change_in_weights = \
                self.layers[i].change_in_weights * self.momentum + \
                self.learning_rate * np.outer(self.layers[i].input, last_delta)
            self.layers[i].delta = NeuralMMAgent.sigmoid_af_deriv(self.layers[i].input) * \
                np.dot(self.layers[i].weights, last_delta)
            self.layers[i].change_in_thetas = self.learning_rate * last_delta
            last_delta = self.layers[i].delta

        return self.cost_function(e)

    def _adjust_weights_thetas(self):
        '''Used to apply deltas
        '''
        for layer in self.layers:
            layer.change_weights_thetas()


    @staticmethod
    def create_neural_structure(num_in, num_hid, num_hid_layers, num_out, rand_obj):
        ''' Creates the structures needed for a simple backprop neural net
        This method creates random weights [-0.5, 0.5]
        Arguments:
            num_in -- total # of input layers for Neural Net
            num_hid -- total # of hidden nodes for each hidden layer
                in the Neural Net
            num_hid_layers -- total # of hidden layers for Neural Net
            num_out -- total # of output layers for Neural Net
            rand_obj -- the random object that will be used to selecting
                random weights
        Outputs:
            Tuple w/ the following items
                1st - 2D list of initial weights
                2nd - 2D list for weight deltas
                3rd - 2D list for activations
                4th - 2D list for errors
                5th - 2D list of thetas for threshold
                6th - 2D list for thetas deltas
        '''


    # -----Begin ACCESSORS-----#
    # -----End ACCESSORS-----#

    @staticmethod
    def sigmoid_af(summed_input):
        return 1 / (1 + (np.exp(-summed_input)))
        # Sigmoid function

    @staticmethod
    def sigmoid_af_deriv(sig_output):
        return sig_output * (1 - sig_output)

    def set_thetas(self, thetas):
        for i, theta in enumerate(thetas):
            self.layers[i].bias = theta

    def print_weights_thetas_deltas(self):
        print(" ----------- weights -------------- ")
        for i, layer in enumerate(self.layers):
            print("layer {}: {}".format(i, layer.weights))
        print(" ----------- thetas --------------- ")
        for i, layer in enumerate(self.layers):
            print("layer {}: {}".format(i, layer.thetas))
        print(" ----------- change in weights----- ")
        for i, layer in enumerate(self.layers):
            print("layer {}: {}".format(i, layer.change_in_weights))
        print(" ----------- change in thetas ----- ")
        for i, layer in enumerate(self.layers):
            print("layer {}: {}".format(i, layer.change_in_thetas))
        print(" ----------- delta ----- ")
        for i, layer in enumerate(self.layers):
            print("layer {}: {}".format(i, layer.delta))

#test_agent.train_net(input_list=test_in, output_list=test_out, )

'''
    Assign example weights.
'''

#test_agent.set_weights([[-.37, .26, .10, -0.24], [-0.01, -0.05]])
#test_agent.set_thetas([[.0, .0], [.0]])


'''
# test the first iteration. 
#test_agent.set_y([1])
#test_agent.set_x([1, 0])
#test_agent.feed_foward()
#test_agent._calculate_deltas()
'''

'''
Test XOR
'''
print("----------------------XOR-------------------------")
xor_agent = NeuralMMAgent(2, 2, 1, 1, random_seed=5, max_epoch=4000,
                          learning_rate=0.2, momentum=0)

xor_agent_m = NeuralMMAgent(2, 2, 1, 1, random_seed=5, max_epoch=4000,
                          learning_rate=0.2, momentum=0.1)
#print(xor_agent.export_weights())
xor_agent_m.set_weights(xor_agent.export_weights())
xor_agent_m.set_thetas(xor_agent.export_thetas())

test_in = [[1, 0], [0, 0], [1, 1], [0, 1]]
test_out = [[1], [0], [0], [1]]
xor_agent.train_net(test_in, test_out, max_sse = xor_agent.max_sse,
                    max_num_epoch = xor_agent.max_epoch)
xor_agent.print_weights_thetas_deltas()
xor_agent.plot_loss("XOR")
'''
xor momentum
'''

print("----------------------XOR_MOMENTUM-------------------------")
test_in = [[1, 0], [0, 0], [1, 1], [0, 1]]
test_out = [[1], [0], [0], [1]]
xor_agent_m.train_net(test_in, test_out, max_sse = xor_agent_m.max_sse,
                    max_num_epoch = xor_agent_m.max_epoch)
xor_agent_m.print_weights_thetas_deltas()
xor_agent_m.plot_loss("XOR_momentum")
'''
Test AND
'''

print("----------------------AND-------------------------")
and_agent = NeuralMMAgent(2, 2, 1, 1, random_seed=5, max_epoch=5000,
                          learning_rate=0.2, momentum=0)
test_in = [[1, 0], [0, 0], [1, 1], [0, 1]]
test_out = [[0], [0], [1], [0]]
and_agent.train_net(test_in, test_out, max_sse = and_agent.max_sse,
                    max_num_epoch = xor_agent.max_epoch)
and_agent.print_weights_thetas_deltas()
and_agent.plot_loss("AND")

'''
Test OR
'''

print("----------------------OR-------------------------")
or_agent = NeuralMMAgent(2, 2, 1, 1, random_seed=5, max_epoch=5000,
                          learning_rate=0.2, momentum=0)
test_in = [[1, 0], [0, 0], [1, 1], [0, 1]]
test_out = [[1], [0], [1], [1]]
or_agent.train_net(test_in, test_out, max_sse = or_agent.max_sse,
                    max_num_epoch = xor_agent.max_epoch)
or_agent.print_weights_thetas_deltas()
or_agent.plot_loss("OR")
