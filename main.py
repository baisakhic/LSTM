import argparse
import datetime
import math
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

root_dir = os.path.dirname(os.path.abspath(__file__))
charAt = root_dir.rfind('\\')
root_dir = root_dir[0:charAt]
np.random.seed(10)

input_nodes = 3
hidden_layers = 0
nodes_per_hidden_layer = 0
output_nodes = 1
labelled_outputs = []
outputs = []  # short term memory
states = []  # long term memory
input_activations = []
input_gates = []
output_gates = []
forget_gates = []

network = []
training_path = root_dir + "/LSTM/DelhiWeatherData/DailyDelhiClimateTrain.csv"
# training_path = root_dir + "/LSTM/train.csv"
testing_path = root_dir + "/LSTM/DelhiWeatherData/DailyDelhiClimateTest.csv"
# testing_path = root_dir + "/LSTM/test.csv"


weights_input_activation = []
weights_input_gate = []
weights_output_gate = []
weights_forget_gate = []

weights_input_activation_previous = []
weights_input_gate_previous = []
weights_output_gate_previous = []
weights_forget_gate_previous = []

bias_input_activation = []
bias_input_gate = []
bias_output_gate = []
bias_forget_gate = []

training_mode = True
sorting_required = False

if 'DailyDelhiClimateTrain' in training_path:
    sorting_required = True

learning_rate = 0.05

data_min = None
data_max = None


def normalize(data, index):
    return (data - data_min[index]) / (data_max[index] - data_min[index])


def de_normalize(data, index):
    return data * (data_max[index] - data_min[index]) + data_min[index]


def random_val():
    return np.random.rand()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    # return x


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
    # return 1


def tanh(x):
    return math.tanh(x)


def tanh_derivative(x):
    return 1 - pow(tanh(x), 2)


def matrix_add(A, B):
    if A is None:
        return B

    if B is None:
        return A

    np_A = np.array(A)
    np_B = np.array(B)

    return np.add(np_A, np_B).tolist()


def mean_squared_error(y_pred, y_true):
    error = []
    for val_pred in y_pred:
        val_true = y_true[y_pred.index(val_pred)]
        minus = val_pred - val_true
        error.append(pow(minus, 2) * 0.5)
    return error


def error_derivative(y_pred, y_true):
    error = []
    for val_pred in y_pred:
        val_true = y_true[y_pred.index(val_pred)]
        minus = val_pred - val_true
        error.append(minus)
    return error


def accuracy(y_pred, y_true):
    sum = 0
    for val_pred in y_pred:
        val_true = y_true[y_pred.index(val_pred)]
        minus = val_pred - val_true
        sum = sum + (1 - abs(minus)) * 100
    acc = sum / len(y_pred)
    return acc


def read_file(filename):
    header = True
    first = False
    add_data = False
    rows = []

    global data_min, data_max

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row_val = []
            if header:
                header = False
                continue
            if sorting_required:
                first = True
            if training_mode and data_min is None:
                data_min = []
                data_max = []
                add_data = True
            for cell in row:
                if not first:
                    row_val.append(float(cell))
                    if add_data:
                        data_min.append(float(cell))
                        data_max.append(float(cell))
                    else:
                        if float(cell) > data_max[row.index(cell)]:
                            data_max[row.index(cell)] = float(cell)
                        if float(cell) < data_min[row.index(cell)]:
                            data_min[row.index(cell)] = float(cell)
                else:
                    if add_data:
                        data_min.append(math.nan)
                        data_max.append(math.nan)
                    row_val.append(datetime.datetime.strptime(cell, '%Y-%m-%d'))
                    first = False
            rows.append(row_val)
            add_data = False
            # rows.append(row)

    column_len = input_nodes + output_nodes

    if sorting_required:
        rows.sort(key=lambda x: x[0])
        column_len = column_len + 1

    # normalized_rows = rows
    normalized_rows = []
    for row in rows:
        first_col = True
        normalized_row = []
        for cell in row:
            if sorting_required and first_col:
                first_col = False
                normalized_row.append(cell)
            else:
                normalized_row.append(normalize(cell, row.index(cell)))
        normalized_rows.append(normalized_row)

    # if len(rows[0]) != column_len:
    #     raise Exception("Incorrect Size of Data")

    return normalized_rows


def plot_network():
    k = 1
    for layer in weights_input_activation:
        n = 1
        for node in layer:
            plt.scatter(k, n, c='k')
            w = 1
            for weight in node:
                plt.plot([k, k + 1], [n, w], c='b')
                w = w + 1
            n = n + 1
        k = k + 1

    for i in range(output_nodes):
        plt.scatter(k, i + 1, c='k')
    plt.show()


def feed_forward_network(t, input_val):
    output_vals_t = []
    states_t = []
    input_activation_t = []
    input_gate_t = []
    output_gate_t = []
    forget_gate_t = []

    output_val = 0


    prev_layer = input_val
    output_vals_t.append(prev_layer.copy())
    layer_no = 0

    for no_of_nodes_in_layer in network[1:]:
        current_layer_output = []
        current_layer_state = []
        current_layer_activation = []
        current_layer_input_gate = []
        current_layer_output_gate = []
        current_layer_forget_gate = []

        for node in range(no_of_nodes_in_layer):
            previous_state = 0
            if t > 0:
                previous_state = states[t - 1][layer_no][node]
            sum_val_state = 0
            sum_val_output = 0
            sum_val_output_gate = 0
            sum_val_input_gate = 0
            sum_val_forget_gate = 0
            sum_val_activation = 0
            for node_in_prev_layer in range(len(prev_layer)):
                if t != 0:
                    output_val = outputs[t - 1][layer_no+1][node]

                prev_val = prev_layer[node_in_prev_layer]

                input_act_val = weights_input_activation[layer_no][node][node_in_prev_layer] * prev_val
                input_gate_val = weights_input_gate[layer_no][node][node_in_prev_layer] * prev_val
                output_gate_val = weights_output_gate[layer_no][node][node_in_prev_layer] * prev_val
                forget_gate_val = weights_forget_gate[layer_no][node][node_in_prev_layer] * prev_val

                # state_val = input_act_val * input_gate_val + forget_gate_val * previous_state
                # output_val = tanh(state_val) * output_gate_val

                # sum_val_state += state_val
                # sum_val_output += output_val
                sum_val_activation += input_act_val
                sum_val_input_gate += input_gate_val
                sum_val_output_gate += output_gate_val
                sum_val_forget_gate += forget_gate_val

            input_act_val_final = tanh(sum_val_activation + weights_input_activation_previous[layer_no][node] * \
                                       output_val + bias_input_activation[layer_no][node])
            sum_val_input_gate_final = sigmoid(sum_val_input_gate + weights_input_gate_previous[layer_no][node] * \
                                               output_val + bias_input_gate[layer_no][node])
            sum_val_output_gate_final = sigmoid(sum_val_output_gate + weights_output_gate_previous[layer_no][node] * \
                                                output_val + bias_output_gate[layer_no][node])
            sum_val_forget_gate_final = sigmoid(sum_val_forget_gate + weights_forget_gate_previous[layer_no][node] * \
                                                output_val + bias_forget_gate[layer_no][node])

            current_layer_activation.append(input_act_val_final)
            current_layer_input_gate.append(sum_val_input_gate_final)
            current_layer_output_gate.append(sum_val_output_gate_final)
            current_layer_forget_gate.append(sum_val_forget_gate_final)

            sum_val_state = input_act_val_final * sum_val_input_gate_final + sum_val_forget_gate_final * previous_state
            sum_val_output = tanh(sum_val_state) * sum_val_output_gate_final

            current_layer_state.append(sum_val_state)
            current_layer_output.append(sum_val_output)

            prev_layer = current_layer_output

        output_vals_t.append(prev_layer)
        states_t.append(current_layer_state)
        input_activation_t.append(current_layer_activation)
        input_gate_t.append(current_layer_input_gate)
        output_gate_t.append(current_layer_output_gate)
        forget_gate_t.append(current_layer_forget_gate)

        layer_no = layer_no + 1

    outputs.append(output_vals_t)
    states.append(states_t)
    input_activations.append(input_activation_t)
    input_gates.append(input_gate_t)
    output_gates.append(output_gate_t)
    forget_gates.append(forget_gate_t)

    # return output_vals_t


def calculate_delta_val():
    previous_delta_state = 0
    previous_big_delta_out = 0

    previous_delta_a_state = 0
    previous_delta_input_state = 0
    previous_delta_forget_state = 0
    previous_delta_output_state = 0

    delta_weights_input_activation_sum = None
    delta_weights_input_gate_sum = None
    delta_weights_output_gate_sum = None
    delta_weights_forget_gate_sum = None

    delta_weights_input_activation_sum_previous = None
    delta_weights_input_gate_sum_previous = None
    delta_weights_output_gate_sum_previous = None
    delta_weights_forget_gate_sum_previous = None

    delta_bias_input_activation_sum = None
    delta_bias_input_gate_sum = None
    delta_bias_output_gate_sum = None
    delta_bias_forget_gate_sum = None

    for t in range(len(labelled_outputs) - 1, -1, -1):
        labelled_output = labelled_outputs[t]
        predicted_output = outputs[t][-1]
        error_val = error_derivative(predicted_output, labelled_output)
        delta_weights = []

        last_layer = True

        delta_weights_input_activation_sum_per_layer = []
        delta_weights_input_gate_sum_per_layer = []
        delta_weights_output_gate_sum_per_layer = []
        delta_weights_forget_gate_sum_per_layer = []

        delta_weights_input_activation_sum_previous_per_layer = []
        delta_weights_input_gate_sum_previous_per_layer = []
        delta_weights_output_gate_sum_previous_per_layer = []
        delta_weights_forget_gate_sum_previous_per_layer = []

        delta_bias_input_activation_sum_per_layer = []
        delta_bias_input_gate_sum_per_layer = []
        delta_bias_output_gate_sum_per_layer = []
        delta_bias_forget_gate_sum_per_layer = []

        for layer_no, no_of_nodes in reversed(list(enumerate(network))):
            if layer_no == 0:
                continue
            delta_weights_per_layer = []

            practical_layer = layer_no - 1

            delta_weights_input_activation_sum_per_node = []
            delta_weights_input_gate_sum_per_node = []
            delta_weights_output_gate_sum_per_node = []
            delta_weights_forget_gate_sum_per_node = []

            delta_weights_input_activation_sum_previous_per_node = []
            delta_weights_input_gate_sum_previous_per_node = []
            delta_weights_output_gate_sum_previous_per_node = []
            delta_weights_forget_gate_sum_previous_per_node = []

            delta_bias_input_activation_sum_per_node = []
            delta_bias_input_gate_sum_per_node = []
            delta_bias_output_gate_sum_per_node = []
            delta_bias_forget_gate_sum_per_node = []

            for node in range(no_of_nodes):
                delta_val = 0
                val_node_current_layer = outputs[t][layer_no][node]
                if last_layer:
                    delta_val = error_val[0]
                else:
                    sum_val = 0
                    for node_next_layer in range(network[layer_no + 1]):
                        delta_val_node_next_layer = delta_weights[layer_no+1][node_next_layer][node]
                        weight = weights_input_activation[layer_no+1][node_next_layer][node]
                        sum_val += delta_val_node_next_layer * weight
                    delta_val = sum_val

                delta_weights_per_layer.append(delta_val * tanh_derivative(val_node_current_layer))
                delta_val += previous_big_delta_out

                f_gate_plus_one = 0
                state_minus_one = 0
                if t != len(labelled_outputs) - 1:
                    f_gate_plus_one = forget_gates[t + 1][practical_layer][node]

                if t > 0:
                    state_minus_one = states[t-1][practical_layer][node]

                delta_state = delta_val * output_gates[t][practical_layer][node] * \
                            tanh_derivative(states[t][practical_layer][node]) + previous_delta_state * \
                            f_gate_plus_one

                delta_a = delta_state * input_gates[t][practical_layer][node] * \
                            (1 - pow(input_activations[t][practical_layer][node], 2))

                delta_input = delta_state * input_gates[t][practical_layer][node] * \
                              input_activations[t][practical_layer][node] * \
                              (1 - input_gates[t][practical_layer][node])

                delta_forget = delta_state * state_minus_one * \
                               forget_gates[t][practical_layer][node] * \
                               (1 - forget_gates[t][practical_layer][node])

                delta_output = delta_val * tanh(states[t][practical_layer][node]) * output_gates[t][practical_layer][node] * \
                               (1 - output_gates[t][practical_layer][node])

                previous_big_delta_out = weights_input_activation_previous[practical_layer][node] * delta_a + \
                                         weights_input_gate_previous[practical_layer][node] * delta_input + \
                                         weights_output_gate_previous[practical_layer][node] * delta_output + \
                                         weights_forget_gate_previous[practical_layer][node] * delta_forget

                delta_weights_input_activation_sum_per_pre_node = []
                delta_weights_input_gate_sum_per_pre_node = []
                delta_weights_output_gate_sum_per_pre_node = []
                delta_weights_forget_gate_sum_per_pre_node = []

                for node_previous_layer in range(network[layer_no - 1]):
                    val = outputs[t][layer_no-1][node_previous_layer]
                    delta_weights_input_activation_sum_per_pre_node.append(delta_a * val)
                    delta_weights_input_gate_sum_per_pre_node.append(delta_input * val)
                    delta_weights_output_gate_sum_per_pre_node.append(delta_output * val)
                    delta_weights_forget_gate_sum_per_pre_node.append(delta_forget * val)

                delta_weights_input_activation_sum_per_node.append(delta_weights_input_activation_sum_per_pre_node)
                delta_weights_input_gate_sum_per_node.append(delta_weights_input_gate_sum_per_pre_node)
                delta_weights_output_gate_sum_per_node.append(delta_weights_output_gate_sum_per_pre_node)
                delta_weights_forget_gate_sum_per_node.append(delta_weights_forget_gate_sum_per_pre_node)

                delta_weights_input_activation_sum_previous_per_node.append(outputs[t][layer_no][node] * previous_delta_a_state)
                delta_weights_input_gate_sum_previous_per_node.append(outputs[t][layer_no][node] * previous_delta_input_state)
                delta_weights_output_gate_sum_previous_per_node.append(outputs[t][layer_no][node] * previous_delta_output_state)
                delta_weights_forget_gate_sum_previous_per_node.append(outputs[t][layer_no][node] * previous_delta_forget_state)

                delta_bias_input_activation_sum_per_node.append(delta_a)
                delta_bias_input_gate_sum_per_node.append(delta_input)
                delta_bias_output_gate_sum_per_node.append(delta_output)
                delta_bias_forget_gate_sum_per_node.append(delta_forget)

                previous_delta_state = delta_state
                previous_delta_a_state = delta_a
                previous_delta_input_state = delta_input
                previous_delta_forget_state = delta_forget
                previous_delta_output_state = delta_output

            if last_layer:
                last_layer = False

            delta_weights.append(delta_weights_per_layer)

            delta_weights_input_activation_sum_per_layer.append(delta_weights_input_activation_sum_per_node)
            delta_weights_input_gate_sum_per_layer.append(delta_weights_input_gate_sum_per_node)
            delta_weights_output_gate_sum_per_layer.append(delta_weights_output_gate_sum_per_node)
            delta_weights_forget_gate_sum_per_layer.append(delta_weights_forget_gate_sum_per_node)

            delta_weights_input_activation_sum_previous_per_layer.append(delta_weights_input_activation_sum_previous_per_node)
            delta_weights_input_gate_sum_previous_per_layer.append(delta_weights_input_gate_sum_previous_per_node)
            delta_weights_output_gate_sum_previous_per_layer.append(delta_weights_output_gate_sum_previous_per_node)
            delta_weights_forget_gate_sum_previous_per_layer.append(delta_weights_forget_gate_sum_previous_per_node)

            delta_bias_input_activation_sum_per_layer.append(delta_bias_input_activation_sum_per_node)
            delta_bias_input_gate_sum_per_layer.append(delta_bias_input_gate_sum_per_node)
            delta_bias_output_gate_sum_per_layer.append(delta_bias_output_gate_sum_per_node)
            delta_bias_forget_gate_sum_per_layer.append(delta_bias_forget_gate_sum_per_node)

        # Matrix Addition
        delta_weights_input_activation_sum = matrix_add(delta_weights_input_activation_sum, delta_weights_input_activation_sum_per_layer)
        delta_weights_input_gate_sum = matrix_add(delta_weights_input_gate_sum, delta_weights_input_gate_sum_per_layer)
        delta_weights_output_gate_sum = matrix_add(delta_weights_output_gate_sum, delta_weights_output_gate_sum_per_layer)
        delta_weights_forget_gate_sum = matrix_add(delta_weights_forget_gate_sum, delta_weights_forget_gate_sum_per_layer)

        delta_weights_input_activation_sum_previous = matrix_add(delta_weights_input_activation_sum_previous, delta_weights_input_activation_sum_previous_per_layer)
        delta_weights_input_gate_sum_previous = matrix_add(delta_weights_input_gate_sum_previous, delta_weights_input_gate_sum_previous_per_layer)
        delta_weights_output_gate_sum_previous = matrix_add(delta_weights_output_gate_sum_previous, delta_weights_output_gate_sum_previous_per_layer)
        delta_weights_forget_gate_sum_previous = matrix_add(delta_weights_forget_gate_sum_previous, delta_weights_forget_gate_sum_previous_per_layer)

        delta_bias_input_activation_sum = matrix_add(delta_bias_input_activation_sum, delta_bias_input_activation_sum_per_layer)
        delta_bias_input_gate_sum = matrix_add(delta_bias_input_gate_sum, delta_bias_input_gate_sum_per_layer)
        delta_bias_output_gate_sum = matrix_add(delta_bias_output_gate_sum, delta_bias_output_gate_sum_per_layer)
        delta_bias_forget_gate_sum = matrix_add(delta_bias_forget_gate_sum, delta_bias_forget_gate_sum_per_layer)

    adjust_weights_and_biases(delta_weights_input_activation_sum, delta_weights_input_gate_sum, delta_weights_output_gate_sum,
                              delta_weights_forget_gate_sum, delta_weights_input_activation_sum_previous, delta_weights_input_gate_sum_previous,
                              delta_weights_output_gate_sum_previous, delta_weights_forget_gate_sum_previous, delta_bias_input_activation_sum,
                              delta_bias_input_gate_sum, delta_bias_output_gate_sum, delta_bias_forget_gate_sum)
    #return delta_weights, delta_biases


def adjust_weights_and_biases(delta_weights_input_activation_sum, delta_weights_input_gate_sum, delta_weights_output_gate_sum,
                              delta_weights_forget_gate_sum, delta_weights_input_activation_sum_previous, delta_weights_input_gate_sum_previous,
                              delta_weights_output_gate_sum_previous, delta_weights_forget_gate_sum_previous, delta_bias_input_activation_sum,
                              delta_bias_input_gate_sum, delta_bias_output_gate_sum, delta_bias_forget_gate_sum):
    for layer in network[1:]:
        practical_layer = layer - 1
        for node in range(layer):
            for previous_node in range(network[network.index(layer) - 1]):
                weights_input_activation[practical_layer][node][previous_node] += learning_rate * delta_weights_input_activation_sum[practical_layer][node][previous_node]
                weights_input_gate[practical_layer][node][previous_node] += learning_rate * delta_weights_input_gate_sum[practical_layer][node][previous_node]
                weights_output_gate[practical_layer][node][previous_node] += learning_rate * delta_weights_output_gate_sum[practical_layer][node][previous_node]
                weights_forget_gate[practical_layer][node][previous_node] += learning_rate * delta_weights_forget_gate_sum[practical_layer][node][previous_node]

            weights_input_activation_previous[practical_layer][node] += learning_rate * delta_weights_input_activation_sum_previous[practical_layer][node]
            weights_input_gate_previous[practical_layer][node] += learning_rate * delta_weights_input_gate_sum_previous[practical_layer][node]
            weights_output_gate_previous[practical_layer][node] += learning_rate * delta_weights_output_gate_sum_previous[practical_layer][node]
            weights_forget_gate_previous[practical_layer][node] += learning_rate * delta_weights_forget_gate_sum_previous[practical_layer][node]

            bias_input_activation[practical_layer][node] += learning_rate * delta_bias_input_activation_sum[practical_layer][node]
            bias_input_gate[practical_layer][node] += learning_rate * delta_bias_input_gate_sum[practical_layer][node]
            bias_output_gate[practical_layer][node] += learning_rate * delta_bias_output_gate_sum[practical_layer][node]
            bias_forget_gate[practical_layer][node] += learning_rate * delta_bias_forget_gate_sum[practical_layer][node]


def test_data(t):
    testing_data = read_file(testing_path)
    input_val = []
    output_val = []

    start = 0
    if sorting_required:
        start = 1

    global training_mode
    training_mode = False

    for row in testing_data:
        input_val.clear()
        output_val.clear()

        for key in range(start, input_nodes + start):
            input_val.append(row[key])

        for key in range(start + input_nodes, output_nodes + input_nodes + start):
            output_val.append(row[key])

        feed_forward_network(t, input_val)
        acc = accuracy(outputs[testing_data.index(row)][-1], output_val)
        print(" Output : " + str(outputs[testing_data.index(row)][-1]) + " :: Accuracy : " + str(round(acc, 2)) + "%")


def train_data():
    training_data = read_file(training_path)
    input_val = []
    output_val = []
    start = 0
    if sorting_required:
        start = 1

    for epoch in range(2):
        outputs.clear()
        states.clear()
        labelled_outputs.clear()
        for row in training_data:
            input_val.clear()
            output_val.clear()
            for key in range(start, input_nodes + start):
                input_val.append(row[key])

            for key in range(start + input_nodes, output_nodes + input_nodes + start):
                output_val.append(row[key])

            feed_forward_network(training_data.index(row), input_val)
            labelled_outputs.append(output_val.copy())
            acc = accuracy(outputs[training_data.index(row)][-1], output_val)
            print("Epoch : " + str(epoch) + " Output : " + str(outputs[training_data.index(row)][-1]) + " :: Accuracy : " + str(round(acc, 2)) + "%")

        calculate_delta_val()

    return len(training_data)


def initialize_weights_and_biases():
    global weights_input_activation
    global weights_input_gate
    global weights_output_gate
    global weights_forget_gate
    global weights_input_activation_previous
    global weights_input_gate_previous
    global weights_output_gate_previous
    global weights_forget_gate_previous
    global bias_input_activation
    global bias_input_gate
    global bias_output_gate
    global bias_forget_gate

    network.append(input_nodes)
    for i in range(hidden_layers):
        network.append(nodes_per_hidden_layer)
    network.append(output_nodes)

    for layer in range(1, len(network)):
        weights_input_activation_per_layer = []
        weights_input_gate_per_layer = []
        weights_output_gate_per_layer = []
        weights_forget_gate_per_layer = []

        weights_input_gate_per_layer_previous = []
        weights_output_gate_per_layer_previous = []
        weights_forget_gate_per_layer_previous = []
        weights_input_activation_per_layer_previous = []

        bias_input_gate_per_layer = []
        bias_output_gate_per_layer = []
        bias_forget_gate_per_layer = []
        bias_input_activation_per_layer = []

        for node in range(network[layer]):
            weights_input_activation_per_node = []
            weights_input_gate_per_node = []
            weights_output_gate_per_node = []
            weights_forget_gate_per_node = []

            weights_input_gate_per_layer_previous.append(random_val())
            weights_output_gate_per_layer_previous.append(random_val())
            weights_forget_gate_per_layer_previous.append(random_val())
            weights_input_activation_per_layer_previous.append(random_val())

            bias_input_gate_per_layer.append(random_val())
            bias_output_gate_per_layer.append(random_val())
            bias_forget_gate_per_layer.append(random_val())
            bias_input_activation_per_layer.append(random_val())

            for previous_node in range(network[layer - 1]):
                weights_input_activation_per_node.append(random_val())
                weights_input_gate_per_node.append(random_val())
                weights_output_gate_per_node.append(random_val())
                weights_forget_gate_per_node.append(random_val())

            weights_input_activation_per_layer.append(weights_input_activation_per_node)
            weights_input_gate_per_layer.append(weights_input_gate_per_node)
            weights_output_gate_per_layer.append(weights_output_gate_per_node)
            weights_forget_gate_per_layer.append(weights_forget_gate_per_node)

        weights_input_activation.append(weights_input_activation_per_layer)
        weights_input_gate.append(weights_input_gate_per_layer)
        weights_output_gate.append(weights_output_gate_per_layer)
        weights_forget_gate.append(weights_forget_gate_per_layer)

        weights_input_activation_previous.append(weights_input_activation_per_layer_previous)
        weights_input_gate_previous.append(weights_input_gate_per_layer_previous)
        weights_output_gate_previous.append(weights_output_gate_per_layer_previous)
        weights_forget_gate_previous.append(weights_forget_gate_per_layer_previous)

        bias_input_activation.append(bias_input_activation_per_layer)
        bias_input_gate.append(bias_input_gate_per_layer)
        bias_output_gate.append(bias_output_gate_per_layer)
        bias_forget_gate.append(bias_forget_gate_per_layer)


    # weights_input_activation = [[[0.45, 0.25]]]
    # weights_input_gate = [[[0.95, 0.8]]]
    # weights_output_gate = [[[0.6, 0.4]]]
    # weights_forget_gate = [[[0.7, 0.45]]]
    #
    # weights_input_activation_previous = [[0.15]]
    # weights_input_gate_previous = [[0.8]]
    # weights_output_gate_previous = [[0.25]]
    # weights_forget_gate_previous = [[0.1]]
    #
    # bias_input_activation = [[0.2]]
    # bias_input_gate = [[0.65]]
    # bias_output_gate = [[0.1]]
    # bias_forget_gate = [[0.15]]


def initialize():
    initialize_weights_and_biases()
    # output_arr = []
    # for i in range(output_nodes):
    #     output_arr.append(0)
    # outputs.append(output_arr)
    t = train_data()
    test_data(t)
    # feed_forward_network([10,20,30,40,50])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_nodes", help="No of Input Nodes", type=int)
    parser.add_argument("-o", "--output_nodes", help="No of Output Nodes", type=int)
    parser.add_argument("-l", "--layers", help="No of Hidden Layers", type=int)
    parser.add_argument("-n", "--hidden_layer_nodes", help="No of Nodes in each Hidden Layer", type=int)
    parser.add_argument("-t", "--train", help="Path For Training Data")
    parser.add_argument("-e", "--test", help="Path For Testing Data")
    parser.add_argument("-s", "--sorting_required", type=bool,
                        help="Indicate if sorting is required. If yes, first column will be used")

    args = parser.parse_args()

    if args.input_nodes:
        input_nodes = args.input_nodes

    if args.output_nodes:
        output_nodes = args.output_nodes

    if args.layers:
        hidden_layers = args.layers

    if args.hidden_layer_nodes:
        nodes_per_hidden_layer = args.hidden_layer_nodes

    if args.train:
        training_path = args.train

    if args.test:
        testing_path = args.test

    if args.sorting_required:
        sorting_required = args.sorting_required

    initialize()
