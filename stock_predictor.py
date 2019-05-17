#! /usr/bin/env python
import sys
import pandas
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
print (sys.executable)
import argparse
from datetime import datetime, timedelta
from itertools import compress, islice, chain
import numpy
# import mpl_finance as finance
from matplotlib import pyplot as plt, mlab
from network import InputLayer, HiddenLayer, OutputLayer, NeuralNetwork
from animate import AnimatePlots
from fetch import Fetcher


def train(nn, data):
    nn.train(data)
    return nn


def get_animated_training(dates=(), background=False):
    def animate_training(nn, training_set):
        animated_plot = AnimatePlots(background)
        norm_input_set, norm_output_set = nn.process_training_set(training_set)
        input_set = nn.input_normalization.de_normalize(norm_input_set)
        animated_plot.update(nn.output_normalization.de_normalize(norm_output_set), 0, -1)  # original data
        for epoch, rel_error in enumerate(nn.itrain(training_set), 1):
            animated_plot.update(map(nn.feed, input_set), epoch, rel_error)
        animated_plot.stop()
        return nn
    return animate_training


def get_trained_stock_neural_network(
        stock_prices_training_set,
        hidden_neuron_count=50,
        training_func=None,
        threshold=.001,
        max_iterations=15000,
        activation='HyperbolicTangent',
        normalization='Statistical',
        learning_rate=0.2,
        momentum=0.2
):
    number_of_inputs, number_of_outputs = len(stock_prices_training_set[0][0]), len(stock_prices_training_set[0][1])

    layers = (
        InputLayer(number_of_inputs, number_of_inputs_per_neuron=1, activation_function='Identity'),
        HiddenLayer(
            hidden_neuron_count,
            number_of_inputs_per_neuron=number_of_inputs,
            activation_function=activation,
            learning_rate=learning_rate,
            momentum=momentum
        ),

        OutputLayer(                            # The activation function is giving the dot product, so do nothing ...
            number_of_outputs, number_of_inputs_per_neuron=hidden_neuron_count, activation_function='Identity'
        )
    )

    return (training_func or train)(
        NeuralNetwork(
            layers,
            allowed_error_threshold=threshold,
            max_number_of_iterations=max_iterations,
            normalization_class='Statistical'
        ),
        stock_prices_training_set
    )


def predict_last(neural_network, network_input, expected_output):
    print("last entry, actual closing_price: {actual} predicted: {predicted}".format(
        actual=expected_output, predicted=neural_network.feed(network_input)
    ))
    return neural_network


def sliding_window(values, window_size):
    return tuple(
        (values[index:(index + window_size)], values[(index + window_size):(index + window_size + 1)])
        for index in range(len(values) - window_size)
    )


def train_as_time_series(stock_prices, window_size, training_func=None, **kwargs):
    return predict_last(
        get_trained_stock_neural_network(
            sliding_window(stock_prices, window_size),
            training_func=training_func,
            **kwargs
        ),
        stock_prices[-(window_size + 1):-1],
        stock_prices[-1]
    )


def train_as_function_interpolation(stock_data, training_func=None, **kwargs):
    return predict_last(
        get_trained_stock_neural_network(
            tuple(((float(index),), (stock_price,)) for index, stock_price in enumerate(stock_data)),
            training_func=training_func,
            **kwargs
        ),
        len(stock_data) - 1,
        stock_data[-1]
    )

def csv2rec(filename):
    return numpy.recfromtxt(filename, dtype=None, delimiter=',', names=True, encoding='utf-8')

def plot(actual_data, predicted_data, endimpo_of_training_point, x_axis, title='', errors=None):
    fig, axs = plt.subplots(2, sharex=False, sharey=False)

    actual_prices_plot = axs[0].plot_date(x_axis, actual_data, 'b*-')
    predicted_prices_plot = axs[0].plot_date(x_axis, predicted_data, 'r*--', label='predicted_prices')

    axs[0].set_ylabel("Adjusted Closing Price")
    axs[0].yaxis.set_major_formatter(plt.FormatStrFormatter('$%.2f'))
    axs[0].axvline(x=endimpo_of_training_point, color='orange')

    errors_as_percentage = \
        ((numpy.asarray(actual_data) - numpy.asarray(predicted_data)) / numpy.asarray(actual_data)) * 100
    positive_values, negative_values = errors_as_percentage >= 0, errors_as_percentage < 0
    axs[1].bar(numpy.asarray(x_axis)[positive_values], errors_as_percentage[positive_values], color='g')
    axs[1].bar(numpy.asarray(x_axis)[negative_values], errors_as_percentage[negative_values], color='r')
    axs[1].axvline(x=endimpo_of_training_point, color='orange')
    axs[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%3.3f%%'))

    axs[1].set_ylabel("difference as percentage")
    axs[0].set_xlabel("Trading Day")

    fig.autofmt_xdate()
    plt.title(title)
    ####################################################################################

    fig, ax = plt.subplots(1)
    actual_prices_plot = plt.plot(x_axis, actual_data, 'b*-')
    predicted_prices_plot = plt.plot(x_axis, predicted_data, 'r*--', label='predicted_prices')
    plt.axvline(x=endimpo_of_training_point, color='orange')
    fig.autofmt_xdate()

    if errors:
        plt.figure()
        _ = plt.plot(range(len(errors)), errors, 'r--')
        plt.xlabel('Epoch')
        plt.ylabel('Relative Error')
        plt.title('Neural Network Relative Error History')

    plt.ioff()
    plt.show()


def main(
    stock=None,
    start_date="2010/01/02",
    end_date="2018/8/8",
    training=None,
    time_series=None,
    animate=None,
    **kwargs
):
    training = 20
    end_date = end_date or datetime.today()
    start_date = start_date or end_date - timedelta(days=int((training > 1 and training) or 30))
    # end_date = datetime.today()
    # start_date = start_date #or end_date - timedelta(days=int((training > 1 and training) or 30))
    # print([start_date.year, start_date.month], end_date)
    # print(Fetcher(stock, start_date, end_date).getHistorical())
    stock_data = Fetcher(stock, start_date, end_date).getHistorical()
    # stock_data = Fetcher(stock, [start_date.year, start_date.month, start_date.day], [end_date.year, end_date.month, end_date.day]).getHistorical()
    stock_prices = tuple(reversed(stock_data.Close))
    dates = tuple(reversed(stock_data.Date))
    training_set = stock_prices[:int(training > 1 and training or training * len(stock_data.Close))]

    animate = (animate in {'foreground', 'background'} and get_animated_training(dates, animate == 'background')) \
        or train

    def _time_series(window_size):
        neural_network = train_as_time_series(training_set, window_size, animate, **kwargs)
        network_samples = sliding_window(stock_prices, window_size)
        predicted_prices = list(network_samples[0][0])
        predicted_prices.extend(chain.from_iterable(neural_network.feed(inputs) for inputs, _ in network_samples))
        plot(
            stock_prices,
            predicted_prices,
            dates[len(training_set) - 1],
            x_axis=dates,
            title='Trained as Time Series',
            errors=neural_network.errors
        )

    def interpolation():
        neural_network = train_as_function_interpolation(training_set, animate, **kwargs)
        predicted_prices = list(chain.from_iterable(map(neural_network.feed, range(len(stock_prices)))))
        plot(
            stock_prices,
            predicted_prices,
            dates[len(training_set) - 1],
            x_axis=dates,
            title='Trained as Function Interpolation',
            errors=neural_network.errors
        )

    if time_series:
        _time_series(time_series)
    else:
        interpolation()


def parse_date(string):
    return datetime.strptime(string, '%Y-%m-%d').date()


def grab_command_line_arguments():
    parser = argparse.ArgumentParser(description='Neural Network Stock price predictor.')
    parser.add_argument('--stock', type=str, default='SBIN.BO', nargs='?', help='Stock symbol such as appl')
    parser.add_argument('--start_date', type=parse_date, nargs='?', default=None, help='start date of the data set such as 2014-01-01')
    parser.add_argument('--end_date', type=parse_date, default=None, nargs='?', help='end date of the data set such as %s' % str(datetime.today().date()))
    parser.add_argument('--training', default=None, type=float, nargs='?', help='Training set quantity, magnitude if value greater than 1 or percentage if less than 1')
    parser.add_argument('--time_series', type=int, default=None, nargs='?', help='treat the training set as a time series with a giving window size otherwise just interpolate')
    parser.add_argument('--animate', default=None, help='Animate current progress either in the foreground of the background' )
    parser.add_argument('--max_iterations', type=int, default=1000, nargs='?', help='Maximum number of iterations.')
    parser.add_argument('--threshold', type=float, default=.001, nargs='?', help='Minimum, relative error before halting.')
    parser.add_argument('--hidden_neuron_count', type=int, default=10, nargs='?', help='Number of Hidden Neurons to be used')
    parser.add_argument('--activation', default='HyperbolicTangent', nargs='?', help='Hidden layers Activation function default: HyperbolicTangent')
    parser.add_argument('--normalization', default='Statistical', nargs='?', help='Normalization Class default: Statistical')
    parser.add_argument('--learning_rate', type=float, default=.2, help='Neural Networks learning rate defaults to 0.2')
    parser.add_argument('--momentum', type=float, default=.2, help='Momentum Term defaults to 0.2')
    
    # print("Checking something: ", parser.parse_args().start_date)

    main(**vars(parser.parse_args()))


if __name__ == '__main__':
    grab_command_line_arguments()
