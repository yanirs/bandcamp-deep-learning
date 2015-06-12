from ast import literal_eval
import inspect
from time import time
import sys
from types import FunctionType

from commandr import command
import lasagne
import numpy as np
import theano
from theano_latest.misc import pkl_utils

from architectures import ARCHITECTURE_NAME_TO_CLASS


@command
def run_experiment(dataset_path, model_architecture, model_params=None, num_epochs=500, batch_size=100,
                   training_chunk_size=0, reshape_to=None, learning_rate=0.01):
    """Run a deep learning experiment, reporting results to standard output.

    Command line or in-process arguments:
     * dataset_path (str) - path of dataset pickle zip (see data.create_datasets)
     * model_architecture (str) - the name of the architecture to use (subclass of architectures.AbstractModelBuilder)
     * model_params (str) - colon-separated list of equals-separated key-value pairs to pass to the model builder.
                            All keys are assumed to be strings, while values are evaluated as Python literals
     * num_epochs (int) - number of training epochs to run
     * batch_size (int) - number of examples to feed to the network in each batch
     * training_chunk_size (int) - number of training examples to copy to the GPU in each chunk. If set to zero, all
                                   examples will be copied. This is faster, but impossible when the size of the training
                                   set is larger than the GPU's memory
     * reshape_to (str) - if given, the data will be reshaped to match this string, which should evaluate to a Python
                          tuple of ints (e.g., may be required to make the dataset fit into a convnet input layer)
     * learning_rate (float) - learning rate to use for training the network
    """
    assert theano.config.floatX == 'float32', 'Theano floatX must be float32 to ensure consistency with pickled dataset'
    if not model_architecture in ARCHITECTURE_NAME_TO_CLASS:
        raise ValueError('Unknown architecture %s (valid values: %s)' % (model_architecture,
                                                                         sorted(ARCHITECTURE_NAME_TO_CLASS)))

    dataset, label_to_index = _load_data(dataset_path, reshape_to)
    model_builder = ARCHITECTURE_NAME_TO_CLASS[model_architecture](
        dataset, output_dim=len(label_to_index), batch_size=batch_size, training_chunk_size=training_chunk_size,
        learning_rate=learning_rate
    )
    output_layer, training_iter, validation_eval = model_builder.build(**_parse_model_params(model_params))
    _print_network_info(output_layer)
    _run_training_loop(training_iter, validation_eval, num_epochs)


def _load_data(dataset_path, reshape_to):
    with open(dataset_path, 'rb') as dataset_file:
        dataset, label_to_index = pkl_utils.load(dataset_file)
    if reshape_to:
        reshape_to = literal_eval(reshape_to)
        for subset_name, (data, labels) in dataset.iteritems():
            dataset[subset_name] = data.reshape((data.shape[0], ) + reshape_to), labels
    return dataset, label_to_index


def _parse_model_params(model_params):
    param_kwargs = {}
    if model_params:
        for pair in model_params.split(':'):
            key, value = pair.split('=')
            try:
                param_kwargs[key] = literal_eval(value)
            except (SyntaxError, ValueError):
                param_kwargs[key] = value
        print('Parsed model params: {}'.format(param_kwargs))
    return param_kwargs


def _get_default_init_kwargs(obj):
    args, _, _, defaults = inspect.getargspec(obj.__init__)
    return dict(zip(reversed(args), reversed(defaults)))


def _print_network_info(output_layer):
    print('Network architecture ({:,} parameters overall):'.format(lasagne.layers.count_params(output_layer)))
    for layer in lasagne.layers.get_all_layers(output_layer):
        init_kwargs = _get_default_init_kwargs(layer)
        filtered_params = {}
        for key, value in layer.__dict__.iteritems():
            if key.startswith('_') or key in ('name', 'input_var', 'input_layer', 'W', 'b', 'params') or \
               (key in init_kwargs and init_kwargs[key] == value):
                continue
            if isinstance(value, FunctionType):
                value = value.__name__
            filtered_params[key] = value
        print('\t%s(%s)' % (layer.__class__.__name__,
                            ', '.join('%s=%s' % (k, v) for k, v in sorted(filtered_params.iteritems()))))


def _run_training_loop(training_iter, validation_eval, num_epochs):
    now = time()
    try:
        validation_loss, validation_accuracy = validation_eval()
        print('Initial validation loss & accuracy:\t %.6f\t%.2f%%' % (validation_loss, validation_accuracy * 100))

        for epoch in xrange(num_epochs):
            training_loss = training_iter()
            validation_loss, validation_accuracy = validation_eval()
            print('Epoch %s of %s took %.3fs' % (epoch + 1, num_epochs, time() - now))
            now = time()
            print('\ttraining loss:\t\t\t %.6f' % training_loss)
            print('\tvalidation loss & accuracy:\t %.6f\t%.2f%%' % (validation_loss, validation_accuracy * 100))
            sys.stdout.flush()

            if np.isnan(training_loss) or np.isnan(validation_loss) or np.isnan(validation_accuracy):
                print('Divergence detected. Stopping now.')
                break
    except KeyboardInterrupt:
        pass
