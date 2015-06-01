import itertools
from time import time
import sys

import commandr
import numpy as np

from architectures import build_model
from data import load_raw_dataset, create_lasagne_dataset
from modeling import create_iter_functions


def train(iter_funcs, dataset, batch_size):
    """Train the model with `dataset` with mini-batch training. Each mini-batch has `batch_size` recordings."""

    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    for epoch in itertools.count(1):
        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in xrange(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        yield {
            'number': epoch,
            'train_loss': np.mean([iter_funcs['train'](b) for b in xrange(num_batches_train)]),
            'valid_loss': np.mean(batch_valid_losses),
            'valid_accuracy': np.mean(batch_valid_accuracies),
        }


def _create_iter_functions_and_dataset(dataset_json, architecture_name, batch_size):
    lasagne_dataset = create_lasagne_dataset(*load_raw_dataset(dataset_json))
    output_layer = build_model(architecture_name,
                               input_dim=lasagne_dataset['input_dim'],
                               output_dim=lasagne_dataset['output_dim'],
                               batch_size=batch_size)
    iter_funcs = create_iter_functions(lasagne_dataset, output_layer)
    return iter_funcs, lasagne_dataset


@commandr.command
def run_experiment(dataset_json=None, architecture_name=None, iter_funcs=None, lasagne_dataset=None, num_epochs=500,
                   batch_size=100):
    """Run a deep learning experiment, reporting results to standard output.

    Command line or in-process arguments:
     * dataset_json (str) - path of JSON containing image paths (see data.collect_dataset_filenames)
     * architecture_name (str) - the name of the architecture to use (see architectures.build_model)
     * num_epochs (int) - number of training epochs to run
     * batch_size (int) - number of examples to feed to the network in each batch

    In-process-only arguments:
     * iter_funcs (dict) - iteration functions, as returned from modeling.create_iter_functions
     * lasagne_dataset (dict) - lasagne dataset, as returned from data.create_lasagne_dataset
    """
    if dataset_json is None:
        assert iter_funcs is not None and lasagne_dataset is not None
    else:
        iter_funcs, lasagne_dataset = _create_iter_functions_and_dataset(dataset_json, architecture_name, batch_size)

    now = time()
    try:
        for epoch in train(iter_funcs, lasagne_dataset, batch_size):
            print('Epoch %s of %s took %.3fs' % (epoch['number'], num_epochs, time() - now))
            now = time()
            print('\ttraining loss:\t\t %.6f' % epoch['train_loss'])
            print('\tvalidation loss:\t\t %.6f' % epoch['valid_loss'])
            print('\tvalidation accuracy:\t\t %.2f%%' % (epoch['valid_accuracy'] * 100))
            sys.stdout.flush()

            if epoch['number'] >= num_epochs:
                break
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    commandr.Run()
