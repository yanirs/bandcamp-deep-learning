from time import time
import sys

from commandr import command

from architectures import build_model
from data import load_raw_dataset
from modeling import create_train_iter_function, create_eval_function


def run_training_iteration(train_iter, num_training_batches):
    """Run a single training iteration, returning the training loss."""
    return sum(train_iter(b) for b in xrange(num_training_batches)) / num_training_batches


def check_validation_loss_and_accuracy(valid_iter, num_validation_batches):
    """Return the validation loss and accuracy."""
    sum_losses = 0.0
    sum_accuracies = 0.0
    for b in xrange(num_validation_batches):
        batch_loss, batch_accuracy = valid_iter(b)
        sum_losses += batch_loss
        sum_accuracies += batch_accuracy
    return sum_losses / num_validation_batches, sum_accuracies / num_validation_batches


def _create_iter_functions_and_dataset(dataset_json, architecture_name, batch_size):
    # Not caching the dataset because it takes longer to load it from pickle
    dataset, label_to_index = load_raw_dataset(dataset_json)
    output_layer = build_model(architecture_name,
                               input_dim=dataset['training'][0].shape[1],
                               output_dim=len(label_to_index),
                               batch_size=batch_size)
    return (
        create_train_iter_function(dataset, output_layer),
        create_eval_function(dataset, 'validation', output_layer),
        dataset
    )


@command
def run_experiment(dataset_json=None, architecture_name=None, train_iter=None, valid_iter=None,
                   dataset=None, num_epochs=500, batch_size=100):
    """Run a deep learning experiment, reporting results to standard output.

    Command line or in-process arguments:
     * dataset_json (str) - path of JSON containing image paths (see data.collect_dataset_filenames)
     * architecture_name (str) - the name of the architecture to use (see architectures.build_model)
     * num_epochs (int) - number of training epochs to run
     * batch_size (int) - number of examples to feed to the network in each batch

    In-process-only arguments:
     * train_iter (theano.function) - training iteration function, as returned from modeling.create_train_iter_function
     * valid_iter (theano.function) - validation evaluation function, as returned from modeling.create_eval_function
     * dataset (dict) - the raw dataset, as returned from data.load_raw_dataset
    """
    if dataset_json is None:
        assert train_iter is not None and dataset is not None
    else:
        train_iter, valid_iter, dataset = \
            _create_iter_functions_and_dataset(dataset_json, architecture_name, batch_size)

    num_training_batches = dataset['training'][0].shape[0] // batch_size
    num_validation_batches = dataset['validation'][0].shape[0] // batch_size
    now = time()
    try:
        for epoch in xrange(num_epochs):
            training_loss = run_training_iteration(train_iter, num_training_batches)
            validation_loss, validation_accuracy = check_validation_loss_and_accuracy(valid_iter,
                                                                                      num_validation_batches)
            print('Epoch %s of %s took %.3fs' % (epoch + 1, num_epochs, time() - now))
            now = time()
            print('\ttraining loss:\t\t %.6f' % training_loss)
            print('\tvalidation loss:\t\t %.6f' % validation_loss)
            print('\tvalidation accuracy:\t\t %.2f%%' % (validation_accuracy * 100))
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
