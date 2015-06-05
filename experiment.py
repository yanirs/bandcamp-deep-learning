from time import time
import sys

from commandr import command

from architectures import build_model
from data import load_raw_dataset
from modeling import create_training_function, create_eval_function


def _create_iter_functions(dataset_json, architecture_name, batch_size, training_chunk_size):
    # Not caching the dataset because it takes longer to load it from pickle
    dataset, label_to_index = load_raw_dataset(dataset_json)
    output_layer = build_model(architecture_name,
                               input_dim=dataset['training'][0].shape[1],
                               output_dim=len(label_to_index),
                               batch_size=batch_size)
    return (create_training_function(dataset, output_layer, batch_size, training_chunk_size),
            create_eval_function(dataset, 'validation', output_layer, batch_size))


@command
def run_experiment(dataset_json=None, architecture_name=None, training_iter=None, validation_eval=None, num_epochs=500,
                   batch_size=100, training_chunk_size=0):
    """Run a deep learning experiment, reporting results to standard output.

    Command line or in-process arguments:
     * dataset_json (str) - path of JSON containing image paths (see data.collect_dataset_filenames)
     * architecture_name (str) - the name of the architecture to use (see architectures.build_model)
     * num_epochs (int) - number of training epochs to run
     * batch_size (int) - number of examples to feed to the network in each batch
     * training_chunk_size (int) - number of training examples to copy to the GPU in each chunk. If set to zero, all
                                   examples will be copied. This is faster, but impossible when the size of the training
                                   set is larger than the GPU's memory

    In-process-only arguments:
     * training_iter (function) - a function that runs one iteration of model updates and returns the training loss
     * validation_eval (function) - a function that returns the loss and accuracy of the model on the validation subset
    """
    if dataset_json is None:
        assert training_iter is not None and validation_eval is not None
    else:
        training_iter, validation_eval = _create_iter_functions(dataset_json, architecture_name, batch_size,
                                                                training_chunk_size)

    now = time()
    try:
        for epoch in xrange(num_epochs):
            training_loss = training_iter()
            validation_loss, validation_accuracy = validation_eval()
            print('Epoch %s of %s took %.3fs' % (epoch + 1, num_epochs, time() - now))
            now = time()
            print('\ttraining loss:\t\t %.6f' % training_loss)
            print('\tvalidation loss:\t\t %.6f' % validation_loss)
            print('\tvalidation accuracy:\t\t %.2f%%' % (validation_accuracy * 100))
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
