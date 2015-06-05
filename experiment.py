from time import time
import sys

from commandr import command
import theano

from theano_latest.misc import pkl_utils
from architectures import build_model
from modeling import create_training_function, create_eval_function


def _create_iter_functions(dataset_path, architecture_name, batch_size, training_chunk_size):
    with open(dataset_path, 'rb') as dataset_file:
        dataset, label_to_index = pkl_utils.load(dataset_file)
    output_layer = build_model(architecture_name,
                               input_dim=dataset['training'][0].shape[1],
                               output_dim=len(label_to_index),
                               batch_size=batch_size)
    return (create_training_function(dataset, output_layer, batch_size, training_chunk_size),
            create_eval_function(dataset, 'validation', output_layer, batch_size))


@command
def run_experiment(dataset_path=None, architecture_name=None, training_iter=None, validation_eval=None, num_epochs=500,
                   batch_size=100, training_chunk_size=0):
    """Run a deep learning experiment, reporting results to standard output.

    Command line or in-process arguments:
     * dataset_path (str) - path of dataset pickle zip (see data.create_datasets)
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
    assert theano.config.floatX == 'float32', 'Theano floatX must be float32 to ensure consistency with pickled dataset'

    if dataset_path is None:
        assert training_iter is not None and validation_eval is not None
    else:
        training_iter, validation_eval = _create_iter_functions(dataset_path, architecture_name, batch_size,
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
