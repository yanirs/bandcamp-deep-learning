from math import ceil
from warnings import warn

import lasagne
import theano
import theano.tensor as T


def create_training_function(dataset, output_layer, batch_size, chunk_size=0, learning_rate=0.01, momentum=0.9):
    """Return a function that runs one iteration of model updates on the training data and returns the training loss."""
    if chunk_size and chunk_size < batch_size:
        raise ValueError('Chunk size must be greater than or equal to batch_size')

    instances_var = lasagne.utils.shared_empty()
    labels_var = lasagne.utils.shared_empty(dim=1, dtype='int32')
    theano_function = _create_theano_training_function(instances_var, labels_var, output_layer, batch_size,
                                                       learning_rate, momentum)

    instances, labels = dataset['training']
    if chunk_size:
        if not instances.shape[0] % chunk_size == 0:
            warn('Number of training instances is not divisable by chunk_size')
        num_chunks = int(ceil(instances.shape[0] / chunk_size))
        num_batches = chunk_size // batch_size

        def run_theano_function():
            sum_loss = 0.0
            for c in xrange(num_chunks):
                chunk_slice = slice(c * chunk_size, (c + 1) * chunk_size)
                instances_var.set_value(instances[chunk_slice])
                labels_var.set_value(labels[chunk_slice])
                for b in xrange(num_batches):
                    sum_loss += theano_function(b)
            return sum_loss / (num_batches * num_chunks)
    else:
        instances_var.set_value(instances)
        labels_var.set_value(labels)
        num_batches = instances.shape[0] // batch_size
        run_theano_function = lambda: sum(theano_function(b) for b in xrange(num_batches)) / num_batches

    return run_theano_function


def _create_theano_training_function(instances_var, labels_var, output_layer, batch_size, learning_rate, momentum):
    """Return a theano.function that takes a batch index input, updates the network, and returns the training loss."""
    batch_index_var, batch_instances_var, batch_labels_var = _create_batch_vars()
    loss_eval_function = _create_loss_eval_func(
        output_layer, batch_instances_var, batch_labels_var, deterministic=False
    )
    return theano.function(
        [batch_index_var],
        loss_eval_function,
        updates=lasagne.updates.nesterov_momentum(
            loss_eval_function,
            lasagne.layers.get_all_params(output_layer),
            learning_rate,
            momentum
        ),
        givens=_create_batch_givens(instances_var, labels_var, batch_size, batch_index_var, batch_instances_var,
                                    batch_labels_var)
    )


def create_eval_function(dataset, subset_name, output_layer, batch_size):
    """Return a function that returns the loss and accuracy of the given network on the given dataset's subset."""
    instances_var, labels_var = (theano.shared(_) for _ in dataset[subset_name])
    theano_function = _create_theano_eval_function(instances_var, labels_var, output_layer, batch_size)
    num_validation_batches = dataset[subset_name][0].shape[0] // batch_size

    def run_theano_function():
        sum_losses = 0.0
        sum_accuracies = 0.0
        for b in xrange(num_validation_batches):
            batch_loss, batch_accuracy = theano_function(b)
            sum_losses += batch_loss
            sum_accuracies += batch_accuracy
        return sum_losses / num_validation_batches, sum_accuracies / num_validation_batches

    return run_theano_function


def _create_theano_eval_function(instances_var, labels_var, output_layer, batch_size):
    """Return a theano.function that takes a batch index input, and returns a tuple of the loss and accuracy."""
    batch_index_var, batch_instances_var, batch_labels_var = _create_batch_vars()
    accuracy_eval_func = T.mean(
        T.eq(T.argmax(lasagne.layers.get_output(output_layer, batch_instances_var, deterministic=True), axis=1),
             batch_labels_var),
        dtype=theano.config.floatX
    )
    return theano.function(
        [batch_index_var],
        [
            _create_loss_eval_func(output_layer, batch_instances_var, batch_labels_var, deterministic=True),
            accuracy_eval_func
        ],
        givens=_create_batch_givens(instances_var, labels_var, batch_size, batch_index_var, batch_instances_var,
                                    batch_labels_var)
    )


def _create_batch_givens(instances_var, labels_var, batch_size, batch_index_var, batch_instances_var, batch_labels_var):
    batch_slice = slice(batch_index_var * batch_size, (batch_index_var + 1) * batch_size)
    return {batch_instances_var: instances_var[batch_slice], batch_labels_var: labels_var[batch_slice]}


def _create_loss_eval_func(output_layer, batch_instances_var, batch_labels_var, deterministic,
                           loss_function=lasagne.objectives.categorical_crossentropy):
    return lasagne.objectives.Objective(output_layer, loss_function=loss_function).get_loss(
        batch_instances_var, target=batch_labels_var, deterministic=deterministic
    )


def _create_batch_vars():
    return T.iscalar('batch_index'), T.matrix('x'), T.ivector('y')
