import lasagne
import theano
import theano.tensor as T


def create_training_function(dataset, output_layer, learning_rate=0.01, momentum=0.9):
    """Return a function that runs one iteration of model updates on the training data and returns the training loss."""
    theano_function = _create_theano_training_function(dataset, output_layer, learning_rate, momentum)
    num_batches = dataset['training'][0].shape[0] // _get_batch_size(output_layer)
    return lambda: sum(theano_function(b) for b in xrange(num_batches)) / num_batches


def _create_theano_training_function(dataset, output_layer, learning_rate, momentum):
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
        givens=_create_batch_givens(dataset, 'training', output_layer,
                                    batch_index_var, batch_instances_var, batch_labels_var)
    )


def create_eval_function(dataset, subset_name, output_layer):
    """Return a function that returns the loss and accuracy of the given network on the given dataset's subset."""
    theano_function = _create_theano_eval_function(dataset, subset_name, output_layer)
    num_validation_batches = dataset[subset_name][0].shape[0] // _get_batch_size(output_layer)

    def run_theano_function():
        sum_losses = 0.0
        sum_accuracies = 0.0
        for b in xrange(num_validation_batches):
            batch_loss, batch_accuracy = theano_function(b)
            sum_losses += batch_loss
            sum_accuracies += batch_accuracy
        return sum_losses / num_validation_batches, sum_accuracies / num_validation_batches

    return run_theano_function


def _create_theano_eval_function(dataset, subset_name, output_layer):
    """
    Return a theano.function that takes a batch index input, and returns a tuple of the loss and accuracy for the
    given subset of the dataset.
    """
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
        givens=_create_batch_givens(dataset, subset_name, output_layer,
                                    batch_index_var, batch_instances_var, batch_labels_var)
    )


def _get_batch_size(layer):
    return _get_batch_size(layer.input_layer) if hasattr(layer, 'input_layer') else layer.shape[0]


def _create_batch_givens(dataset, subset_name, output_layer, batch_index_var, batch_instances_var, batch_labels_var):
    batch_size = _get_batch_size(output_layer)
    batch_slice = slice(batch_index_var * batch_size, (batch_index_var + 1) * batch_size)
    instances, labels = (theano.shared(_) for _ in dataset[subset_name])
    return {batch_instances_var: instances[batch_slice], batch_labels_var: labels[batch_slice]}


def _create_loss_eval_func(output_layer, batch_instances_var, batch_labels_var, deterministic,
                           loss_function=lasagne.objectives.categorical_crossentropy):
    return lasagne.objectives.Objective(output_layer, loss_function=loss_function).get_loss(
        batch_instances_var, target=batch_labels_var, deterministic=deterministic
    )


def _create_batch_vars():
    return T.iscalar('batch_index'), T.matrix('x'), T.ivector('y')
