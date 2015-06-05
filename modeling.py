import lasagne
import theano
import theano.tensor as T


def create_train_iter_function(dataset, output_layer, learning_rate=0.01, momentum=0.9):
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


def _create_batch_givens(dataset, subset_name, output_layer, batch_index_var, batch_instances_var, batch_labels_var):
    get_batch_size = lambda l: get_batch_size(l.input_layer) if hasattr(l, 'input_layer') else l.shape[0]
    batch_size = get_batch_size(output_layer)
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
