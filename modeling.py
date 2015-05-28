import lasagne
import theano
import theano.tensor as T


def create_iter_functions(dataset, output_layer, X_tensor_type=T.matrix, learning_rate=0.01, momentum=0.9):
    """Create functions for training, validation and testing to iterate one epoch."""

    get_batch_size = lambda l: get_batch_size(l.input_layer) if hasattr(l, 'input_layer') else l.shape[0]
    batch_size = get_batch_size(output_layer)

    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')
    batch_slice = slice(batch_index * batch_size, (batch_index + 1) * batch_size)

    objective = lasagne.objectives.Objective(output_layer,
                                             loss_function=lasagne.objectives.categorical_crossentropy)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)

    pred = T.argmax(lasagne.layers.get_output(output_layer, X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.nesterov_momentum(loss_train, all_params, learning_rate, momentum)

    return dict(
        train=theano.function([batch_index],
                              loss_train,
                              updates=updates,
                              givens={X_batch: dataset['X_train'][batch_slice],
                                      y_batch: dataset['y_train'][batch_slice]}),
        valid=theano.function([batch_index],
                              [loss_eval, accuracy],
                              givens={X_batch: dataset['X_valid'][batch_slice],
                                      y_batch: dataset['y_valid'][batch_slice]}),
        test=theano.function([batch_index],
                             [loss_eval, accuracy],
                             givens={X_batch: dataset['X_test'][batch_slice],
                                     y_batch: dataset['y_test'][batch_slice]})
    )
