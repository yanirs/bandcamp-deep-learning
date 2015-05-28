import lasagne


def build_model(name, input_dim, output_dim, batch_size=100, num_hidden_units=512):
    """
    Create a symbolic representation of a neural network with `intput_dim` input nodes, `output_dim` output
    nodes and `num_hidden_units` per hidden layer.

    The training function of this model must have a mini-batch size of `batch_size`.

    A theano expression which represents such a network is returned.
    """

    l_in = lasagne.layers.InputLayer(shape=(batch_size, input_dim))

    if name == 'mlp':
        l_penultimate = lasagne.layers.DenseLayer(l_in, num_units=num_hidden_units)
    elif name == 'mnist_demo':
        l_hidden1 = lasagne.layers.DenseLayer(l_in,
                                              num_units=num_hidden_units,
                                              nonlinearity=lasagne.nonlinearities.rectify)
        l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)
        l_hidden2 = lasagne.layers.DenseLayer(l_hidden1_dropout,
                                              num_units=num_hidden_units,
                                              nonlinearity=lasagne.nonlinearities.rectify)
        l_penultimate = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)
    else:
        raise ValueError('Unknown architecture name %s' % name)

    return lasagne.layers.DenseLayer(l_penultimate,
                                     num_units=output_dim,
                                     nonlinearity=lasagne.nonlinearities.softmax)
