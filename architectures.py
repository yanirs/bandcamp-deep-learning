import inspect
import sys

from lasagne.layers import Conv2DLayer, DenseLayer, DropoutLayer, MaxPool2DLayer

from modeling import AbstractModelBuilder


class SingleLayerMlp(AbstractModelBuilder):
    """Builder of a multi-layer perceptron with a single hidden layer and a user-specified number of hidden units."""

    def _build_middle(self, l_in, num_hidden_units=512, **_):
        return DenseLayer(l_in, num_units=num_hidden_units)


class LasagneMnistExample(AbstractModelBuilder):
    """Builder of Lasagne's basic MNIST example model (examples/mnist.py).

    The network's architecture is: in -> dense -> dropout -> dense -> dropout -> out.
    """

    def _build_middle(self, l_in, num_hidden_units=512, **_):
        return _build_dense_plus_dropout(_build_dense_plus_dropout(l_in, num_hidden_units), num_hidden_units)


class ConvNet(AbstractModelBuilder):
    """Builder of a convnet architecture with a user-specified number of convolutional layers and dense layers.

    Every convolutional layer is followed by a max pool layer, and every dense layer is followed by a dropout layer.
    """

    def _build_middle(self, l_in, num_conv_layers=1, num_dense_layers=1, **kwargs):
        assert len(l_in.shape) == 4, 'InputLayer shape must be (batch_size, channels, width, height) -- ' \
                                     'reshape data or use RGB format?'
        l_bottom = l_in
        for i in xrange(num_conv_layers):
            l_bottom = _build_conv_plus_max_pool(l_bottom,
                                                 num_filters=kwargs['lc%d_num_filters' % i],
                                                 filter_size=kwargs['lc%d_filter_size' % i],
                                                 pool_size=kwargs['lc%d_pool_size' % i])
        for i in xrange(num_dense_layers):
            l_bottom = _build_dense_plus_dropout(l_bottom,
                                                 num_hidden_units=kwargs['ld%d_num_hidden_units' % i])
        return l_bottom


class LasagneMnistConvExample(ConvNet):
    """Builder of Lasagne's MNIST basic CNN example (examples/mnist_conv.py).

    The network's architecture is: in -> conv(5, 5) x 32 -> max-pool(2, 2) -> conv(5, 5) x 32 -> max-pool(2, 2) ->
                                   dense (256) -> dropout -> out.
    """

    def _build_middle(self, l_in, **_):
        return super(LasagneMnistConvExample, self)._build_middle(
            l_in, num_conv_layers=2, num_dense_layers=1,
            lc0_num_filters=32, lc0_filter_size=(5, 5), lc0_pool_size=(2, 2),
            lc1_num_filters=32, lc1_filter_size=(5, 5), lc1_pool_size=(2, 2),
            ld0_num_hidden_units=256
        )


def _build_dense_plus_dropout(incoming_layer, num_hidden_units):
    return DropoutLayer(DenseLayer(incoming_layer, num_units=num_hidden_units), p=0.5)


def _build_conv_plus_max_pool(incoming_layer, num_filters, filter_size, pool_size):
    return MaxPool2DLayer(Conv2DLayer(incoming_layer, num_filters=num_filters, filter_size=filter_size),
                          pool_size=pool_size)


# A bit of Python magic to publish the available architectures.
ARCHITECTURE_NAME_TO_CLASS = dict(inspect.getmembers(sys.modules[__name__],
                                                     lambda obj: (inspect.isclass(obj) and
                                                                  issubclass(obj, AbstractModelBuilder) and
                                                                  obj != AbstractModelBuilder)))
