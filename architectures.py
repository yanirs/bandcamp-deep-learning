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


class LasagneMnistConvExample(AbstractModelBuilder):
    """Builder of Lasagne's MNIST basic CNN example (examples/mnist_conv.py).

    The network's architecture is: in -> conv(5, 5) x 32 -> max-pool(2, 2) -> conv(5, 5) x 32 -> max-pool(2, 2) ->
                                   dense (256) -> dropout -> out.
    """

    def _build_middle(self, l_in, l1_num_filters=32, l1_filter_size=(5, 5), l1_pool_size=(2, 2),
                      l2_num_filters=32, l2_filter_size=(5, 5), l2_pool_size=(2, 2), dense_num_hidden_units=256, **_):
        assert len(l_in.shape) == 4, 'InputLayer shape must be (batch_size, channels, width, height) -- reshape data?'
        l_conv_pool1 = _build_conv_plus_max_pool(
            l_in, num_filters=l1_num_filters, filter_size=l1_filter_size, pool_size=l1_pool_size
        )
        l_conv_pool2 = _build_conv_plus_max_pool(
            l_conv_pool1, num_filters=l2_num_filters, filter_size=l2_filter_size, pool_size=l2_pool_size
        )
        return _build_dense_plus_dropout(l_conv_pool2, num_hidden_units=dense_num_hidden_units)


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
