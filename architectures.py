import inspect
import sys

from lasagne.nonlinearities import softmax
from lasagne.layers import Conv2DLayer, DenseLayer, DropoutLayer, InputLayer, MaxPool2DLayer


class AbstractModelBuilder(object):
    """Builder of Lasagne models, with architecture-specific implementation by concrete subclasses."""

    def __init__(self, input_shape, output_dim, batch_size):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.batch_size = batch_size

    def build(self, **kwargs):
        """Build the model, returning the output layer."""
        l_in = InputLayer(shape=(self.batch_size, ) + self.input_shape)
        return DenseLayer(self._build_middle(l_in, **kwargs), num_units=self.output_dim, nonlinearity=softmax)

    def _build_middle(self, l_in, **kwargs):
        raise NotImplementedError


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

    def _build_middle(self, l_in, **_):
        assert len(l_in.shape) == 4, 'InputLayer shape must be (batch_size, channels, width, height) -- reshape data?'
        l_conv_pool1 = _build_conv_plus_max_pool(l_in, num_filters=32, filter_size=(5, 5), pool_size=(2, 2))
        l_conv_pool2 = _build_conv_plus_max_pool(l_conv_pool1, num_filters=32, filter_size=(5, 5), pool_size=(2, 2))
        return _build_dense_plus_dropout(l_conv_pool2, num_hidden_units=256)


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
