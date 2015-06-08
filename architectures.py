import inspect
import lasagne
import sys


class AbstractModelBuilder(object):
    """Builder of Lasagne models, with architecture-specific implementation by concrete subclasses."""

    def __init__(self, input_dim, output_dim, batch_size):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

    def build(self, **kwargs):
        """Build the model, returning the output layer."""
        l_in = lasagne.layers.InputLayer(shape=(self.batch_size, self.input_dim))
        return lasagne.layers.DenseLayer(self._build_middle(l_in, **kwargs),
                                         num_units=self.output_dim,
                                         nonlinearity=lasagne.nonlinearities.softmax)

    def _build_middle(self, l_in, **kwargs):
        raise NotImplementedError


class SingleLayerMlp(AbstractModelBuilder):
    """Builder of a multi-layer perceptron with a single hidden layer and a user-specified number of hidden units."""

    def _build_middle(self, l_in, num_hidden_units=512):
        return lasagne.layers.DenseLayer(l_in, num_units=num_hidden_units)


class LasagneMnistExample(AbstractModelBuilder):
    """Builder of Lasagne's basic MNIST example model: in -> dense -> dropout -> dense -> dropout -> out."""

    def _build_middle(self, l_in, num_hidden_units=512):
        return self._build_dense_plus_dropout(self._build_dense_plus_dropout(l_in, num_hidden_units), num_hidden_units)

    def _build_dense_plus_dropout(self, l_in, num_hidden_units):
        return lasagne.layers.DropoutLayer(
            lasagne.layers.DenseLayer(l_in, num_units=num_hidden_units, nonlinearity=lasagne.nonlinearities.rectify),
            p=0.5
        )

# A bit of Python magic to publish the available architectures.
ARCHITECTURE_NAME_TO_CLASS = dict(inspect.getmembers(sys.modules[__name__],
                                                     lambda obj: (inspect.isclass(obj) and
                                                                  issubclass(obj, AbstractModelBuilder) and
                                                                  obj != AbstractModelBuilder)))
