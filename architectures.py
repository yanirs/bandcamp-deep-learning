import inspect
import sys

from lasagne.layers import Conv2DLayer, DenseLayer, DropoutLayer, MaxPool2DLayer

from modeling import AbstractModelBuilder


class SingleLayerMlp(AbstractModelBuilder):
    """Builder of a multi-layer perceptron with a single hidden layer and a user-specified number of hidden units."""

    def _build_middle(self, l_in, num_units=512, **_):
        return DenseLayer(l_in, num_units=num_units)


class LasagneMnistExample(AbstractModelBuilder):
    """Builder of Lasagne's basic MNIST example model (examples/mnist.py).

    The network's architecture is: in -> dense -> dropout -> dense -> dropout -> out.
    """

    def _build_middle(self, l_in, num_units=512, **_):
        return _build_dense_plus_dropout(_build_dense_plus_dropout(l_in, num_units), num_units)


class ConvNet(AbstractModelBuilder):
    """Builder of a convnet architecture with a user-specified number of convolutional layers and dense layers.

    Every conv_pool_ratio convolutional layers are followed by a max pool layer, and every dense layer is followed by a
    dropout layer.
    """

    def _build_middle(self, l_in, num_conv_layers=1, num_dense_layers=1, conv_pool_ratio=1, **kwargs):
        assert len(l_in.shape) == 4, 'InputLayer shape must be (batch_size, channels, width, height) -- ' \
                                     'reshape data or use RGB format?'
        assert num_conv_layers % conv_pool_ratio == 0

        l_bottom = l_in
        for i in xrange(num_conv_layers):
            conv_kwargs = self._extract_layer_kwargs('c', i, kwargs)
            if 'border_mode' not in conv_kwargs:
                conv_kwargs['border_mode'] = 'full'
            l_bottom = Conv2DLayer(l_bottom, **conv_kwargs)

            if (i + 1) % conv_pool_ratio == 0:
                max_pool_kwargs = self._extract_layer_kwargs('m', i / conv_pool_ratio, kwargs)
                if 'pool_size' not in max_pool_kwargs:
                    max_pool_kwargs['pool_size'] = (2, 2)
                l_bottom = MaxPool2DLayer(l_bottom, **max_pool_kwargs)

        for i in xrange(num_dense_layers):
            l_bottom = _build_dense_plus_dropout(l_bottom, **self._extract_layer_kwargs('d', i, kwargs))

        return l_bottom

    def _extract_layer_kwargs(self, layer_letter, layer_index, kwargs):
        return {k.split('_', 1)[1]: v for k, v in kwargs.iteritems()
                if k.startswith('l%s%d' % (layer_letter, layer_index))}


class LasagneMnistConvExample(ConvNet):
    """Builder of Lasagne's MNIST basic CNN example (examples/mnist_conv.py).

    The network's architecture is: in -> conv(5, 5) x 32 -> max-pool(2, 2) -> conv(5, 5) x 32 -> max-pool(2, 2) ->
                                   dense (256) -> dropout -> out.
    """

    def _build_middle(self, l_in, **_):
        return super(LasagneMnistConvExample, self)._build_middle(
            l_in, num_conv_layers=2, num_dense_layers=1,
            lc0_num_filters=32, lc0_filter_size=(5, 5),
            lc1_num_filters=32, lc1_filter_size=(5, 5),
            ld0_num_units=256
        )


def _build_dense_plus_dropout(incoming_layer, num_units):
    return DropoutLayer(DenseLayer(incoming_layer, num_units=num_units), p=0.5)


# A bit of Python magic to publish the available architectures.
ARCHITECTURE_NAME_TO_CLASS = dict(inspect.getmembers(sys.modules[__name__],
                                                     lambda obj: (inspect.isclass(obj) and
                                                                  issubclass(obj, AbstractModelBuilder) and
                                                                  obj != AbstractModelBuilder)))
