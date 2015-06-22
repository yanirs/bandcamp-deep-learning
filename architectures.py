import inspect
import sys

from lasagne.init import HeUniform
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
except ImportError:
    from lasagne.layers import Conv2DLayer, MaxPool2DLayer
from lasagne.layers import DenseLayer, DropoutLayer

from modeling import AbstractModelBuilder


class SingleLayerMlp(AbstractModelBuilder):
    """Builder of a multi-layer perceptron with a single hidden layer and a user-specified number of hidden units."""

    def _build_middle(self, l_in, num_units=512, **_):
        return DenseLayer(l_in, num_units=num_units, W=HeUniform(gain='relu'))


class LasagneMnistExample(AbstractModelBuilder):
    """Builder of Lasagne's basic MNIST example model (examples/mnist.py).

    The network's architecture is: in -> dense -> dropout -> dense -> dropout -> out.
    """

    def _build_middle(self, l_in, num_units=512, **_):
        return _build_dense_plus_dropout(_build_dense_plus_dropout(l_in, num_units), num_units)


class ConvNet(AbstractModelBuilder):
    """Builder of a convnet architecture with a user-specified number of convolutional layers and dense layers.

    Every convolutional layer is optionally followed by a max pool layer, and every dense layer is optionally followed
    by a dropout layer.
    """

    def _build_middle(self, l_in, num_conv_layers=1, num_dense_layers=1, **kwargs):
        assert len(l_in.shape) == 4, 'InputLayer shape must be (batch_size, channels, width, height) -- ' \
                                     'reshape data or use RGB format?'

        l_bottom = l_in
        for i in xrange(num_conv_layers):
            conv_kwargs = self._extract_layer_kwargs('c', i, kwargs)
            if 'border_mode' not in conv_kwargs:
                conv_kwargs['border_mode'] = 'same'
            has_max_pool = conv_kwargs.pop('mp', False)
            l_bottom = Conv2DLayer(l_bottom, W=HeUniform(gain='relu'), **conv_kwargs)

            if has_max_pool:
                max_pool_kwargs = self._extract_layer_kwargs('m', i, kwargs)
                if 'pool_size' not in max_pool_kwargs:
                    max_pool_kwargs['pool_size'] = (2, 2)
                l_bottom = MaxPool2DLayer(l_bottom, **max_pool_kwargs)

        for i in xrange(num_dense_layers):
            dense_kwargs = self._extract_layer_kwargs('d', i, kwargs)
            has_dropout = dense_kwargs.pop('hd', True)
            l_bottom = DenseLayer(l_bottom, W=HeUniform(gain='relu'), **dense_kwargs)
            if has_dropout:
                l_bottom = DropoutLayer(l_bottom, p=0.5)

        return l_bottom

    def _extract_layer_kwargs(self, layer_letter, layer_index, kwargs):
        return {k.split('_', 1)[1]: v for k, v in kwargs.iteritems()
                if k.startswith('l%s%d_' % (layer_letter, layer_index))}


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


class VggNet(ConvNet):
    """Network D by VGGNet, from Very deep convolutional networks for large scale image recognition
    (Simonyan and Zisserman, 2014).

    Includes an optional reduction factor to reduce the number of filters and hidden units.
    """
    def _build_middle(self, l_in, reduction_factor=1, **_):
        kwargs = dict(
            num_conv_layers=13, num_dense_layers=2,
            lc0_num_filters=64, lc0_filter_size=3,
            lc1_num_filters=64, lc1_filter_size=3, lc1_mp=True,
            lc2_num_filters=128, lc2_filter_size=3,
            lc3_num_filters=128, lc3_filter_size=3, lc3_mp=True,
            lc4_num_filters=256, lc4_filter_size=3,
            lc5_num_filters=256, lc5_filter_size=3,
            lc6_num_filters=256, lc6_filter_size=3, lc6_mp=True,
            lc7_num_filters=512, lc7_filter_size=3,
            lc8_num_filters=512, lc8_filter_size=3,
            lc9_num_filters=512, lc9_filter_size=3, lc9_mp=True,
            lc10_num_filters=512, lc10_filter_size=3,
            lc11_num_filters=512, lc11_filter_size=3,
            lc12_num_filters=512, lc12_filter_size=3, lc12_mp=True,
            ld0_num_units=4096,
            ld1_num_units=4096
        )
        for key, value in kwargs.iteritems():
            if key.endswith('num_filters') or key.endswith('num_units'):
                kwargs[key] = value / reduction_factor
        return super(VggNet, self)._build_middle(l_in, **kwargs)


class AlexNet(ConvNet):
    """AlexNet, from ImageNet classification with deep convolutional neural networks (Krizhevsky et al., 2012).

    The parameters of the network can be overriden by passing in different keyword argument values.
    """
    def _build_middle(self, l_in, **kwargs):
        net_kwargs = dict(
            num_conv_layers=5, num_dense_layers=2,
            lc0_num_filters=48, lc0_filter_size=11, lc0_stride=4,
            lc0_mp=True, lm0_pool_size=3, lm0_stride=2,
            lc1_num_filters=128, lc1_filter_size=5,
            lc1_mp=True, lm1_pool_size=3, lm1_stride=2,
            lc2_num_filters=192, lc2_filter_size=3,
            lc3_num_filters=192, lc3_filter_size=3,
            lc4_num_filters=128, lc4_filter_size=3,
            lc4_mp=True, lm4_pool_size=3, lm4_stride=2,
            ld0_num_units=2048,
            ld1_num_units=2048
        )
        net_kwargs.update(kwargs)
        return super(AlexNet, self)._build_middle(l_in, **net_kwargs)


def _build_dense_plus_dropout(incoming_layer, num_units):
    return DropoutLayer(DenseLayer(incoming_layer, num_units=num_units, W=HeUniform(gain='relu')), p=0.5)


# A bit of Python magic to publish the available architectures.
ARCHITECTURE_NAME_TO_CLASS = dict(inspect.getmembers(sys.modules[__name__],
                                                     lambda obj: (inspect.isclass(obj) and
                                                                  issubclass(obj, AbstractModelBuilder) and
                                                                  obj != AbstractModelBuilder)))
