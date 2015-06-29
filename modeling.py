from math import ceil, sqrt
from warnings import warn
import itertools

import lasagne
from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import softmax
from lasagne.objectives import categorical_crossentropy
import numpy as np
import theano
import theano.tensor as T


class AbstractModelBuilder(object):
    """Builder of Lasagne models and iteration functions, with architecture-specific implementation by subclasses."""

    def __init__(self, dataset, output_dim, batch_size, chunk_size=0, verbose=False,
                 update_func_name='nesterov_momentum', learning_rate=0.01, update_func_kwargs=(),
                 loss_function=categorical_crossentropy, num_crops=0, crop_shape=None, mirror_crops=True):
        self.num_crops_with_mirrors = num_crops * 2 if mirror_crops else num_crops
        if self.num_crops_with_mirrors and batch_size % self.num_crops_with_mirrors != 0:
            raise ValueError('batch_size must be divisible by num_crops_with_mirrors')
        self.dataset = dataset
        self.output_dim = output_dim
        self.batch_size = batch_size
        # Note: this may result in the dataset being copied multiple times for small datasets
        self.chunk_size = chunk_size or self.dataset['training'][0].shape[0]
        self.verbose = verbose
        self.update_func_name = update_func_name
        self.learning_rate = learning_rate
        self.update_func_kwargs = dict(update_func_kwargs)
        self.loss_function = loss_function
        self.num_crops = num_crops
        self.crop_shape = crop_shape
        self.mirror_crops = mirror_crops
        if self.crop_shape:
            self.input_shape = (self.batch_size, dataset['training'][0].shape[1]) + self.crop_shape
        else:
            self.input_shape = (self.batch_size, ) + tuple(dataset['training'][0].shape[1:])

        effective_chunk_size = self.chunk_size * (self.num_crops_with_mirrors or 1)
        if effective_chunk_size < batch_size:
            raise ValueError('Effective chunk size %s must be greater than or equal to batch_size %s' %
                             (effective_chunk_size, self.batch_size))
        if effective_chunk_size % self.batch_size != 0:
            warn('Effective chunk size %s is not divisible by batch_size %s' % (effective_chunk_size, self.batch_size))

    def build(self, output_layer=None, **kwargs):
        """Build the model, returning the output layer and iteration functions."""
        if not output_layer:
            l_in = InputLayer(shape=self.input_shape)
            output_layer = DenseLayer(self._build_middle(l_in, **kwargs), num_units=self.output_dim,
                                      nonlinearity=softmax)
        training_iter = self._create_training_function(output_layer)
        validation_eval = self.create_eval_function('validation', output_layer)
        return output_layer, training_iter, validation_eval

    def _build_middle(self, l_in, **kwargs):
        raise NotImplementedError

    def _generate_crop_positions(self, instance_shape, deterministic):
        if deterministic:
            max_crop_start = [instance_shape[i + 1] - self.crop_shape[i] for i in (0, 1)]
            # Start with centre crop
            yield (max_crop_start[i] / 2 for i in (0, 1))
            # Generate as many evenly-distributed crops as we can (the centre crop may show up again)
            linspace_size = int(ceil(sqrt(self.num_crops - 1)))
            positions = itertools.product(*(np.linspace(0, max_crop_start[i], linspace_size, dtype='int32')
                                            for i in (0, 1)))
            for _ in xrange(self.num_crops - 1):
                yield positions.next()
        else:
            for _ in xrange(self.num_crops):
                yield (np.random.randint(0, instance_shape[i + 1] - self.crop_shape[i]) for i in (0, 1))

    def _transform_chunk(self, chunk_instances, chunk_labels, deterministic):
        if not self.num_crops:
            return chunk_instances, chunk_labels
        transformed_instances = []
        for instance in chunk_instances:
            for start_x, start_y in self._generate_crop_positions(instance.shape, deterministic):
                cropped = instance[:, start_x:(start_x + self.crop_shape[0]), start_y:(start_y + self.crop_shape[1])]
                transformed_instances.append(cropped)
                if self.mirror_crops:
                    transformed_instances.append(cropped[:, :, ::-1])
        return np.array(transformed_instances), chunk_labels.repeat(self.num_crops_with_mirrors)

    def _create_theano_function_runner(self, theano_function, instances_var, labels_var, instances, labels,
                                       deterministic):
        num_chunks = int(ceil(instances.shape[0] / float(self.chunk_size)))

        def run_theano_function():
            batch_results = []
            for c in xrange(num_chunks):
                chunk_slice = slice(c * self.chunk_size, (c + 1) * self.chunk_size)
                chunk_instances, chunk_labels = self._transform_chunk(instances[chunk_slice], labels[chunk_slice],
                                                                      deterministic)
                instances_var.set_value(chunk_instances)
                labels_var.set_value(chunk_labels)
                for b in xrange(chunk_instances.shape[0] // self.batch_size):
                    batch_result = theano_function(b)
                    batch_results.append(batch_result)
                    if self.verbose and not b:
                        print('...%s chunk %s batch %s results: %s' % (theano_function.name, c, b, batch_result))
                    if np.any(np.isnan(batch_result)):
                        raise OverflowError('NaN batch result detected at chunk %s, batch %s of %s function' %
                                            (c, b, theano_function.name))
            return np.mean(batch_results, axis=0)

        return run_theano_function

    def _create_training_function(self, output_layer):
        """Return a function that runs one iteration of model updates on the training data and returns the training
        loss."""
        instances_var, labels_var, batch_index_var, batch_instances_var, batch_labels_var = self._create_data_vars()
        loss_eval_function = self._create_loss_eval_func(
            output_layer, batch_instances_var, batch_labels_var, deterministic=False
        )
        theano_function = theano.function(
            [batch_index_var],
            loss_eval_function,
            updates=getattr(lasagne.updates, self.update_func_name)(loss_eval_function,
                                                                    lasagne.layers.get_all_params(output_layer),
                                                                    self.learning_rate,
                                                                    **self.update_func_kwargs),
            givens=self._create_batch_givens(instances_var, labels_var, batch_index_var, batch_instances_var,
                                             batch_labels_var),
            name='train'
        )
        return self._create_theano_function_runner(theano_function, instances_var, labels_var,
                                                   *self.dataset['training'], deterministic=False)

    def create_eval_function(self, subset_name, output_layer):
        """Return a function that returns the loss and accuracy of the given network on the given dataset's subset."""
        instances_var, labels_var, batch_index_var, batch_instances_var, batch_labels_var = self._create_data_vars()
        theano_function = theano.function(
            [batch_index_var],
            [
                self._create_loss_eval_func(output_layer, batch_instances_var, batch_labels_var, deterministic=True),
                self._create_accuracy_func(output_layer, batch_instances_var, batch_labels_var)
            ],
            givens=self._create_batch_givens(instances_var, labels_var, batch_index_var, batch_instances_var,
                                             batch_labels_var),
            name='eval_%s' % subset_name
        )
        return self._create_theano_function_runner(theano_function, instances_var, labels_var,
                                                   *self.dataset[subset_name], deterministic=True)

    def _create_data_vars(self):
        batch_instances_var_type = T.TensorType(theano.config.floatX, [False] * len(self.input_shape))
        return (lasagne.utils.shared_empty(dim=len(self.input_shape)),
                lasagne.utils.shared_empty(dim=1, dtype='int32'),
                T.iscalar('batch_index'),
                batch_instances_var_type('x'),
                T.ivector('y'))

    def _create_batch_givens(self, instances_var, labels_var, batch_index_var, batch_instances_var, batch_labels_var):
        batch_slice = slice(batch_index_var * self.batch_size, (batch_index_var + 1) * self.batch_size)
        return {batch_instances_var: instances_var[batch_slice], batch_labels_var: labels_var[batch_slice]}

    def _create_loss_eval_func(self, output_layer, batch_instances_var, batch_labels_var, deterministic):
        return lasagne.objectives.Objective(output_layer, loss_function=self.loss_function).get_loss(
            batch_instances_var, target=batch_labels_var, deterministic=deterministic
        )

    def _create_accuracy_func(self, output_layer, batch_instances_var, batch_labels_var):
        prob_output_var = lasagne.layers.get_output(output_layer, batch_instances_var, deterministic=True)
        if self.num_crops_with_mirrors:
            # Sum the probabilities for each instance's crops/mirrors and use that to make a prediction
            instances_per_batch = self.batch_size / self.num_crops_with_mirrors
            prob_output_var = T.sum(
                T.reshape(prob_output_var, (instances_per_batch, self.num_crops_with_mirrors, self.output_dim)),
                axis=1
            )
            sliced_batch_labels_var = T.reshape(batch_labels_var,
                                                (instances_per_batch, self.num_crops_with_mirrors))[:, 0]
        else:
            sliced_batch_labels_var = batch_labels_var

        # TODO: accuracy may not be accurate in case of weirdly-sized chunks/batches
        return T.mean(T.eq(T.argmax(prob_output_var, axis=1), sliced_batch_labels_var), dtype=theano.config.floatX)
