from math import ceil
from warnings import warn

import lasagne
from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import softmax
from lasagne.objectives import categorical_crossentropy
import theano
import theano.tensor as T


class AbstractModelBuilder(object):
    """Builder of Lasagne models and iteration functions, with architecture-specific implementation by subclasses."""

    def __init__(self, dataset, output_dim, batch_size, training_chunk_size=0, learning_rate=0.01,
                 momentum=0.9, loss_function=categorical_crossentropy):
        if training_chunk_size and training_chunk_size < batch_size:
            raise ValueError('Chunk size must be greater than or equal to batch_size')
        self.dataset = dataset
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.input_shape = (self.batch_size, ) + tuple(dataset['training'][0].shape[1:])
        self.training_chunk_size = training_chunk_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.loss_function = loss_function

    def build(self, **kwargs):
        """Build the model, returning the output layer."""
        l_in = InputLayer(shape=self.input_shape)
        output_layer = DenseLayer(self._build_middle(l_in, **kwargs), num_units=self.output_dim, nonlinearity=softmax)
        training_iter = self._create_training_function(output_layer)
        validation_eval = self._create_eval_function('validation', output_layer)
        return output_layer, training_iter, validation_eval

    def _build_middle(self, l_in, **kwargs):
        raise NotImplementedError

    def _create_training_function(self, output_layer):
        """Return a function that runs one iteration of model updates on the training data and returns the training
        loss."""
        instances_var = lasagne.utils.shared_empty(dim=len(self.input_shape))
        labels_var = lasagne.utils.shared_empty(dim=1, dtype='int32')
        theano_function = self._create_theano_training_function(instances_var, labels_var, output_layer)

        instances, labels = self.dataset['training']
        if self.training_chunk_size:
            if not instances.shape[0] % self.training_chunk_size == 0:
                warn('Number of training instances is not divisable by chunk_size')
            num_chunks = int(ceil(instances.shape[0] / self.training_chunk_size))
            num_batches = self.training_chunk_size // self.batch_size

            def run_theano_function():
                sum_loss = 0.0
                for c in xrange(num_chunks):
                    chunk_slice = slice(c * self.training_chunk_size, (c + 1) * self.training_chunk_size)
                    instances_var.set_value(instances[chunk_slice])
                    labels_var.set_value(labels[chunk_slice])
                    for b in xrange(num_batches):
                        sum_loss += theano_function(b)
                return sum_loss / (num_batches * num_chunks)
        else:
            instances_var.set_value(instances)
            labels_var.set_value(labels)
            num_batches = instances.shape[0] // self.batch_size
            run_theano_function = lambda: sum(theano_function(b) for b in xrange(num_batches)) / num_batches

        return run_theano_function

    def _create_theano_training_function(self, instances_var, labels_var, output_layer):
        """Return a theano.function that takes a batch index input, updates the network, and returns the training
        loss."""
        batch_index_var, batch_instances_var, batch_labels_var = self._create_batch_vars()
        loss_eval_function = self._create_loss_eval_func(
            output_layer, batch_instances_var, batch_labels_var, deterministic=False
        )
        return theano.function(
            [batch_index_var],
            loss_eval_function,
            updates=lasagne.updates.nesterov_momentum(
                loss_eval_function,
                lasagne.layers.get_all_params(output_layer),
                self.learning_rate,
                self.momentum
            ),
            givens=self._create_batch_givens(instances_var, labels_var, batch_index_var, batch_instances_var,
                                             batch_labels_var)
        )

    def _create_eval_function(self, subset_name, output_layer):
        """Return a function that returns the loss and accuracy of the given network on the given dataset's subset."""
        instances_var, labels_var = (theano.shared(_) for _ in self.dataset[subset_name])
        theano_function = self._create_theano_eval_function(instances_var, labels_var, output_layer)
        num_validation_batches = self.dataset[subset_name][0].shape[0] // self.batch_size

        def run_theano_function():
            sum_losses = 0.0
            sum_accuracies = 0.0
            for b in xrange(num_validation_batches):
                batch_loss, batch_accuracy = theano_function(b)
                sum_losses += batch_loss
                sum_accuracies += batch_accuracy
            return sum_losses / num_validation_batches, sum_accuracies / num_validation_batches

        return run_theano_function

    def _create_theano_eval_function(self, instances_var, labels_var, output_layer):
        """Return a theano.function that takes a batch index input, and returns a tuple of the loss and accuracy."""
        batch_index_var, batch_instances_var, batch_labels_var = self._create_batch_vars()
        accuracy_eval_func = T.mean(
            T.eq(T.argmax(lasagne.layers.get_output(output_layer, batch_instances_var, deterministic=True), axis=1),
                 batch_labels_var),
            dtype=theano.config.floatX
        )
        return theano.function(
            [batch_index_var],
            [
                self._create_loss_eval_func(output_layer, batch_instances_var, batch_labels_var, deterministic=True),
                accuracy_eval_func
            ],
            givens=self._create_batch_givens(instances_var, labels_var, batch_index_var, batch_instances_var,
                                             batch_labels_var)
        )

    def _create_batch_vars(self):
        # TODO: is there a better way of doing this?
        input_dim_to_tensor_type = {2: T.matrix, 4: T.tensor4}
        return T.iscalar('batch_index'), input_dim_to_tensor_type[len(self.input_shape)]('x'), T.ivector('y')

    def _create_batch_givens(self, instances_var, labels_var, batch_index_var, batch_instances_var, batch_labels_var):
        batch_slice = slice(batch_index_var * self.batch_size, (batch_index_var + 1) * self.batch_size)
        return {batch_instances_var: instances_var[batch_slice], batch_labels_var: labels_var[batch_slice]}

    def _create_loss_eval_func(self, output_layer, batch_instances_var, batch_labels_var, deterministic):
        return lasagne.objectives.Objective(output_layer, loss_function=self.loss_function).get_loss(
            batch_instances_var, target=batch_labels_var, deterministic=deterministic
        )
