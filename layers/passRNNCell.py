import tensorflow as tf
import numpy as np

class PassRNNCell(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, input_size=None):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
    with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
        # state = tf.Print(state, [tf.reduce_mean(state*state)], message="state")
        update = tf.nn.relu(tf.nn.rnn_cell._linear([inputs, state], self._num_units, True))
        # update = tf.Print(update, [tf.reduce_mean(update*update)], message="update")
        mask = 1 - tf.ceil(tf.minimum(update, 1))
        output = state*mask + update
        # output = tf.Print(output, [tf.reduce_mean(output*output)], message="state")
    return output, output
