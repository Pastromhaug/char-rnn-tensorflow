import tensorflow as tf
import numpy as np

class HwyRNNCell(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, input_size=None, drop=tf.constant(0)):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._drop = drop

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
        update = tf.nn.relu(tf.nn.rnn_cell._linear([inputs, state], self._num_units*2, True))
        (update, gate) = tf.split(1, 2, update)
        update = tf.tanh(update)
        gate = tf.sigmoid(gate)

        cond = tf.equal(self._drop, 0)
        def f1():
            ret = gate
            # ret = tf.Print(ret, [ret], message="f1")
            return ret
        def f2():
            ret =  tf.round(gate)
            ret = tf.Print(ret, [tf.reduce_sum(ret)], message="num_activations")
            # ret = tf.Print(ret, [ret], message="f2")
            return ret
        gate = tf.cond(cond, f1, f2)
        output = state*gate + (1-gate)*update
    return output, output
