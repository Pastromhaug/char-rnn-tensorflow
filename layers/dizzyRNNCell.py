import tensorflow as tf
import numpy as np

class DizzyRNNCell(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""
  def __init__(self, num_units, rotationsA, rotationsB):
        self._num_units = num_units
        self._rotationsA = rotationsA
        self._rotationsB = rotationsB

  @property
  def state_size(self):
        return self._num_units

  @property
  def output_size(self):
        return self._num_units

  def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope or type(self).__name__):

            t_state = tf.transpose(state)

            state_out = state
            with tf.variable_scope("Do_Rotations_A"):
                for sparse_rot in self._rotationsA:
                    state_out = tf.sparse_tensor_dense_matmul(sparse_rot, state_out)

            input_out = inputs
            with tf.variable_scope("Do_Rotations_B"):
                for sparse_rot in self._rotationsB:
                    input_out = tf.sparse_tensor_dense_matmul(sparse_rot, input_out)

            state_out = tf.transpose(state_out)
            input_out = tf.transpose(input_out)

            with tf.variable_scope("Rotation_Bias"):
                bias = tf.get_variable(
                    "Bias", [self._num_units],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(dtype=tf.float32))

            output = tf.abs(state_out + input_out + bias)
        return output, output
