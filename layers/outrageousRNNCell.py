import tensorflow as tf
import numpy as np

class OutrageousRNNCell(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, softmax_size, targets, input_size=None, step=tf.constant(5), epoch=None,):

    self._num_units = num_units
    self._step = step
    self._epoch = epoch
    self._softmax_size = softmax_size
    self._targets = targets

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return (self._softmax_size, self._softmax_size)

  def __call__(self, inputs, state, scope=None):
    # print("state")
    # print(state)
    """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
    with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
        output = tf.tanh(tf.nn.rnn_cell._linear([inputs,state], self._num_units, True, scope="OutrageousLinear"))
        # print("output")
        # print(output)
        # output = tf.reshape(tf.concat(1, output), [-1,self._num_units], name="OutrageousReshapeOutputs")
        # print("output reshaped")
        # print(output)
        logits = tf.nn.rnn_cell._linear([output], self._softmax_size, True, scope="OutrageousSoftmax")
        # print("logits")
        # print(logits)
        probs = tf.nn.softmax(logits)
        # print("probs")
        # print(probs)
        maxi = tf.reduce_max(probs, axis=1)
        # print("maxi")
        # print(maxi)
        # for j in range(self._epoch/self._step):
        #     output = tf.tanh(tf.nn.rnn_cell._linear([inputs,state], self._num_units, True, name="OutrageousLinear"+str(j)))
        #     soft = tf.nn.rnn_cell._linear([output], self._softmax_size, True, name="OutrageousSoftmax")
        # output = tf.Print(output, [maxi], message="maxi")
    return (logits, probs), output

def nextState(inputs,outputs, marker=""):
    output = tf.tanh(tf.nn.rnn_cell._linear(inputs, outputs, True, name="OutrageousLinear"+marker))
