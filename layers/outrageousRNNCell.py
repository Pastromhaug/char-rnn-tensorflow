import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest

class OutrageousRNNCell(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, softmax_size,input_size=None, step=tf.constant(5), epoch=None,):

    self._num_units = num_units
    self._step = step
    self._epoch = epoch
    self._softmax_size = softmax_size
    self._stats = [tf.constant(0, dtype=tf.float32) for i in range(5)]

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
        output = tf.tanh(tf.nn.rnn_cell._linear([inputs,state], self._num_units, True, scope="OutrageousLinearTranh"))
        output = tf.nn.relu(tf.nn.rnn_cell._linear([output,state], self._num_units, True, scope="OutrageousLinearReLU"))
        with tf.variable_scope("Softmax"):
            logits = tf.nn.rnn_cell._linear([output], self._softmax_size, True, scope="OutrageousSoftmax")
        probs = tf.nn.softmax(logits)
        maxi = tf.reduce_max(probs, axis=1)
        cutoff = 0.7


        i = tf.floor(self._epoch/self._step)
        for j in range(5):
            cond = j < i
            def f1():
                binary = tf.expand_dims(tf.ceil(maxi - cutoff),1)
                new_output = (1-binary)*output
                self._stats[j] = self._stats[j]+tf.reduce_sum(1-binary)
                old_output = binary*output
                new_output = tf.nn.relu(identityLinear([new_output], self._num_units, True, scope="OutrageousLinear"+str(j)))
                old_logits = binary*logits
                with tf.variable_scope("Softmax") as scope:
                    scope.reuse_variables()
                    new_logits = tf.nn.rnn_cell._linear([new_output], self._softmax_size, True, scope="OutrageousSoftmax")
                log = old_logits + new_logits
                out = old_output + new_output
                # out = tf.Print(out, [tf.constant(j)], message="j")
                return (out, log)
            def f2():
                (out, log) = (output, logits)
                return (out, log)
            (output, logits) = tf.cond(cond, f1, f2)
            probs = tf.nn.softmax(logits)
            maxi = tf.reduce_max(probs, axis=1)

    return (logits, probs), output


def identityLinear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable(
        "Matrix", dtype=dtype, initializer=np.identity(output_size, dtype=np.float32))
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(array_ops.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=tf.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term
