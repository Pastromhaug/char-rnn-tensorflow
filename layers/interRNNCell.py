import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops

class InterRNNCell(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, mask, sparsity, drop=0, input_size=None, activation=tf.tanh):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._mask = mask
    self._sparsity = sparsity
    self._drop = drop

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
    with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
      output = self._activation(_linear([inputs, state], self._num_units, True,
                                    self._mask, self._sparsity, drop=self._drop))
    return output, output


def _linear(args, output_size, bias, mask, sparsity, drop=0, bias_start=0.0, scope=None):
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
    # mask = tf.Print(mask, [tf.reduce_sum(mask)], message="mask_sum")
    gate = tf.get_variable("Gate", [total_arg_size, 1], dtype=tf.float32)
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size], dtype=dtype)
    matrix = matrix/(1-sparsity)
    matrixA = matrix * mask
    matrixB = matrix * (1-mask)
    resA = math_ops.matmul(array_ops.concat(1, args), matrixA)
    resB = math_ops.matmul(array_ops.concat(1, args), matrixB)
    g = math_ops.matmul(array_ops.concat(1, args), gate)
    # g = tf.Print(g, [g], message="g")
    g = tf.sigmoid(g)
    cond = tf.equal(drop, 0)
    def f1():
        ret = g
        # ret = tf.Print(ret, [ret], message="f1")
        return ret
    def f2():
        ret =  tf.round(g)
        # ret = tf.Print(ret, [tf.reduce_sum(ret)], message="num_activations")
        # ret = tf.Print(ret, [ret], message="f2")
        return ret
    g = tf.cond(cond, f1, f2)
    # print("p")
    # print(g)
    # print(resA)
    # print(resB)
    res = resA*g + (1-g)*resB
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=init_ops.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term
