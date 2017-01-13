import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest

class OutrageousRNNCell(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, softmax_size,input_size=None, step=tf.constant(5), epoch=None, cutoff=0.7):

    self._num_units = num_units
    self._step = step
    self._epoch = epoch
    self._softmax_size = softmax_size
    self._stats = [tf.constant(0, dtype=tf.float32) for i in range(5)]
    self._cutoff = cutoff

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return (self._softmax_size, self._softmax_size)

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
    with tf.variable_scope(scope or type(self).__name__):
        output = tf.tanh(tf.nn.rnn_cell._linear([inputs,state], self._num_units, True, scope="OutrageousLinearTranh"))
        output = tf.nn.relu(tf.nn.rnn_cell._linear([output,state], self._num_units, True, scope="OutrageousLinearReLU"))
        with tf.variable_scope("Softmax"):
            logits = tf.nn.rnn_cell._linear([output], self._softmax_size, True, scope="OutrageousSoftmax")
        probs = tf.nn.softmax(logits)
        maxi = tf.reduce_max(probs, axis=1)
        cutoff = 0.7
        binary = 1-tf.expand_dims(tf.ceil(maxi - self._cutoff),1)

        i = tf.cast(tf.floor(self._epoch/self._step), tf.int32)
        for j in range(5):
            cond = tf.less(j,i)
            # cond = tf.Print(cond, [cond], message="cond")
            def f1():
                out1 = binary*output
                out1 = tf.nn.relu(identityLinear([out1], self._num_units, True, scope="OutrageousLinear"+str(j)))
                out2 = (1-binary)*output
                out = out1 + out2
                with tf.variable_scope("Softmax") as scope:
                    scope.reuse_variables()
                    log = tf.nn.rnn_cell._linear([out], self._softmax_size, True, scope="OutrageousSoftmax")
                p = tf.nn.softmax(log)
                m = tf.reduce_max(p, axis=1)
                b = tf.expand_dims(tf.ceil(maxi - self._cutoff),1)
                # out = tf.Print(output, [tf.constant(j)], message="f1")
                return (out, log, p, m, b)
                # return (out, logits, probs, maxi)
            def f2():
                # out = tf.Print(output, [tf.constant(j)], message="f2")
                return (output, logits, probs, maxi, binary*0)
            (output, logits, probs, maxi, binary) = tf.cond(cond, f1, f2)
            probs = tf.nn.softmax(logits)
            maxi = tf.reduce_max(probs, axis=1)
            self._stats[j] = self._stats[j]+tf.reduce_sum(binary)

            # print(output)
        # print("output")
        # logits = tf.Print(logits, self._stats, message="stats_out")
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
