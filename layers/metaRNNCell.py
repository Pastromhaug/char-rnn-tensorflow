import tensorflow as tf
import numpy as np
import collections
from tensorflow.python.ops import variable_scope as vs

# _metaStateTuple = collections.namedtuple("metaStateTuple", ("h", "ctrl"))
# class MetaStateTuple(_metaStateTuple):
#     __slots__ = ()
#     @property
#     def dtype(self):
#         (h, ctrl) = self
#         if not h.dtype == ctrl.dtype:
#             raise TypeError("Inconsistent internal state: %s vs %s" %
#                           (str(h.dtype), str(ctrl.dtype)))
#         return h.dtype
#
#     def get_shape(): return
#


class MetaRNNCell(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, ctrl_units, cutoff=0.5, input_size=None, activation=tf.nn.relu):
    self._num_units = num_units
    self._ctrl_units = ctrl_units
    self._activation = activation
    self._cutoff = cutoff

  @property
  def state_size(self):
    return (self._num_units, self._ctrl_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    (state, ctrl) = state
    with vs.variable_scope(scope or type(self).__name__):
        with vs.variable_scope("CalcCtrl"):
            Ctrl = tf.nn.rnn_cell._linear([inputs, state, ctrl], bias=True, output_size=self._ctrl_units*2, scope="hCtrl")
            Ctrl = tf.nn.sigmoid(Ctrl)
            # hSumGate = hCtrl - 0.5
            # hSumGate = tf.ceil(hSumGate)
            # hSumGate = tf.reduce_sum(hSumGate)
            # hCtrl = tf.Print(hCtrl, [hSumGate], "hSumGate")
            # hCtrl_splt = tf.split(1, self._ctrl_units, hCtrl)
            # xCtrl = tf.nn.rnn_cell._linear([inputs, state, ctrl], bias=True, output_size=self._ctrl_units, scope="xCtrl")
            # xCtrl = tf.nn.sigmoid(xCtrl)
            # xSumGate = xCtrl - 0.5
            # xSumGate = tf.ceil(xSumGate)
            # xSumGate = tf.reduce_sum(xSumGate)
            # xCtrl = tf.Print(xCtrl, [xSumGate], "xSumGate")
            # xCtrl_splt = tf.split(1, self._ctrl_units, xCtrl)
        with va.variable_scope("hCalcUpdate"):
            hUpdate = tf.nn.rnn_cell._linear([state], bias=True, output_size=self._num_units, scope="hUpdate")
            xUpdate = tf.nn.rnn_cell._linear([inputs], bias=True, output_size=self._num_units, scope="xUpdate")

        states = tf.split(1, self._ctrl_units, state, name="SplittingState")
        state_out = []
        with vs.variable_scope("DoingStateUpdate"):

        ctrl_out = hCtrl + xCtrl
        state_out = tf.concat(1, state_out, name="PackingStateBackTogether")
        output = (state_out, ctrl_out)

    return state_out, output
