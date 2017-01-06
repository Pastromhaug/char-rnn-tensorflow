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
        with vs.variable_scope("CalculatingCtrl"):
            hCtrl = tf.nn.rnn_cell._linear([inputs, state, ctrl], bias=True, output_size=self._ctrl_units, scope="hCtrl")
            hCtrl = tf.nn.sigmoid(hCtrl)
            #
            # hSumGate = hCtrl - 0.5
            # hSumGate = tf.ceil(hSumGate)
            # hSumGate = tf.reduce_sum(hSumGate)
            # hCtrl = tf.Print(hCtrl, [hSumGate], "hSumGate")

            hCtrl_splt = tf.split(1, self._ctrl_units, hCtrl)
            xCtrl = tf.nn.rnn_cell._linear([inputs, state, ctrl], bias=True, output_size=self._ctrl_units, scope="xCtrl")
            xCtrl = tf.nn.sigmoid(xCtrl)

            # xSumGate = xCtrl - 0.5
            # xSumGate = tf.ceil(xSumGate)
            # xSumGate = tf.reduce_sum(xSumGate)
            # xCtrl = tf.Print(xCtrl, [xSumGate], "xSumGate")


            xCtrl_splt = tf.split(1, self._ctrl_units, xCtrl)

        with vs.variable_scope("BuildingStateUpdateMatrices"):
            hShape = [self._num_units, self._num_units]
            hMatrix = vs.get_variable("hCols", hShape, dtype=tf.float32)
            hMatrix = tf.split(1, self._ctrl_units, hMatrix)
            input_shape = inputs.get_shape().as_list()
            xShape = [self._num_units, input_shape[1]]
            xMatrix = vs.get_variable("xCols", xShape, dtype=tf.float32)
            xMatrix = tf.split(1, self._ctrl_units, xMatrix)

        states = tf.split(1, self._ctrl_units, state, name="SplittingState")

        state_out = []
        with vs.variable_scope("DoingStateUpdate"):
            for i in range(self._ctrl_units):
                hGate = tf.squeeze(hCtrl_splt[i], name="hGate")
                hCond = tf.less(hGate, 0.5, name="hCondFunc")
                with vs.variable_scope("hF1"):
                    ret = states[i]
                    # ret = tf.Print(ret, [ret], str(i)+"hF1")
                    def hF1(): return ret
                with vs.variable_scope("hF2"):
                    def hF2():
                        # ret = tf.matmul(tf.mul(state,hGate), hMatrix[i])
                        ret = tf.nn.rnn_cell._linear([state], bias=True, output_size=self._ctrl_units, scope="hLinear" + str(i))
                        ret = tf.mul(ret, hGate)
                        # ret = tf.Print(ret, [ret], str(i)+"hF2")
                        return ret
                out_h = tf.cond(hCond, hF1, hF2, "hCond")
                xGate = tf.squeeze(xCtrl_splt[i], name="xGate")
                xCond = tf.less(xGate, 0.5, name="xCondFunc")
                with vs.variable_scope("xF1"):
                    def xF1(): return tf.zeros(shape=[1,self._ctrl_units], dtype=tf.float32)
                with vs.variable_scope("xF2"):
                    def xF2():
                        ret = tf.nn.rnn_cell._linear([state], bias=True, output_size=self._ctrl_units, scope="xLinear" + str(i))
                        ret = tf.mul(ret, hGate)
                        return ret
                out_x = tf.cond(xCond, xF1, xF2, "xCond")
                state_out.append(out_h + out_x)







        ctrl_out = hCtrl + xCtrl
        state_out = tf.concat(1, state_out, name="PackingStateBackTogether")
        output = (state_out, ctrl_out)
    return state_out, output
