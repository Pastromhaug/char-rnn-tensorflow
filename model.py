import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
# from layers.metaRNNCell import MetaRNNCell
from layers.passRNNCell import PassRNNCell
from layers.hwyRNNCell import HwyRNNCell
from layers.sparseRNNCell import SparseRNNCell
from layers.interRNNCell import InterRNNCell

import numpy as np

class Model():
    def __init__(self, args, infer=False):

        with tf.variable_scope('PlaceHolders'):
            self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length], name="input_data")
            self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length], name="targets")
            self.drop = tf.placeholder(tf.int32, name="drop")
            self.mask = tf.placeholder(tf.float32, [2*args.rnn_size, args.rnn_size], name="mask")
            targets = tf.reshape(self.targets, [-1])
            embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
            inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]


        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        elif args.model =='meta':
            cell_fn = MetaRNNCell
        elif args.model =='pass':
            cell_fn = PassRNNCell
        elif args.model == 'hwy':
            cell_fn = HwyRNNCell
        elif args.model == 'sparse':
            cell_fn = SparseRNNCell
        elif args.model == 'inter':
            cell_fn = InterRNNCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        if args.model == 'meta':
            cell = cell_fn(args.rnn_size, args.ctrl_size)
        elif args.model == 'hwy':
            cell = cell_fn(args.rnn_size, drop=self.drop)
        elif args.model == 'sparse':
            cell = cell_fn(args.rnn_size, mask=self.mask, sparsity=args.sparsity)
        elif args.model == 'inter':
            cell = cell_fn(args.rnn_size, mask=self.mask, sparsity=args.sparsity, drop=self.drop)
        else:
            cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)


        with tf.variable_scope("InitialState"):
            self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('Seq2Seq'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            def loop(prev, _):
                prev = tf.matmul(prev, softmax_w) + softmax_b
                prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
                return tf.nn.embedding_lookup(embedding, prev_symbol)
            outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='CallingSeq2Seq')

        with tf.variable_scope('ProcessingRNNOutputs'):
            output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size], name="ReshapeOutputs")
            with tf.variable_scope("ExctactProbabilities"):
                self.logits = tf.matmul(output, softmax_w) + softmax_b
                self.probs = tf.nn.softmax(self.logits)

            with tf.variable_scope("CalculateLoss"):
                loss = seq2seq.sequence_loss_by_example([self.logits],
                        [targets],
                        [tf.ones([args.batch_size * args.seq_length])],
                        args.vocab_size)
                self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

            self.final_state = last_state

            with tf.variable_scope("ApplyingGradients"):
                self.lr = tf.Variable(0.0, trainable=False)
                tvars = tf.trainable_variables()
                for t in tvars:
                    print t
                grads = tf.gradients(self.cost, tvars)
                grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)
                optimizer = tf.train.AdamOptimizer(self.lr)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars))

            with tf.variable_scope("CalculateAccuracy"):
                predictions = tf.argmax(self.probs, 1)
                accuracy = [tf.equal(tf.cast(predictions, tf.int32), targets)]
                self.accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

        cost_summary = tf.summary.scalar('train_loss', self.cost)
        test_cost_summary = tf.summary.scalar('test_loss', self.cost)
        acc_summary = tf.summary.scalar('train_acc', self.accuracy)
        test_acc_summary = tf.summary.scalar('test_acc', self.accuracy)
        self.train_summary = tf.summary.merge([cost_summary, acc_summary], name="MergedTrainSummaries")
        self.test_summary = tf.summary.merge([test_cost_summary, test_acc_summary], name="MergedTestSummaries")

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        print("num")
        print(num)
        print("chars")
        print(chars)
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
