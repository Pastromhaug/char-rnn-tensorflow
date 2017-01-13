import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
# from layers.metaRNNCell import MetaRNNCell
from layers.outrageousRNNCell import OutrageousRNNCell

import numpy as np

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        with tf.variable_scope('PlaceHolders'):
            self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length], name="input_data")
            self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length], name="targets")
            self.drop = tf.placeholder(tf.int32, name="drop")
            self.mask = tf.placeholder(tf.float32, [2*args.rnn_size, args.rnn_size], name="mask")
            self.step = tf.placeholder(tf.float32, name="outrageous_step")
            self.epoch = tf.placeholder(tf.float32, name="epoch_num")
            targets = tf.reshape(self.targets, [-1])
            embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
            inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        if args.model == 'out':
            self.cell = OutrageousRNNCell(num_units=args.rnn_size, softmax_size=args.vocab_size,
                         step=self.step, epoch=self.epoch, cutoff=args.cutoff)

        with tf.variable_scope("InitialState"):
            self.initial_state = self.cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('Seq2Seq'):
            outputs, final_state = tf.nn.rnn(self.cell, inputs, self.initial_state, scope='CallingSeq2Seq')
            print("final state")
            print(final_state)
            print("stats")
            stats  = self.cell._stats
            print(stats)
            logits, probs = zip(*outputs)
            self.logits = tf.reshape(tf.concat(1, logits), [-1,args.vocab_size], name="ReshapeLogits")
            self.probs = tf.reshape(tf.concat(1, probs), [-1,args.vocab_size], name="ReshapeProbs")

        with tf.variable_scope('ProcessingRNNOutputs'):
            with tf.variable_scope("CalculateLoss"):
                loss = seq2seq.sequence_loss_by_example([self.logits],
                        [targets],
                        [tf.ones([args.batch_size * args.seq_length])],
                        args.vocab_size, name="CallingSeq2Seq")
                print("loss")
                print(loss)

                su = tf.reduce_sum(loss, name="reduce_sum")
                tot = tf.mul(tf.cast(args.seq_length, tf.float32),tf.cast(args.batch_size, tf.float32), name="_mul")
                self.cost = tf.truediv(su, tot, name="_div")

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
                self.predictions = tf.argmax(self.probs, 1)
                self.elm_accuracy = [tf.equal(tf.cast(self.predictions, tf.int32), targets)]
                self.accuracy = tf.reduce_mean(tf.cast(self.elm_accuracy, tf.float32))

        total = args.batch_size * args.seq_length
        train_stats_summaries=[]
        test_stats_summaries=[]
        for i,stat in enumerate(list(stats)):
            print(stat)
            stat = stat/total
            # stat = tf.Print(stat, [stat], message="stat"+str(i))
            train_summ = tf.summary.scalar('train_stats_level_'+str(i), stat)
            train_stats_summaries.append(train_summ)
            test_summ = tf.summary.scalar('test_stats_level_'+str(i), stat)
            test_stats_summaries.append(test_summ)

        cost_summary = tf.summary.scalar('train_loss', self.cost)
        test_cost_summary = tf.summary.scalar('test_loss', self.cost)
        acc_summary = tf.summary.scalar('train_acc', self.accuracy)
        test_acc_summary = tf.summary.scalar('test_acc', self.accuracy)
        train_sums = [cost_summary, acc_summary]+train_stats_summaries
        test_sums = [test_cost_summary, test_acc_summary]+test_stats_summaries
        self.train_summary = tf.summary.merge(train_sums, name="MergedTrainSummaries")
        self.test_summary = tf.summary.merge(test_sums, name="MergedTestSummaries")

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
