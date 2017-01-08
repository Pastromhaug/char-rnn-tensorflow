from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils.dataUtils import TextLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=100,
                       help='size of RNN hidden state')
    parser.add_argument('--ctrl_size', type=int, default=10,
                      help='num of gate controls for meta RNN')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='meta',
                       help='rnn, gru, lstm, or meta, inter, dizzy, block')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help="Determines percent of matrix in sparseRNNCell that is 'masked' to 0")
    parser.add_argument('--block_size', type=int, default=10,
                        help="dimensionality of each block in sparse block matrices for 'block' layer type")
    parser.add_argument('--tb_dir', type=str, default='bleh')
    parser.add_argument('--num_rots', type=int, default=5,
                        help="number of packed rotations for DizzyRNNCell")

    args = parser.parse_args()
    train(args)

def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl')) as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl')) as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars==data_loader.chars, "Data and loaded model disagree on character set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    model = Model(args)

    if args.model == 'sparse':
        mask = np.random.uniform(0,1,[2*args.rnn_size, args.rnn_size])
        print(np.sum(mask))
        mask = np.ceil(mask - args.sparsity)
        print("mask ratio: ", np.sum(mask)/(2*args.rnn_size*args.rnn_size))
    elif args.model == 'block':
        dim1 = args.rnn_size / args.block_size
        mask1 = np.random.uniform(0,1,[2*dim1, dim1])
        mask1 = np.ceil(mask1 - args.sparsity)
        block = np.ones([args.block_size, args.block_size])
        mask = []
        for i in range(2*dim1):
            row = []
            for j in range(dim1):
                row.append(block*mask1[i,j])
            mask.append(np.concatenate(row, axis=1))
        mask = np.concatenate(mask, axis=0)
        print("mask ratio: ", np.sum(mask)/(2*args.rnn_size*args.rnn_size))
        # print(mask)



    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        summary_writer = tf.summary.FileWriter('./dizzy_tensorflaz/' + args.tb_dir , sess.graph)
        # summary_writer = tf.summary.FileWriter('./rnn_tensorflaz/rnn7' , sess.graph)
        batch_num = 0
        test_batch_num = 0
        # restore model



        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            # state = sess.run(model.initial_state)
            for b in range(data_loader.num_batches):

                start = time.time()
                x, y = data_loader.next_batch()
                drop = 0
                if e >= 1:
                    drop=1
                feed = {model.input_data: x, model.targets: y, model.drop:drop, model.mask:mask}
                # for i, (c, h) in enumerate(model.initial_state):
                # for i, (c, h) in enumerate(model.initial_state):
                #     feed[c] = state[i].c
                #     feed[h] = state[i].h
                if (b%10) == 0:
                    test_batch_num += 1
                    test_loss, test_acc, test_summary_, probs_, elm_accuracy_ = sess.run([model.cost, model.accuracy, model.test_summary, model.probs, model.elm_accuracy], feed)
                    maxs = [np.max(i) for i in probs_]
                    # print(zip(maxs, elm_accuracy_[0]))
                    summary_writer.add_summary(test_summary_, test_batch_num)
                    end = time.time()
                    print("test loss: {:.3f}, test acc: {:.3f}".format(test_loss, test_acc))
                else:
                    batch_num += 1
                    train_loss, train_acc, train_summary_,  _ = sess.run([model.cost, model.accuracy, model.train_summary, model.train_op], feed)
                    summary_writer.add_summary(train_summary_, batch_num)
                    end = time.time()
                    print("{}/{} (epoch {}), train_loss = {:.3f}, train_acc = {:.3f}, time/batch = {:.3f}" \
                        .format(e * data_loader.num_batches + b,
                                args.num_epochs * data_loader.num_batches,
                                e, train_loss, train_acc,  end - start))
                    if (e * data_loader.num_batches + b) % args.save_every == 0\
                        or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                        print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    main()
