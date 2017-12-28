#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2017 Rahul Gupta, Soham Pal, Aditya Kanade, Shirish Shevade.
Indian Institute of Science.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
sys.path.append("/home/hirose/git/deepfix")

import numpy as np
import tensorflow as tf
import os, time, sys
from neural_net.load_data import load_data
from neural_net.model import model
from util.helpers import make_dir_if_not_exists, calculate_token_level_accuracy, calculate_localization_accuracy, calculate_repair_accuracy
from shutil import copyfile
# Parse command line arguments
import argparse

parser = argparse.ArgumentParser(description='Train a repairing seq2seq RNN.')
parser.add_argument('fold', help='Fold to train on', type=int)
parser.add_argument('batch_size', type=int)
parser.add_argument('embedding_dim', type=int)
parser.add_argument('memory_dim', type=int)
parser.add_argument('num_layers', type=int)
parser.add_argument('epochs', type=int)
parser.add_argument('resume_at', type=int)
parser.add_argument('resume_epoch', type=int)
parser.add_argument('resume_training_minibatch', type=int)
parser.add_argument('rnn_cell', help='One of RNN, LSTM or GRU.')
parser.add_argument('ckpt_every', help='How often to checkpoint', type=int)
parser.add_argument('dropout', help='Probability to use for dropout', type=float)
parser.add_argument('--bidirectional', action="store_true", help="Use a bidirectional encoder")
parser.add_argument('--checkpoints_directory', help='Checkpoints directory', default=None)
parser.add_argument('-v', '--vram', help='Fraction of GPU memory to use', type=float, default=1.0)

args = parser.parse_args()

# Default checkpoints directory
if args.checkpoints_directory is None:
    checkpoints_directory = os.path.join('checkpoints', 'fold_%d' % args.fold)
else:
    checkpoints_directory = args.checkpoints_directory

# Make checkpoint directories
make_dir_if_not_exists(checkpoints_directory)
make_dir_if_not_exists(os.path.join(checkpoints_directory, 'best'))

# Print options
print 'Checkpoint every:', args.ckpt_every
print 'Batch size:', args.batch_size
print 'Embedding dim:', args.embedding_dim
print 'Memory dim:', args.memory_dim
print 'Layers:', args.num_layers
print 'Epochs:', args.epochs
print 'Resume at:', args.resume_at
print 'Resume epoch:', args.resume_epoch
print 'Resume training minibatch:', args.resume_training_minibatch
print 'RNN cell:', args.rnn_cell
print 'Bidirectional:', args.bidirectional
sys.stdout.flush()

# Load data
dataset = load_data(args.fold)
train_x, train_y, valid_x, valid_y, test_x, test_y = dataset.get_data()
dictionary = dataset.get_dictionary()

num_train      = len(train_x)
num_validation = len(valid_x)
num_test       = len(test_x)

print 'Training:', num_train, 'examples'
print 'Validation:', num_validation, 'examples'
print 'Test:', num_test, 'examples'
sys.stdout.flush()

in_seq_length  = np.shape(train_x)[1]
out_seq_length = np.shape(train_y)[1]
vocab_size     = len(dictionary)

print 'In sequence length:', in_seq_length
print 'Out sequence length:', out_seq_length
print 'Vocabulary size:', vocab_size
sys.stdout.flush()

# Restrict VRAM usage
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.vram)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Build the network
seq2seq = model(in_seq_length, out_seq_length, vocab_size, rnn_cell=args.rnn_cell,
                memory_dim=args.memory_dim, num_layers=args.num_layers, dropout=args.dropout,
                embedding_dim=args.embedding_dim, bidirectional=args.bidirectional)

# Build a session and load parameters
sess = tf.Session()

if args.resume_at == 0:
    sess.run(tf.initialize_all_variables())
else:
    seq2seq.load_parameters(sess, os.path.join(checkpoints_directory, 'saved-model-attn-' + str(args.resume_at)))


# Train
step = args.resume_at
resume_training_minibatch = args.resume_training_minibatch
best_test_repair = 0

for t in range(args.resume_epoch, args.epochs):
    # Training
    start_time = time.time()
    train_loss = []
    
    for i in range(resume_training_minibatch, num_train/args.batch_size):
        start  = i * args.batch_size
        end    = (i+1) * args.batch_size
        f_loss = seq2seq.train_step(sess, train_x[start:end], train_y[start:end])
        train_loss.append(f_loss)
        
        # Print progress
        step += 1
        
        print "Step: %d\tEpoch: %g\tLoss: %g" % (step, t + float(i+1)/(num_train/args.batch_size), train_loss[-1])
        sys.stdout.flush()

        # Checkpoint
        if step % args.ckpt_every == 0:
            seq2seq.save_parameters(sess, os.path.join(checkpoints_directory, 'saved-model-attn'), global_step=step)
            print "[Checkpoint] Checkpointed at Epoch %d, Minibatch %d." % (t, i)
            sys.stdout.flush()
        
    train_loss = np.mean(train_loss)
    resume_training_minibatch = 0

    # Checkpoint before going into validation/testing
    if step % args.ckpt_every != 0:
        seq2seq.save_parameters(sess, os.path.join(checkpoints_directory, 'saved-model-attn'), global_step=step)
        print "[Checkpoint] Checkpointed at Epoch %d, Minibatch %d." % (t+1, 0)
        sys.stdout.flush()
    
    print "End of Epoch: %d" % (t+1)
    print "[Training] Loss: %g" % (train_loss)
    sys.stdout.flush()

    # Validation
    valid_loss, valid_token, valid_local, valid_repair = [], [], [], []
    
    for i in range(num_validation/args.batch_size):
        start  = i * args.batch_size
        end    = (i+1) * args.batch_size
        f_loss, y_hat = seq2seq.validate_step(sess, valid_x[start:end], valid_y[start:end])
        valid_loss.append(f_loss)
        
        valid_token.append(calculate_token_level_accuracy(valid_y[start:end], y_hat))
        valid_local.append(calculate_localization_accuracy(valid_y[start:end], y_hat, dictionary))
        valid_repair.append(calculate_repair_accuracy(valid_y[start:end], y_hat))
        
    valid_loss, valid_token, valid_local, valid_repair = np.mean(valid_loss), np.mean(valid_token), np.mean(valid_local), np.mean(valid_repair)
    
    # Print epoch step and validation information
    print "[Validation] Loss: %g Token: %g Localization: %g Repair: %g" % (valid_loss, valid_token, valid_local, valid_repair)
    sys.stdout.flush()

    # Testing
    test_loss, test_token, test_local, test_repair = [], [], [], []
    
    for i in range(num_test/args.batch_size):
        start  = i * args.batch_size
        end    = (i+1) * args.batch_size
        f_loss, y_hat = seq2seq.validate_step(sess, test_x[start:end], test_y[start:end])
        test_loss.append(f_loss)
        
        test_token.append(calculate_token_level_accuracy(test_y[start:end], y_hat))
        test_local.append(calculate_localization_accuracy(test_y[start:end], y_hat, dictionary))
        test_repair.append(calculate_repair_accuracy(test_y[start:end], y_hat))
        
    test_loss, test_token, test_local, test_repair = np.mean(test_loss), np.mean(test_token), np.mean(test_local), np.mean(test_repair)

    print "[Test] Loss: %g Token: %g Localization: %g Repair: %g" % (test_loss, test_token, test_local, test_repair)
    sys.stdout.flush()

    if test_repair >= best_test_repair:
        best_test_repair = test_repair
        os.system('rm %s' % os.path.join(checkpoints_directory, 'best', '*'))
        copyfile(os.path.join(checkpoints_directory, 'saved-model-attn-%d' % step), os.path.join(os.path.join(checkpoints_directory, 'best'), 'saved-model-attn-%d' % step))
        print "[Best Checkpoint] Checkpointed at Epoch %d, Minibatch %d." % (t+1, 0)
        sys.stdout.flush()

    print "[Time] Took %g minutes to run." % ((time.time() - start_time)/60)
    sys.stdout.flush()

sess.close()
