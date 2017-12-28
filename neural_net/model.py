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

import numpy as np
import tensorflow as tf
import custom_seq2seq

class model:
    def _new_RNN_cell(self, memory_dim):
        if self.rnn_cell == 'LSTM':
            constituent_cell = tf.nn.rnn_cell.BasicLSTMCell(memory_dim)
        elif self.rnn_cell == 'GRU':
            constituent_cell = tf.nn.rnn_cell.GRUCell(memory_dim)
        elif self.rnn_cell == 'RNN':
            constituent_cell = tf.nn.rnn_cell.BasicRNNCell(memory_dim)
        else:
            raise Exception('unsupported rnn cell type: %s' % self.rnn_cell)
        
        if self.dropout != 0:
            constituent_cell = tf.nn.rnn_cell.DropoutWrapper(constituent_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)

        if self.num_layers > 1:
            return tf.nn.rnn_cell.MultiRNNCell([constituent_cell] * self.num_layers)
        
        return constituent_cell
    
    def __init__(self, in_seq_length, out_seq_length, vocabulary_size, rnn_cell="GRU", memory_dim=300, num_layers=4, dropout=0.2, embedding_dim=50, bidirectional=False, trainable=True):
        self.in_seq_length   = in_seq_length
        self.out_seq_length  = out_seq_length
        self.vocabulary_size = vocabulary_size
        self.rnn_cell        = rnn_cell
        self.memory_dim      = memory_dim
        self.num_layers      = num_layers
        self.dropout         = dropout
        self.embedding_dim   = embedding_dim
        self.bidirectional   = bidirectional
        self.trainable       = trainable

        self.encoder_input = [tf.placeholder(tf.int32, shape=(None,), name="inp%i" % t) for t in range(in_seq_length)]
        self.labels        = [tf.placeholder(tf.int32, shape=(None,), name="labels%i" % t) for t in range(out_seq_length)]
        self.weights       = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in self.labels]

        self.decoder_input = [tf.zeros_like(self.encoder_input[0], dtype=np.int32, name="GO")] + \
                             [tf.placeholder(tf.int32, shape=(None,), name="dec_inp%i" % t) for t in range(out_seq_length - 1)]

        if dropout != 0:
            self.keep_prob = tf.placeholder(tf.float32)

        if not self.bidirectional:
            self.cell = self._new_RNN_cell(self.memory_dim)
            #self.dec_outputs, self.dec_memory = tf.nn.seq2seq.embedding_attention_seq2seq(self.encoder_input, self.decoder_input,
            self.dec_outputs, self.dec_memory = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(self.encoder_input, self.decoder_input,
                                                                                                            self.cell, vocabulary_size, vocabulary_size,
                                                                                                            embedding_dim, feed_previous=True)
        else:
            self.input_cell_forward = self._new_RNN_cell(self.memory_dim/2)
            self.input_cell_backward = self._new_RNN_cell(self.memory_dim/2)
            self.output_cell = self._new_RNN_cell(self.memory_dim)
            
            self.dec_outputs, self.dec_memory, = custom_seq2seq.embedding_attention_bidirectional_seq2seq(self.encoder_input, self.decoder_input, self.input_cell_forward,
                                                                                                          self.input_cell_backward, self.output_cell, self.vocabulary_size,
                                                                                                          self.vocabulary_size, self.embedding_dim, feed_previous=True)
            
        
        if trainable:
            #self.loss = tf.nn.seq2seq.sequence_loss(self.dec_outputs, self.labels, self.weights, vocabulary_size)
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.dec_outputs, self.labels, self.weights, vocabulary_size)
    
            self.optimizer = tf.train.AdamOptimizer()
            gvs = self.optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    
            self.train_op = self.optimizer.apply_gradients(capped_gvs)
        
        self.saver    = tf.train.Saver(tf.all_variables(), max_to_keep=5)

    def load_parameters(self, sess, filename):
        self.saver.restore(sess, filename)

    def save_parameters(self, sess, filename, global_step=None):
        self.saver.save(sess, filename, global_step=global_step)

    def train_step(self, sess, X, Y):
        if not self.trainable:
            raise Exception
        
        X = np.array(X).T
        Y = np.array(Y).T
            
        feed_dict = {self.encoder_input[t]: X[t] for t in range(self.in_seq_length)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.out_seq_length)})
        
        if self.dropout != 0:
            feed_dict.update({self.keep_prob: 1.0-self.dropout})

        _, loss_t = sess.run([self.train_op, self.loss], feed_dict)
            
        return loss_t

    def validate_step(self, sess, X, Y):
        X = np.array(X).T
        Y = np.array(Y).T

        feed_dict = {self.encoder_input[t]: X[t] for t in range(self.in_seq_length)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.out_seq_length)})

        if self.dropout != 0:
            feed_dict.update({self.keep_prob: 1.0})

        loss_t = sess.run([self.loss], feed_dict)
        dec_outputs_batch = sess.run(self.dec_outputs, feed_dict)
        Y_hat = [logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]
            
        return loss_t, np.array(Y_hat).T

    def sample(self, sess, X):
        X = np.array(X).T

        feed_dict = {self.encoder_input[t]: X[t] for t in range(self.in_seq_length)}

        if self.dropout != 0:
            feed_dict.update({self.keep_prob: 1.0})

        dec_outputs_batch = sess.run(self.dec_outputs, feed_dict)
        Y_hat = [logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]

        return np.array(Y_hat).T

    def get_attention_vectors(self, sess, X):
        pass
