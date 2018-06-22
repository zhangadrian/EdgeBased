import sys
import numpy as np 
import tensorflow as tf 
from modelNew import EdgeBase

import six.moves.cPickle as pickle 

import gzip

tf.set_random_seed(0)
import time

tf.flags.DEFINE_float('learning_rate', 0.01, 'learning_rate')
tf.flags.DEFINE_integer('sequence_batch_size', 16, 'sequence_batch_size')
tf.flags.DEFINE_integer('n_hidden_gru', 32, 'n_hidden_gru')
tf.flags.DEFINE_float('l1', 5e-5, 'l1')
tf.flags.DEFINE_float('l2', 5e-5, 'l2')
tf.flags.DEFINE_float('l1l2', 1.0, 'l1l2')
tf.flags.DEFINE_string('activation', 'tanh', 'activation_function')
tf.flags.DEFINE_integer('training_iters', 50*3200+1, 'max training iteration')
tf.flags.DEFINE_integer('display_step', 100, 'display_step')
tf.flags.DEFINE_integer('embedding_size', 50, 'embedding_size')
tf.flags.DEFINE_integer('n_hidden_dense', 32, 'dense size')
tf.flag.DEFINE_float('max_grad_norm', 100.0, 'max gradient clip')
tf.flags.DEFINE_float('stdev', 0.01, 'parameter initialization standard deviation')
tf.flags.DEFINE_float('dropout_prob', 1.0, 'dropout_prob')

config = tf.flags.FLAGS 

def get_batch(sourceNodeList, targetNodeList, batch_size=32):
	print('Test1')