import sys
import numpy as np 
import tensorflow as tf 

class EdgeBase(object):
	def __init__(self, config, sess, node_embed, sourceNodeList, targetNodeList):
		self.n_snapshot = config,n_snapshot
		self.max_seq_len = config.max_seq_len
		self.learning_rate = config.learning_rate
		self.training_iters = config.training_iters
		self.sequence_batch_size = config.sequence_batch_size
		self.batch_size = config.batch_size
		self.display_step = config.display_step

		if config.activation == 'tanh':
			self.activation = tf.tanh
		else:
			self.activation = tf.nn.relu
		self.max_grad_norm = config.max_grad_norm
		self.initializer = tf.random_noram_initializer(stddev=config.stddev)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		self.regularizer = tf.contrib.layer.l1_l2_regularizer(self.scale1, self.scale2)
		self.dropout_prob = config.dropout_prob
		self.sess = sess
		self.node_embed = node_embed

		self.n_hidden = config.n_hidden

		self.name = 'edgebase'

		self.build_input(sourceNodeList, targetNodeList)
		self.build_var()
	def build_input(self, sourceNodeList, targetNodeList):
		self.inputData = np.zeros((self.max_seq_len, 2*self.node_embed.shape()[1]))
		for i in range(len(sourceNodeList)):
			tempSourceNode = sourceNodeList[i]
			tempTargetNode = targetNodeList[i]

			sourceEmb = [tf.nn.embedding_lookup(self.node_embed, nodeID) for nodeID in tempSourceNode]
			targetEmb = [tf.nn.embedding_lookup(self.node_embed, nodeID) for nodeID in tempTargetNode]
			tempFeature = []
			for j in range(len(sourceEmb)):
				tempFeature.append(tf.concat([sourceEmb[j], targetEmb[j]]), axis=1)
			# Aggregate function. This is be a class in the future.
			self.inputData[i,:] = tf.nn.reduce_mean(tf.constant(tempFeature), axis=1)
	def build_var(self):
		with tf.variable_scope(self.name) as scope:
			with tf.variable_scope('BasicLSTM'):
				self.BasicLSTM = tf.contrib.rnn.BasicLSTMCell(n_hidden)



	def build_dynamic_model(self):
		with tf.device('/gpu:0'):
			with tf.variable_scope(self.name) as scope:
				with tf.variable_scope('input'):
					input = tf.nn.dropout(self.inputData, self.dropout_prob)
				with tf.variable_scope('RNN'):
					outputs, _, _ = tf.nn.dynamic_rnn(self.BasicLSTMCell, input, dtype=tf.float32)
				with tf.variable_scope('dense'):
					dense = self.activation(tf.add(tf.matmul(outputs, self.weight, self.biases)))

	def train_batch(self, sourceNodeList, targetNodeList):
		self.sess.run(self.train_op, feed_dict={self.sourceNodeList: sourceNodeList, self.targetNodeList: targetNodeList})

	def get_error(self, sourceNodeList, targetNodeList):
		return self.sess.run(self.error, feed_dict={self.sourceNodeList: sourceNodeList, self.targetNodeList: targetNodeList})






		


