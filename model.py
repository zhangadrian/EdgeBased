from collections import namedtuple

import tensorflow as tf 
import math

import edgebase.layers as layers
import edgebase.metrics as metrics

from .prediction import EdgePredLayer
from .aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
	"""docstring for Model"""
	def __init__(self, **kwargs):
		allowed_kwargs = {'name', 'logging', 'model_size'}
		for kwarg in kwargs.keys():
			assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
		name = kwargs.get('name')
		if not name:
			name = self.__class__.__name__.lower()
			self.name = name

		logging = kwargs.get('logging', False)

		self.logging = logging

		self.vars = {}
		self.placeholders = {}

		self.layers = []
		self.activations = []

		self.inputs = None
		self.outputs = None

		self.loss = 0
		self.accuracy = 0
		self.optimizer = None
		self.opt_op = None

	def _build(self):
		raise NotImplementedError

	def build(self):
		with tf.variable_scope(self.name):
			self._build()

		self.activations.append(self.inputs)
		for layer in self.layers:
			hidden = layer(self.activations[-1])
			self.activations.append(hidden)
		self.outputs = self.activations[-1]

		variables = tf.get_collections(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		self.vars = {var.name: var for var in variables}

		self._loss()
		self._accuracy()

		self.opt_op = self.optimizer.minimize(self.loss)

	def predict(self):
		pass

	def _loss(self):
		raise NotImplementedError

	def _accuracy(self):
		raise NotImplementedError

	def save(self, sess=None):
		if not sess:
			raise AttributeError("TensorFlow session not provided.")
		saver = tf.train.Saver(self.vars)
		save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
		print('Model saved in file: %s' % save_path)

	def load(self, sess=None):
		if not sess:
			raise AttributeError('TensorFlow session not provided.')
		saver = tf.train.Saver(self.vars)
		save_path = 'tmp/%s.ckpt' % self.name
		saver.restore(sess, save_path)
		print('Model restored from file: %s' % save_path)

class MLP(Model):
	def __init__(self, placeholders, dims, categorical = True, **kwargs):
		super(MLP, self).__init__(**kwargs)

		self.dims = dims
		self.input_dim = dims[0]
		self.output_dim = dims[-1]
		self.placeholders = placeholders
		self.categorical = categorical

		self.inputs = placeholders['features']
		self.labels = placeholders['labels']

		self.optimizer = tf.train.AdamOptimizer(learning_rates = FLAGS.learning_rates)

		self.build()

	def _loss(self):
		for var in self.layers[0].vars.values():
			self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

		if self.categorical:
			self.loss += metrics.masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
				self.placeholders['labels_mask'])

		else:
			diff = self.labels - self.outputs
			self.loss += tf.reduce_sum(tf.sqrt(tf.reduce_sum(diff*diff, axis = 1)))

	def _accuracy(self):
		if self.categorical:
			self.accuracy = metrics.masked_accuracy(self.outputs, self.placeholder['labels'],
				self.placeholders['labels_mask'])

	def _build(self):
		self.layers.append(layers.Dense(input_dim=self.input_dim,
								 output_dim = self.dims[1],
								 act=tf.nn.relu,
								 dropout=self.placeholders['dropout'],
								 sparse_inputs=False,
								 logging=self.logging))

		self.layers.append(layers.Dense(input_dim=self.dims[1],
								 output_dim=self.output_dim,
								 act=lambda x: x,
								 dropout=self.placeholders['dropout'],
								 logging=self.logging))

	def predict(self):
		return tf.nn.softmax(self.outputs)

class GeneralizedModel(Model):
	def __init__(self, **kwargs):
		super(GeneralizedModel, self).__init__(**kwargs)

	def build(self):
		with tf.variable_scope(self.name):
			self._build()
		variables = tf.get_collections(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		self.vars = {var.name: var for var in variables}

		self._loss()
		self._accuracy()

		self.opt_op = self.optimizer.minimize(self.loss)

SAGEInfo = namedtuple("SAGEInfo",
	['layer_name',
	 'neigh_sampler',
	 'num_samples',
	  'output_dim'])

class SampleAndAggregate(GeneralizedModel):

	def __init__(self, placeholders, features, adj, degrees,
			layer_infos, concat=True, aggregator_type='mean',
			model_size='small', identity_dim=0, **kwargs):
		super(SampleAndAggregate, self).__init__(**kwargs)
		if aggregator_type == 'mean':
			self.aggregators_cls = MeanAggregator
		elif aggregator_type == 'seq':
			self.aggregators_cls = SeqAggregator
		elif aggregator_type == 'maxpool':
			self.aggregator_cls = MaxPoolingAggregator
		elif aggregator_type == 'meanpool':
			self.aggregator_cls == MeanPoolingAggregator
		else:
			raise Exception('Unknown aggragator:', self.aggregators_cls)

		self.inputs1 = placeholders['batch1'] #the source node
		self.inputs2 = placeholders['batch2'] #the target node
		self.embeds = tf.constant(embeddings, dtype=tf.float32, name='GraphEmbedding')
		self.features = []
		self.source_embed = [tf.nn.embedding_lookup(self.embeds, nodeID) for nodeID in self.inputs1]
		self.target_embed = [tf.nn.embedding_lookup(self.embeds, nodeID) for nodeID in self.inputs2]

		for i in range(len(self.source_embed)):
			self.features[i] = tf.concat([self.source_embed[i], self.target_embed[i]], axis=1)

		# self.model_size = model_size
		# self.adj_info = adj
		# if identity_dim > 0:
		# 	self.embeds = tf.get_variable('node_embeddings', [adj.get_shape().as_list()[0], identity_dim])
		# else:
		# 	self.embeds = None
		# if features is None:
		# 	if identity_dim == 0:
		# 		raise Exception('Must have a positive value for identity feature dimension if no input features given.')
		# 	self.features = self.embeds
		# else:
		# 	self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
		# 	if not self.embeds is None:
		# 		self.features = tf.concat([self.embeds, self.features], axis=1)

		self.degrees = degrees
		self.concat = concat

		self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
		self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
		self.batch_size = placeholders['batch_size']
		self.placeholders = placeholders
		self.layer_infos = layer_infos

		self.optimizer = tf.train.AdamOptimizer(learning_rates=FLAGS.learning_rates)

		self.build()


	def sample(self, inputs, layer_infos, batch_size=None):
		if batch_size is None:
			batch_size = self.batch_size
		samples = [inputs]

		support_size = 1
		support_sizes = [support_size]
		for k in range(len(layer_infos)):
			t = len(layer_infos)-k-1
			support_size *= layer_infos[t].num_samples
			sampler = layer_infos[t].neigh_sampler
			node = sampler((samples[k], layer_infos[t].num_samples))
			samples.append(tf.reshape(node, [support_size*batch_size,]))
			support_size.append(support_size)
		return samples, support_sizes

	def aggregate(self, input_features, dims, batch_size=None,
			aggregators=None, name=None, concat=False, model_size='small'):

		if batch_size is None:
			batch_size = self.batch_size
		return tf.nn.reduce_mean(input_features, axis)

		# hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]

		# aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x: x, 
		# 					 dropout=self.placeholders['dropout'],
		# 					 name=name, concat=concat, model_size=model_size)
		# new_agg = aggregators is None
		# if new_agg:
		# 	aggregators = []
		# for layer in range(len(num_samples)):
		# 	if new_agg:
		# 		dim_mult = 2 if concate and (layer != 0) else 1

		# 		if layer == len(num_samples) - 1:
		# 			aggregators = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x: x, 
		# 					 dropout=self.placeholders['dropout'],
		# 					 name=name, concat=concat, model_size=model_size)
		# 		else:
		# 			aggregator = self.aggregators_cls(dim_mult*dims[layer], dims[layer+1],
		# 					dropout=self.placeholders['dropout'],
		# 					name=name, concat=concat, model_size=model_size)
		# 			aggregator.append(aggregator)
		# 	else:
		# 		aggregator = aggregator[layer]

		# 	next_hidden = []

		# 	for hop in range(len(num_samples)-layer):
		# 		dim_mult = 2 if concat and (layer != 0) else 1
		# 		neigh_dims = [batch_size * support_sizes[hop],
		# 					  num_samples[len(num_samples) - hop - 1],
		# 					  dim_mult*dims[layer]]
		# 		h = aggregator((hidden[hop], 
		# 						tf.reshape(hidden[hop + 1], neigh_dims)))
		# 		next_hidden.append(h)
		# 	hidden = next_hidden
		# return hidden[0], aggregators

	def _build(self):
		labels = tf.reshape(tf.cast(self.placeholders['batch2'], dtype=tf.int64),
				 [self.batch_size, 1])
		self.neg_samples, _ = tf.nn.fixed_unigram_candidate_sampler(
			true_classes=labels,
			num_true=1,
			num_sampled=FLAGS.neg_samples_size,
			unique=False,
			range_max=len(self.degrees),
			distortion=0.75,
			unigrams=self.degrees.tolist())

		samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
		samples2, supoort_sizes2 = self.sample(self.inputs2, self.layer_infos)
		num_samples = [layer_infos.num_samples for layer_info in self.layer_infos]
		self.outputs1, self.aggregators = self.aggregate(samples1, [self.feature], self.dims, num_samples,
			support_sizes1, concat=self.concat, model_size=self.model_size)
		self.outputs2, _ = self.aggregate(samples2, [self.features], self.dims, num_samples,
				supoort_sizes2, aggregators=self.aggregators, concat=self.concat,
				model_size=self.model_size)
		self.outputs = self.aggreate([self.features], self.dims,
				aggregators=self.aggregators, concat=self.concat,
				model_size=self.model_size)

		neg_samples, neg_support_sizes = self.sample(self.neg_samples, self.layer_infos,
			FLAGS.neg_sample_size)
		sel.neg_output, _ = self.aggregate(neg_samples, [fle.features], self.dims, num_samples,
			neg_support_sizes, batch_size=FLAGS.neg_sample_size, aggregators=self.aggregators,
			concat=self.concat, model_size=self.model_size)

		dim_mult = 2 if self.concat else 1
		self.ink_pred_layer = EdgePredLayer(dim_mult*self.dims[-1],
			dim_mult*self.dims[-1], self.placeholders, act=tf.nn.sigmoid,
			bilinear_weight=False,
			name='edge_predict')

		self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
		self.outputs2 = tf.nn.l2_normalize(self.outputs2, 1)
		self.neg_outputs = tf.nn.l2_normalize(self.outputs, 1)

		def build(self):
			self._build()

			self._loss()
			self._accuracy()
			self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
			grads_and_vars = self.optimizer.compute_gradients(self.loss)
			clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
										for grad, var in grads_and_vars]
			self.grad, _ = clipped_grads_and_vars[0]
			self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

		def _loss(self):
			for aggregator in self.aggregators:
				for var in aggregator.vars.values():
					self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

			self.loss += self.link_pred_layer.loss(self.outputs1, self.outputs2, self.neg_outputs)
			tf.summary.scalar('loss', self.loss)

		def _accuracy(self):
			aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
			self.neg_aff = self.link_pred_layer.neg_cost(self.outputs1, self.neg_outputs)
			self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
			_aff = tf.expand_dims(aff, axis=1)
			self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
			size = tf.shape[self.aff_all][1]

			_, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
			_, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
			self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1]+1, tf.float32)))
			tf.summary.scalar('mrr', self.mrr)

class Node2VecModel(GeneralizedModel):
	def __init__(self, placeholders, dict_size, degrees, name=None,
				nodevec_dim=50, lr=0.001, **kwargs):
		super(Node2VecModel, self).__init__(**kwargs)

		self.placeholders = placeholders
		self.degrees = degrees
		self.inputs1 = placeholders['batch1']
		self.inputs2 = placeholders['batch2']

		self.batch_size = placeholders['batch_size']
		self.hidden_dim = nodevec_dim

		self.target_embeds = tf.Variable(tf.random_uniform([dict_size, nodevec_dim], -1, 1),
										name='target_embeds')
		self.context_embeds = tf.Variable(tf.truncated_normal([dict_size, nodevec_dim],
										  stddev=1.0/math.sqrt(nodevec_dim)),
										  name='context_embeds')
		self.context_bias = tf.Variable(tf.zeros([dict_size]),
										name='context_bias')

		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

		self.build()

	def _build(self):
		labels = tf.reshape(tf.cast(self.placeholders['batch2'], dtype = tf.int64), [self.batch_size, 1])
		self.neg_samples, _, _, = tf.nn.fixed_unigram_candidate_sampler(true_classes=labels,
																		num_true=1,
																		num_sampled=FLAGS.neg_sample_size,
																		unique=True,
																		range_max=len(self.degrees),
																		distortion=0.75,
																		unigrams=self.degrees.tolist())
		self.outputs1 = tf.nn.embedding_lookup(self.target_embeds, self.inputs1)
		self.outputs2 = tf.nn.embedding_lookup(self.context_embeds, self.inputs2)
		self.outputs2_bias = tf.nn.embedding_lookup(self.context_bias, self.inputs2)
		self.neg_outputs = tf.nn.embedding_lookup(self.context_embeds, self.neg_samples)
		self.neg_outputs_bias = tf.nn.embedding_lookup(self.context_bias, self.neg_samples)

		self.link_pred_layer = EdgePredLayer(self.hidden_dim, self.hidden_dim, self.placeholders, bilinear_weight=False)

	def build(self):
		self._build()
		self._loss()
		self._minimize()
		self._accuracy()

	def _minimize(self):
		self.opt_op = self.optimizer.minimize(self.loss)

	def _loss(self):
		aff = tf.reduce_sum(tf.multiply(self.outputs1, self.outputs2), 1) + self.outputs2_bias
		neg_aff = tf.matmul(self.outputs2, tf.transpose(self.neg_outputs)) + self.neg_outputs_bias
		true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
		negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)
		loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
		self.loss = loss / tf.cast(self.batch_size, tf.float32)
		tf.summary.scalar('loss', self.loss)

	def _accuracy(self):
		add = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
		self.neg_aff = self.link_pred_layer.neg_cost(self.outputs1, self.neg_outputs)

		self.neg_aff = tf. reshape(self.neg_affm [self.batch_size, FLAGS.neg_sample_size])
		_aff = tf.expand_dims(aff, axis=1)
		self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
		size = tf.shape(self.aff_all)[1]
		_, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
		_, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
		self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:,-1]+1, tf.float32)))
		tf.summary.scalar('mrr', self.mrr)
 


















	