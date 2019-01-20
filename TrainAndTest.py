import tensorflow as tf
import numpy as np
import datetime
import os
import time
now = datetime.datetime.now()
start_time = time.time()
#	
if __name__ == '__main__':
	
	features, targets = np.loadtxt("features.txt"), np.loadtxt("targets.txt")
	print('-> Loading of Features done !')
	print(features.shape, targets.shape)

	features_train = features[:-2]
	targets_train = targets[:-2]
	features_test = features[-2:]
	targets_test = targets[-2:]

	nb_input=features.shape[1]
	nb_output=targets.shape[1]
	
	# Inputs/Outputs
	tf_features = tf.placeholder(tf.float32, shape=[None, nb_input])
	tf_targets  = tf.placeholder(tf.float32, shape=[None, nb_output])
	
	# Variables
	w1 = tf.Variable(tf.random.uniform([nb_input, nb_output], minval=0., maxval=0.01, dtype=tf.float32))
	b1 = tf.Variable(tf.zeros([nb_output]))
	#  Operations
	z1 = tf.matmul(tf_features, w1) + b1
	
	# tf session and variables initialization
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
		
	error = tf.squared_difference(z1,tf_targets)
	train = tf.train.GradientDescentOptimizer(1e-4).minimize(error)
		
	# training of the model
	epochs = 10000
	batch_size=100
	pred,err=sess.run([z1,error], feed_dict={
	tf_features: [features[0]],
	tf_targets: [targets[0]]})
	print("\t error =",np.mean(err))
	print("\t pred =",pred)
		
	for e in range(epochs):
		for b in range(0, len(features_train), batch_size):
			batch_features=features_train[b:b+batch_size]
			batch_targets=targets_train[b:b+batch_size]
			sess.run(train, feed_dict={
				tf_features: batch_features,
				tf_targets: batch_targets})
		pred,err=sess.run([z1,error], feed_dict={
			tf_features: batch_features,
			tf_targets: batch_targets})
		print("iter =",e+1)
		print("\t error =",np.mean(err))
		print("\t--- %s minutes ---" % ((time.time() - start_time)/60.))


	#Launch the test to check accuracy of the model
	for i in range(len(features_test)):
		py=sess.run(z1, feed_dict={
			tf_features: [features_test[i]]
		})
		#10th first values displayed 
		print(py[0][:10], targets_test[i][:10])
