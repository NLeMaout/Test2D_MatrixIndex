import tensorflow as tf
import numpy as np
import datetime
import os
import time
now = datetime.datetime.now()
start_time = time.time()
#	
global checkpoint_dir
checkpoint_dir="./Check_point/"
#
def loadmodel(session, saver, checkpoint_dir):
    session.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
#
def save(session, saver, checkpoint_dir, step):
    dir = os.path.join(checkpoint_dir, "model")
    saver.save(session, dir, global_step=step)
#
if __name__ == '__main__':
	
	Restart=True
	if not os.path.exists(checkpoint_dir) and Restart:
		os.makedirs(checkpoint_dir)
	
	features, targets = np.loadtxt("./03_From_Python/features.txt"), np.loadtxt("./03_From_Python/targets.txt")
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
	
	if True: #test with one hidden layer (delete all variables from other configuration saved previously)
		
		hidden_layer1=int(nb_input*2)
		hidden_layer2=int(nb_input*2)
		
		# Variables
		w0 = tf.Variable(tf.random.uniform([nb_input, hidden_layer1], minval=0., maxval=0.01, dtype=tf.float32))
		b0 = tf.Variable(tf.zeros([hidden_layer1]))
		#  Operations
		z0 = tf.matmul(tf_features, w0) + b0
		
		# Variables
		w1 = tf.Variable(tf.random.uniform([hidden_layer1, hidden_layer2], minval=0., maxval=0.01, dtype=tf.float32))
		b1 = tf.Variable(tf.zeros([hidden_layer2]))
		#  Operations
		z11 = tf.matmul(z0, w1) + b1
		
		# Variables
		w2 = tf.Variable(tf.random.uniform([hidden_layer2, hidden_layer1], minval=0., maxval=0.01, dtype=tf.float32))
		b2 = tf.Variable(tf.zeros([hidden_layer1]))
		#  Operations
		z2 = tf.matmul(z11, w2) + b2
		
		# Variables
		w3 = tf.Variable(tf.random.uniform([hidden_layer1, nb_output], minval=0., maxval=0.01, dtype=tf.float32))
		b3 = tf.Variable(tf.zeros([nb_output]))
		#  Operations
		z1 = tf.matmul(z2, w3) + b3
	
	else:
		# Variables
		w1 = tf.Variable(tf.random.uniform([nb_input, nb_output], minval=0., maxval=0.01, dtype=tf.float32))
		b1 = tf.Variable(tf.zeros([nb_output]))
		#  Operations
		z1 = tf.matmul(tf_features, w1) + b1
	
	
	# tf session and variables initialization
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
		
	error = tf.squared_difference(z1,tf_targets)*1000**2
	train = tf.train.GradientDescentOptimizer(1e-11).minimize(error)#1e-4).minimize(error)
		
	# Reload previous check point
	if Restart:
		saver = tf.train.Saver()#max_to_keep=4, keep_checkpoint_every_n_hours=2)
		loadmodel(sess, saver, checkpoint_dir)
		print('-> Loading of previous values of variables done !')
		
	# training of the model
	epochs = 1000
	batch_size=100
	alter=1
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
		if (e % 100 == 0):
			pred,err=sess.run([z1,error], feed_dict={
				tf_features: batch_features,
				tf_targets: batch_targets})
			print("iter =",e+1)
			print("\t error =",np.mean(err))
			print("\t--- %s minutes ---" % ((time.time() - start_time)/60.))
		
		if (e % 1000 == 0) and Restart:
			save(sess, saver, checkpoint_dir, alter)
			if alter==1:	
				alter=2
			else:
				alter=1
	print("\t--- %s minutes ---" % ((time.time() - start_time)/60.))

	#Launch the test to check accuracy of the model
	for i in range(len(features_test)):
		pred,err=sess.run([z1,error], feed_dict={
			tf_features: [features_test[i]],
			tf_targets: [targets_test[i]]
		})
		#10th first values displayed 
		print("\t error =",np.mean(err))
		print(pred[0][:10], targets_test[i][:10])
