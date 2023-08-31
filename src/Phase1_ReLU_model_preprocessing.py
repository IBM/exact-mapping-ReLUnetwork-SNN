import numpy as np    
import os
import pickle as pkl
import tensorflow as tf
import logging
from Dataset import Dataset
from utils import *
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.keras.backend.set_floatx('float64')

FLAGS = tf.compat.v1.app.flags.FLAGS
# Basic flags
tf.compat.v1.flags.DEFINE_string("data_name", "MNIST", "[MNIST, fMNIST, cifar10, cifar100, pass, places]")
tf.compat.v1.flags.DEFINE_string("places_data_path", "", "path to the places dataset files (obtained by calling PLACES365_pretrained_model_preprocessed_data.py)")
tf.compat.v1.flags.DEFINE_string("pass_data_path", "", "path to the PASS dataset stored in 20 subfolders")
tf.compat.v1.flags.DEFINE_string("model_name", "", "name of the pretrained model (needs to end with _original)")
tf.compat.v1.flags.DEFINE_bool("CNN", True, "CNN architecture yes/no")
tf.compat.v1.flags.DEFINE_string("logging_dir", '', "directory for logging")
tf.compat.v1.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.compat.v1.flags.DEFINE_integer("layers", 16, "number of layers")
# Train models with activity regularization (or pretrain for smaller datasets); 
tf.compat.v1.flags.DEFINE_integer("epochs", 100, "number of training epochs")
tf.compat.v1.flags.DEFINE_float("l1", 0, "activity regularization to increase sparsity")
# Phase 1: preprocessing of the network
tf.compat.v1.flags.DEFINE_bool("BN", False, "whether there is batch norm")
tf.compat.v1.flags.DEFINE_bool("BN_before_ReLU", False, "whether batch norm is before ReLU")
tf.compat.v1.flags.DEFINE_float("lower_b", -10, "lower boundary value to be used for weights scaling")
tf.compat.v1.flags.DEFINE_float("upper_b", 0.99, "upper boundary value to be used for weights scaling")

set_up_logging(FLAGS.logging_dir, 'phase1:' + FLAGS.model_name.replace('_original', ''))
logging.info (FLAGS.logging_dir)

# Create data object
data = Dataset(
    FLAGS.data_name,
    FLAGS.logging_dir, 
    FLAGS.CNN,
    ReLU_preproc=True,
    data_paths={'PLACES365':FLAGS.places_data_path, 'PASS':FLAGS.pass_data_path}
)  

# Create model and load weights;
if 'MNIST' in FLAGS.data_name:
    # For MNIST we consider 3 architectures, 1. 2-layer fully-connected network, 2. 5-layer LeNet and 3. 16-layer VGG-like network.
    # For Fashion-MNIST we consider one architecture, a 16-layer VGG-like network.
    # Create optimizer in case we need to pretrain models.
    optimizer = get_optimizer()
    if not FLAGS.CNN:
        # if not CNN model, then create a shallow 2-layer fully connected network.
        model=create_fc_model_ReLU(optimizer=optimizer, dropout=0.08, l1=FLAGS.l1)
    else:
        if FLAGS.layers==5:
            # 5-layer CNN model, LeNet architecture. Set flags for the processing later on.
            model=create_lenet_model_ReLU (dropout=0.08, optimizer=optimizer)
            FLAGS.BN=True
            FLAGS.BN_before_ReLU=True
        elif FLAGS.layers==16:
            # 16-layer CNN model, VGG-like architecture. Set flags for the processing later on.
            layers2D = [64, 64, 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
            kernel_size=(3,3)
            layers1D=[512, 512]
            FLAGS.BN=True
            FLAGS.BN_before_ReLU=False
            model=create_vgg_model_ReLU (layers2D, kernel_size, layers1D, data, BN=FLAGS.BN, dropout=0.08, optimizer=optimizer)
    if FLAGS.l1!=0:
        # Pretrain model with activity regularization.
        logging.info("Training.")
        history=model.fit(
            data.x_train, data.y_train, 
            batch_size = FLAGS.batch_size,
            epochs=FLAGS.epochs,
            verbose=1,
            validation_data=(data.x_test, data.y_test)
            )
        model.save_weights(FLAGS.logging_dir + '/' + FLAGS.model_name + '.h5')
    # Load pretrained weights
    model.load_weights(FLAGS.logging_dir + '/' + FLAGS.model_name + '.h5')
if 'CIFAR' in FLAGS.data_name:
    # For CIFAR10 and CIFAR100 we consider one architecture, a 15-layer VGG-like network.
    layers2D = [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
    kernel_size=(3,3)
    layers1D=[512]
    if FLAGS.l1!=0:
        # Pretrain model with activity regularization and save weights. 
        train_cifar10(data, FLAGS.logging_dir, FLAGS.model_name, FLAGS.l1)   
    FLAGS.BN=True
    FLAGS.BN_before_ReLU=False
    model=create_vgg_model_ReLU (layers2D, kernel_size, layers1D, data, BN=FLAGS.BN)
    model.load_weights(FLAGS.logging_dir + '/' + FLAGS.model_name + '.h5')
if FLAGS.data_name=='PLACES365':
    # For PLACES 365, we consider one architecture, a VGG16 network.
    layers2D = [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
    kernel_size=(3,3)
    layers1D=[4096, 4096]
    model=create_vgg_model_ReLU (layers2D, kernel_size, layers1D, data, BN=FLAGS.BN)
    # Load parameters for VGG16 and places365 dataset.
    load_places365_model(model, FLAGS.logging_dir)
if FLAGS.data_name=='PASS':
    # For PASS, we consider one architecture, a VGG16 network.
    # We stored pretrained imagenet weights using this code.
    model = VGG16(weights='imagenet')
    model.save_weights(FLAGS.logging_dir + '/' + FLAGS.model_name + '.h5')
    layers2D = [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
    kernel_size=(3,3)
    layers1D=[4096, 4096]
    model=create_vgg_model_ReLU (layers2D, kernel_size, layers1D, data, BN=FLAGS.BN)
    # Load weights.
    model.load_weights(FLAGS.logging_dir + '/' + FLAGS.model_name + '.h5')

    
# Obtain accuracy of the original ReLU model (unless we work with PASS dataset).
if FLAGS.data_name!='PASS':
    logging.info(model.summary())
    logging.info("Testing.")
    test_acc = model.evaluate(data.x_test, data.y_test, batch_size = FLAGS.batch_size)
    logging.info("Testing accuracy (ReLU original) is {}.".format(test_acc))
    
# Create predictions of the original model which will be used to calculate 'Agreement' metric.
logging.info('predictions for the agreement metric')
predictions = model.predict(data.x_test, batch_size = FLAGS.batch_size, verbose=1)
predictions = tf.argmax(predictions, axis=1)
predictions = (tf.keras.utils.to_categorical(predictions, data.num_of_classes))
pkl.dump(predictions, open(FLAGS.logging_dir + '/' + FLAGS.model_name + '_labels.pkl', 'wb'))    

# Fuse (imaginary) batch normalization layers. 
logging.info('fuse (imaginary) BN layers')
# shift/scale input data accordingly
data.x_test, data.x_train = (data.x_test - data.p)/(data.q-data.p), (data.x_train - data.p)/(data.q-data.p)
model = fuse_bn(model, FLAGS.BN, FLAGS.BN_before_ReLU, p=data.p, q=data.q)
logging.info(model.summary())
    
# Scale convolutional and fully-connected layers. 
logging.info('scale weights')
c, layer_num, X_n= 0, 0, [] 
non_zero_activations, total_output = 0, 0
for k, layer in enumerate(model.layers):
    if 'flatten' in layer.name:
        c = tf.tile(tf.squeeze(c), [layer.output_shape[-1]//layer.input_shape[-1]])
        c = tf.expand_dims(c, axis=0)
    if 'conv' in layer.name or 'dense' in layer.name: 
        #logging.info (layer.name, '\n\n\n')
        # Scale weights of the current layer.
        c, layer_num = set_params_ReLU(layer, c, layer_num, FLAGS.layers, FLAGS.upper_b, FLAGS.lower_b)
        if k!=len(model.layers)-2:
            # Calculate X_n of the current layer.
            extractor = tf.keras.Model(inputs=model.inputs, outputs=tf.reduce_max(tf.nn.relu(layer.output)))
            output = extractor.predict(data.x_train, batch_size = FLAGS.batch_size, verbose=1)
            X_n.append(np.max(output))
            # Calculate the amount of non-zero acutvation of the non-zero layer.
            extractor = tf.keras.Model(inputs=model.inputs, outputs=tf.math.count_nonzero(tf.nn.relu(layer.output)))
            output = extractor.predict(data.x_train, batch_size = FLAGS.batch_size, verbose=1)
            non_zero_activations += tf.reduce_sum(output)
            total_output+=len(data.x_train)*np.prod(list(layer.output_shape[1:]))
logging.info('X_n:', X_n)
pkl.dump(X_n, open(FLAGS.logging_dir + '/' + FLAGS.model_name.replace('original', 'scaled_X_n.pkl'), 'wb'))
logging.info(f'Non zero percentage: {non_zero_activations/total_output}') 
logging.info(model.summary())

# Save scaled ReLU model.
model.save_weights(FLAGS.logging_dir + '/' + FLAGS.model_name.replace('original', 'scaled.h5'))

# Obtain accuracy of the scaled ReLU model (unless we work with PASS dataset).
if FLAGS.data_name!='PASS':
    logging.info("Testing.")
    test_acc = model.evaluate(data.x_test, data.y_test, batch_size = FLAGS.batch_size)
    logging.info("Testing accuracy (ReLU scaled) is {}.".format(test_acc))

# In case of cifar10 dataset save layer outputs for plotting in Fig. 4. 
if FLAGS.data_name=='CIFAR10': dump_cifar10_outputs_for_plotting(model, data, FLAGS.logging_dir)


    
    
    
    
    
    

