import logging
import os
import sys
import numpy as np
import pickle as pkl
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPooling2D, MaxPool2D, Flatten, Activation, Dropout, BatchNormalization, ActivityRegularization
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from PIL import Image

def set_up_logging(logging_dir, model_name, zeta=0.5, noise=0, alpha_noise=0, trial=0):
    """
    Set up logging for the simulation.
    """
    if zeta!=0.5: logging_dir=logging_dir + f'/zeta/{zeta}/trial_{trial}'
    if noise!=0: logging_dir=logging_dir + f'/noise/{noise}/trial_{trial}'
    if alpha_noise!=0: logging_dir=logging_dir + f'/alpha_noise/{alpha_noise}/trial_{trial}'
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
           logging.FileHandler(logging_dir + f'/log_{model_name}.txt', mode='w'), 
           logging.StreamHandler(sys.stdout),
        ],
    )
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)


def set_up_plotting(logging_dir, model_name):
    """
    Set up plotting for the simulation.
    """
    plot_dir = logging_dir + f'/plotting_{model_name}'
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    return plot_dir


def shuffle_pass_data(logging_dir, pass_path):
    """
    Shuffle pass dataset and generate paths to images in random order. It is assumed that the dataset is stored in 20 folders.
    """
    folder_sizes, file_names=[], []
    # PASS dataset is split in 20 different folders.
    for folder in range(20):
        path = pass_path + "/" + str(folder)
        file_names.extend(os.listdir(path))
        folder_sizes.append(len(os.listdir(path)))
    folders_boundaries = np.cumsum(folder_sizes)
    rand_int = list(range(len(file_names)))
    np.random.shuffle(rand_int)
    folders = np.digitize(rand_int, folders_boundaries)
    # Images in random order.
    random_file_names = [file_names[i] for i in rand_int]
    # Absolute paths to images in random order.
    paths = list(zip (folders, random_file_names))
    # Store this order to have the exact same samples for ReLU network and SNN.
    pkl.dump(paths, open(logging_dir + '/random_image_paths.pk', 'wb'))
    return paths

    
def preprocess_pass_images(random_image_paths, pass_path, size, start=0):
    """
    Loads size number of images, skipps the ones which rise ValueError, otherwise the images are reshaped to (224, 224).
    """
    i, images = 0, []
    while i < size:
        path = random_image_paths[start]
        start+=1
        try:
            image=tf.image.resize(np.asarray(Image.open(pass_path + f'/{path[0]}/{path[1]}')), (224, 224))
            if np.shape(image)==(224, 224, 3):
                images.append(image.numpy())
                i+=1
        except ValueError:
            print ('pass except')
            pass
    return images

def get_optimizer():
    """
    Get optimizer for the training on MNIST/Fashion-MNIST dataset.
    """
    learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0005,
        decay_steps=5000,  
        decay_rate=0.9, 
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule) 
    return optimizer

def create_fc_model_ReLU(optimizer='adam', dropout=0, l1=0):
    """
    Create a 2-layer fully-connected ReLU network to for MNIST dataset. 
    """
    inputs = Input(shape=(784))
    x = Dropout(dropout)(inputs)
    x=Dense(600, activation=None)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = ActivityRegularization(l1)(x)
    x = Dropout(dropout)(x)
    x = Dense(10, activation=None)(x)
    outputs = tf.keras.layers.Activation('softmax')(x)
    model = Model (inputs=inputs, outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["categorical_accuracy"])
    return model   

def create_lenet_model_ReLU(dropout, optimizer):
    """
    Create a 5-layer LeNet5 ReLU network for MNIST dataset. 
    """
    inputs = Input(shape=(28, 28, 1))
    x=Conv2D(6, kernel_size=(5,5), padding='same', activation=None)(inputs)
    x=BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x) 
    x = Dropout(dropout)(x)
    x=MaxPool2D(pool_size=(2,2))(x)
    x=Conv2D(16, kernel_size=(5,5), padding='valid', activation=None)(x)
    x=BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x) 
    x = Dropout(dropout)(x)
    x=MaxPool2D(pool_size=(2,2))(x)
    x=Conv2D(120, kernel_size=(5,5), padding='valid', activation=None)(x)
    x=BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x) 
    x = Dropout(dropout)(x)
    x=Flatten()(x)
    x=Dense(84, activation=None)(x)
    x=BatchNormalization()(x) 
    x = tf.keras.layers.Activation('relu')(x)
    x=Dense(10, activation=None)(x)
    outputs = tf.keras.layers.Activation('softmax')(x)
    model = Model (inputs=inputs, outputs=outputs)  
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    return model

def create_vgg_model_ReLU(layers2D, kernel_size, layers1D, data, BN, dropout=0, optimizer='adam'):
    """
    Create a VGG-like ReLU network for various dataset. 
    """
    inputs = Input(shape=data.input_shape)
    for k, f in enumerate(layers2D):
        if f!='pool':
            if k==0:
                x=Conv2D(f,  kernel_size, padding='same', activation=None)(inputs)
            else:
                x=Conv2D(f,  kernel_size, padding='same', activation=None)(x)
            x = tf.keras.layers.Activation('relu')(x)
            if BN: x=BatchNormalization()(x) 
            x = Dropout(dropout)(x)
        else:
            x=MaxPool2D()(x)
    x=Flatten()(x)
    for j, d in enumerate(layers1D):
        x=Dense(d, activation=None)(x)
        x = tf.keras.layers.Activation('relu')(x)
        if BN: x=BatchNormalization()(x) 
        x = Dropout(dropout)(x)
    x=Dense(data.num_of_classes, activation=None)(x)
    outputs = tf.keras.layers.Activation('softmax')(x)
    model = Model (inputs=inputs, outputs=outputs)  
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    return model

def build_cifar_model(input_shape, num_of_classes, optimizer, l1=0, weight_decay=0.0005):
    """
    Build model for CIFAR10 training with ActivityRegularization layer and weight decay.
    """
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(ActivityRegularization(l1))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(ActivityRegularization(l1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(ActivityRegularization(l1))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(ActivityRegularization(l1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(ActivityRegularization(l1))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(ActivityRegularization(l1))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(ActivityRegularization(l1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(ActivityRegularization(l1))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(ActivityRegularization(l1))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(ActivityRegularization(l1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(ActivityRegularization(l1))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(ActivityRegularization(l1))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(ActivityRegularization(l1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(ActivityRegularization(l1))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    return model

        
def train_cifar10(data, logging_dir, model_name, l1):
    
    """
    Train VGG-like model on CIFAR10 dataset, with activity regularization. 
    """
    # Training speifications.
    optimizer = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) 
    def lr_reduce(epoch): return 0.1 * (0.5 ** (epoch // 20))
    model = build_cifar_model(data.input_shape, data.num_of_classes, optimizer, l1)
    # Load/preprocess training data for the fit method.
    (x_train, y_train), _=tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_train = (x_train-np.mean(x_train, axis=(0,1,2,3)))/(np.std(x_train, axis=(0, 1, 2, 3))+1e-7)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    validation_data=(data.x_test, data.y_test)
    
    #Augment data.
    data_generator = ImageDataGenerator(
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        zca_whitening=False,  
        rotation_range=15,  
        horizontal_flip=True,  
        vertical_flip=False,
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
    )  
    data_generator.fit(x_train)
    
    #Fit model with L1 regularization and save weights.
    model.fit_generator(data_generator.flow(x_train, y_train, batch_size=128),
                        steps_per_epoch=x_train.shape[0] // 128,
                        epochs=250, 
                        validation_data=validation_data, 
                        callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_reduce)]
                       )
    model.save_weights(logging_dir + '/' + model_name + '.h5')
    tf.keras.backend.clear_session()
        

def load_places365_model(model, logging_dir):
    """
    Load pretrained parameters for VGG16 model and places365 dataset.
    """
    params = dict(np.load(f'{logging_dir}/places_vgg_relu_original.npz', allow_pickle=True, encoding="latin1"))
    names_map = {'conv1_1':'conv2d', 'conv1_2':'conv2d_1', 'conv2_1':'conv2d_2', 'conv2_2':'conv2d_3', 'conv3_1':'conv2d_4', 'conv3_2':'conv2d_5', 'conv3_3':'conv2d_6', 'conv4_1':'conv2d_7', 'conv4_2':'conv2d_8', 'conv4_3':'conv2d_9', 'conv5_1':'conv2d_10', 'conv5_2':'conv2d_11', 'conv5_3':'conv2d_12', 'fc6':'dense', 'fc7':'dense_1', 'fc8a':'dense_2'}
    for k in params:
        kernel = np.stack(params[k][0], axis=0)
        if 'conv' in k:
            layer = model.get_layer(names_map[k])
            layer.kernel.assign(np.transpose(kernel, axes=[2, 3, 1, 0]))
            layer.bias.assign(np.array(params[k][1]))
        elif 'fc' in k and k!='fc8a':
            layer = model.get_layer(names_map[k])
            if k=='fc6':
                out_shape, inp_shape = np.shape(kernel)
                # First dense layer (after flatten), change channel first to channel last.
                kernel = np.transpose(np.reshape(kernel, (out_shape, 512, 7, 7)), axes=[0, 2, 3, 1])
                kernel = np.reshape(kernel, (out_shape, inp_shape))
                layer.kernel.assign(np.transpose(kernel, axes=[1, 0]))
            else:
                layer.kernel.assign(np.transpose(kernel, axes=[1, 0]))
            layer.bias.assign(params[k][1])
    layer = model.get_layer(names_map['fc8a'])
    layer.kernel.assign(np.transpose(np.stack(params['fc8a'][0], axis=0), axes=[1,0]))
    layer.bias.assign(params['fc8a'][1])
    
def set_params_ReLU(layer, c, layer_num, layers, upper_b=0.99, lower_b=-10):
    """
    Scale parameters of a given layer in the ReLU network. After scaling, the summation of weights to each neuron (c) is between lower_b and upper_b.   
    """
    if layer_num!=0:
        # Scale outgoing weights (incoming weights to the next layer) to keep the network output equivalent to the original ReLU network.
        if 'conv' not in layer.name:
            c = tf.transpose(c)
        else:
            c = tf.transpose(c, perm=[0, 1, 3, 2])
        layer.kernel.assign(tf.where(c>upper_b, (1/upper_b * layer.kernel*c), layer.kernel))
        layer.kernel.assign(tf.where(c<lower_b, (1/lower_b * layer.kernel*c), layer.kernel))
    if 'conv' not in layer.name:
        c = tf.expand_dims(tf.math.reduce_sum(layer.kernel, axis=0), axis=0)
    else:
        c = tf.math.reduce_sum(layer.kernel, axis=(0, 1, 2))
        c = tf.expand_dims(tf.expand_dims(tf.expand_dims(c, axis=0), axis=0), axis=0)
    if layer_num==layers-1: return 0, 0
    # Scale incoming weights such that the summation of input weights for each neuron is within the bundaries.
    layer.kernel.assign(tf.where(c>upper_b, ((layer.kernel)*upper_b/c), layer.kernel))
    layer.kernel.assign(tf.where(c<lower_b, ((layer.kernel)*lower_b/c), layer.kernel))
    layer.bias.assign(tf.where(tf.squeeze(c)>upper_b, ((layer.bias)*upper_b/tf.squeeze(c)), layer.bias))
    layer.bias.assign(tf.where(tf.squeeze(c)<lower_b, ((layer.bias)*lower_b/tf.squeeze(c)), layer.bias))
    return c, layer_num + 1
    
class Conv2DWithBias(tf.keras.layers.Conv2D): 
    """
    Convolutional layer with potentially different bias at different locations.
    """
    def build(self, input_shape):
        super().build(input_shape)
        # These two variables determine the processing of the layer.
        self.BN=tf.Variable(tf.constant([0]), name='BN')
        self.BN_before_ReLU=tf.Variable(tf.constant([0]), name='BN_before_ReLU')
        
    def set_bias(self, bias, W=None, b_term=[0.0]):
        """
        Creates bias variable and changes bias on certain locations when needed.
        """
        # The bias Variable is added in this function and can have 9 potential values for each filter.
        # W can represent the kernel before fusion and b_term corresponds to the term which is multiplied with kernel (see Eqs. 9, 11, etc.).  
        self.bias = self.add_weight(shape=(9, self.filters), initializer='zeros', dtype=tf.float64, name='bias')
        self.use_bias=True
        if W is not None: 
            # Calculate the overall kernel summation.
            W_sum_2D  = tf.math.reduce_sum(W, axis=(0, 1))
            b_term = b_term[:tf.shape(W)[2]]
        for i in range(9):
            # delta_sum_W calculates kernel summation of the weights which correspond to the zero-padded inputs for the particular image part. 
            if i==0:
                # This branch corresponds to the inner part of the image which has unchanged bias.
                b_term=tf.cast(b_term, dtype=tf.float64)
                # delta_sum_W is equal to 0 which yields unchanged bias.
                delta_sum_W = tf.zeros((tf.shape(b_term)[0], 1), dtype=tf.float64)
            elif i==1:
                # This branch corresponds to the top left corner of the image, etc. 
                delta_sum_W = (W_sum_2D - tf.reduce_sum(W[1:, 1:, :, :], axis=[0, 1]))
            elif i==2:
                delta_sum_W = tf.reduce_sum(W[:1, :, :, :], axis=[0, 1])
            elif i==3:
                delta_sum_W = (W_sum_2D - tf.reduce_sum(W[1:, :-1, :, :], axis=[0, 1]))
            elif i==4:
                 delta_sum_W = tf.reduce_sum(W[:, -1:, :, :], axis=[0, 1])
            elif i==5:
                delta_sum_W = (W_sum_2D - tf.reduce_sum(W[:-1, :-1, :, :], axis=[0, 1]))
            elif i==6:
                delta_sum_W = tf.reduce_sum(W[-1:, :, :, :], axis=[0, 1])
            elif i==7:
                delta_sum_W = (W_sum_2D - tf.reduce_sum(W[:-1, 1:, :, :], axis=[0, 1]))
            elif i==8:
                delta_sum_W = tf.reduce_sum(W[:, :1, :, :], axis=[0, 1])
            # See Eqs. 11, 14.  
            delta_bias = tf.reduce_sum(tf.matmul(tf.linalg.diag(b_term), delta_sum_W), axis=0)
            # The bias is decreased for the value which corresponds to the terms which come from zero-padding. 
            self.bias[i].assign(bias - delta_bias)
            # When padding=='valid' or there is no batch normalization layer, or the batch normalization layer is fused with the previous convolutional layer, all locations have the same bias and we break from the for loop. 
            if self.padding=='valid' or self.BN!=1 or self.BN_before_ReLU==1: break
                
    def call(self, inputs):
        # Convolution operation is initially called. 
        result = self._convolution_op(inputs, self.kernel)
        if self.use_bias:
            if self.padding=='valid' or self.BN!=1 or self.BN_before_ReLU==1:
                #When padding=='valid' or there is no batch normalization layer, or the batch normalization layer is fused with the previous convolutional layer, all locations have the same bias.
                result=result + self.bias[0]
            else:
                #Otherwise we add different bias to 9 different locations.
                result_0 = result[:, 1:-1, 1:-1, :] + self.bias[0]
                result_1 = result[:, :1, :1, :] + self.bias[1]
                result_2 = result[:, :1, 1:-1, :] + self.bias[2]
                result_3 = result[:, :1, -1:, :] + self.bias[3]
                result_4 = result[:, 1:-1, -1:, :] + self.bias[4]
                result_5 = result[:, -1:, -1:, :] + self.bias[5]
                result_6 = result[:, -1:, 1:-1, :] + self.bias[6]
                result_7 = result[:, -1:, :1, :] + self.bias[7]
                result_8 = result[:, 1:-1, :1, :] + self.bias[8]
                # Parts of image are concatenated to generate a complete output.
                top_row = tf.concat([result_1, result_2, result_3], axis=2)
                middle = tf.concat([result_8, result_0, result_4], axis=2)
                bottom_row = tf.concat([result_7, result_6, result_5], axis=2)
                result = tf.concat([top_row, middle, bottom_row], axis=1)
        return result  
    
class MaxMinPool2D(tf.keras.layers.MaxPool2D):
    """
    Max Pooling or Min Pooling operation, depends on the sign of the batch normalization layer before.
    """
    def build(self, input_shape):
        super().build(input_shape)
        # By default the sign is set to 1, which yields max pooling functionality.
        # The sign variable can be changed for some channels when batch normalization is fused with the next convolutonal layer and it changes the sign of the weights. 
        self.sign=tf.Variable(tf.constant(np.ones((1, 1, 1, input_shape[-1]))), dtype=tf.float64, name='sign')
    def call(self, inputs):
        # Max pooling functionality is called on (self.sign*inputs) input. 
        return super().call(self.sign*inputs)*self.sign
    
def copy_layer(orig_layer):
    """
    Deep copy of a layer with MaxPooling2D layer being replaced with MaxMinPool2D and Conv2D with Conv2DWithBias layer.
    """
    config = orig_layer.get_config()
    if 'pool' in orig_layer.name:
        layer = MaxMinPool2D()
    elif 'conv' in orig_layer.name: 
        config['use_bias']=False
        layer = Conv2DWithBias.from_config(config)
    else:
        layer = type(orig_layer).from_config(config)
    layer.build(orig_layer.input_shape)
    return layer

def copy_model(fused_model, model, i):
    """
    Deep copy of a model and exchange Conv with Conv2DWithBias layers and MaxPool with MaxMinPool layers.
    The layers which are not used during inferece are dropped.
    """
    while i < len(model.layers):
        while 'dropout' in model.layers[i].name or 'activity_regularization' in model.layers[i].name: i+=1
        # Deep copy of layer.
        fused_layer = copy_layer(model.layers[i])
        if 'conv' in model.layers[i].name:
            W, b = model.layers[i].get_weights()
            # Set parameters of Conv2DWithBias layer. 
            fused_layer.set_weights([W, np.array([0]), np.array([0])])
            fused_layer.set_bias(bias=b)
        elif 'dense' in model.layers[i].name:
            # Set parameters of fully-connected layer. 
            fused_layer.set_weights(model.layers[i].get_weights())
        fused_model.add(fused_layer)
        i+=1

def fuse_bn(model, BN, BN_before_ReLU, p, q):
    """
    Creates new models which:
        Fuses all (imaginary) batch normalization layers; 
        Changes bias on locations where it is needed; 
        Transforms MaxPooling layers in MaxMinPooling layers and Conv2D layers in Conv2DWithBias.  
    """
    fused_model = tf.keras.Sequential()
    # Add input layer.
    fused_model.add(copy_layer(model.layers[0]))
    i=1
    # If condition is satisfied, there is an imaginary batch normalization layer which is merged.
    if not (p==0 and q==1): i = fuse_imaginary_bn(fused_model, model, p, q)
    if BN:
        # There are batch normalization layers.
        if BN_before_ReLU:
            # Batch normalization layers are always found before ReLU activation function.
            while i<len(model.layers):
                if 'batch_norm' in model.layers[i].name:
                    # Fuse this batch normalization layer with previous convolutional or fully-connected layer. 
                    i = fuse_bn_before_activation(fused_model, model, i)
                if i==(len(model.layers)-1):
                    # Add last Dense layer with 'softmax'.
                    layer = copy_layer(model.layers[-2])
                    layer.set_weights(model.layers[-2].get_weights())
                    fused_model.add(layer)
                    fused_model.add(copy_layer(model.layers[-1]))
                i+=1
        else:
            # Batch normalization layers are always found after ReLU activation function.
            while i<len(model.layers): 
                # Add first Dense or Convolutional layer if there was no imaginary batch normalization.
                if (p==0 and q==1) and i==1:
                    layer = copy_layer(model.layers[1])
                    if 'conv' in model.layers[1].name:
                        kernel, bias = model.layers[1].get_weights()
                        # Set BN flag to 0 and BN_before_ReLU to 0.
                        layer.set_weights([kernel, np.array([0]), np.array([0])])
                        # Bias is same for all locations.
                        layer.set_bias(bias)
                    else:
                        layer.set_weights(model.layers[1].get_weights())
                    fused_model.add(layer)
                    fused_model.add(copy_layer(model.layers[2]))
                if 'batch_norm' in model.layers[i].name:
                    # Fuse this batch normalization layer with next convolutional or fully-connected layer.
                    i = fuse_bn_after_activation(fused_model, model, i)
                i+=1
    else:
        # If there is no batch normalization layers, copy model such that Conv2D and MaxPooling layers are replaced with ConvWithBias and MaxMinPooling respectively. 
        copy_model(fused_model, model, i)
    fused_model.compile(metrics=['accuracy'], loss="categorical_crossentropy")  
    return fused_model

def fuse_imaginary_bn(fused_model, model, p, q): 
    """
    Fuse an imaginary batch normalization layer due to an input on arbitrary [p, q] range different from [0, 1].
    """
    first_layer = model.layers[1]
    input_image_shape, _, input_channels, _ = tf.shape(first_layer.kernel)
    kappa = tf.cast(tf.fill((input_channels), value=q-p), dtype=tf.float64)
    b_term=tf.cast(tf.fill((input_channels), value=p), dtype=tf.float64)
    if 'conv' in first_layer.name:
        kappa=tf.tile(kappa, [input_image_shape**2])
        b_term=tf.tile(b_term, [input_image_shape**2])
    W = tf.reshape(first_layer.kernel, (-1, first_layer.filters))
    kappa = tf.linalg.diag(kappa)
    W_fused = tf.matmul(kappa, W)
    # See Eq. 13. 
    W_fused = tf.reshape(W_fused, tf.shape(first_layer.kernel))   
    # See Eq. 12. 
    b_fused = first_layer.bias + tf.reduce_sum(tf.matmul(tf.linalg.diag(b_term), W), axis=0)
    # Copy first convolutional or fully-connected layer.
    layer = copy_layer(first_layer)
    if 'conv' in first_layer.name:
        # Set BN flag to 1 and BN_before_ReLU to 0.
        layer.set_weights([W_fused, np.array([1]), np.array([0])])
        # Create bias which will have 9 different values. 
        # Those values are generated by subtracting from b_fused the terms which come through padded input.
        # The obtained results is in Eq. 14. 
        layer.set_bias(bias=b_fused, W=first_layer.kernel, b_term=b_term)
    else:
        layer.set_weights([W_fused, b_fused])
    fused_model.add(layer)
    fused_model.add(copy_layer(model.layers[2]))  
    return 3


def fuse_bn_before_activation(fused_model, model, i):
    """
    Fuses batch normalization layer with previous layer.
    """
    bn = model.layers[i]
    kappa = tf.linalg.diag(bn.gamma/tf.sqrt(bn.epsilon + bn.moving_variance))
    previous_layer = model.layers[i-1]
    output_shape = tf.shape(previous_layer.kernel)[-1]
    W = tf.reshape(previous_layer.kernel, (-1, output_shape))
    # See Eq. 8.
    W_fused = tf.transpose(tf.matmul(kappa, tf.transpose(W)))
    W_fused = tf.reshape(W_fused, tf.shape(previous_layer.kernel))  
    # See Eq. 7.
    b_fused = bn.beta - bn.moving_mean*tf.linalg.diag_part(kappa)
    b_fused += tf.squeeze(tf.matmul(kappa, previous_layer.bias[:, tf.newaxis]))
    layer = copy_layer(previous_layer)
    if 'conv' in previous_layer.name:
        # Set BN flag to 1 and BN_before_ReLU to 1.
        layer.set_weights([W_fused, np.array([1]), np.array([1])])
        # Bias is same everywhere.
        layer.set_bias(b_fused)
    else:
        layer.set_weights([W_fused, b_fused])
    fused_model.add(layer)
    fused_model.add(copy_layer(model.layers[i+1]))
    # Skip Dropout and ActivityRegularization layers. 
    while 'dropout' in model.layers[i+2].name or 'activity_regularization' in model.layers[i+2].name: i+=1
    # Add Flatten layer when it appears. 
    if 'flatten' in model.layers[i+2].name or 'pool' in model.layers[i+2].name: 
        fused_model.add(copy_layer(model.layers[i+2]))
        i+=1
    return i+1

        
def fuse_bn_after_activation(fused_model, model, i): 
    """
    Fuse batch normalization with following layer.
    """
    bn = model.layers[i]
    kappa = bn.gamma/tf.sqrt(bn.epsilon + bn.moving_variance)
    # See Eq. 9. 
    b_term = bn.beta - bn.moving_mean*kappa
    # Skip Dropout and ActivityRegularization layers. 
    while 'dropout' in model.layers[i+1].name or 'activity_regularization' in model.layers[i+2].name: i+=1
    # if there is a MaxPooling layer in before the next parameterized layer, the sign will be changed for the channels where it is needed. 
    if 'max_pool' in model.layers[i+1].name:
        mp = model.layers[i+1]
        mmp = MaxMinPool2D() 
        mmp.build(mp.input_shape)
        # Change sign to the sign of kappa. 
        mmp.sign.assign(tf.math.sign(kappa)[tf.newaxis, tf.newaxis, tf.newaxis, :])
        fused_model.add(mmp)
        i+=1
        if 'dropout' in model.layers[i+1].name: i+=1
    if 'flatten' in model.layers[i+1].name:
        # Add Flatten layer when it appears.
        fused_model.add(copy_layer(model.layers[i+1]))
        ft = model.layers[i+1]
        kappa = tf.tile(kappa, [ft.output_shape[-1]//ft.input_shape[-1]])
        b_term = tf.tile(tf.squeeze(b_term), [ft.output_shape[-1]//ft.input_shape[-1]])
        i+=1
    next_layer = model.layers[i+1]
    input_image_shape = tf.shape(next_layer.kernel)[0]
    output_shape = tf.shape(next_layer.kernel)[-1]
    if 'conv' in next_layer.name:
        kappa=tf.tile(kappa, [input_image_shape**2])
        b_term=tf.tile(b_term, [input_image_shape**2])
    W = tf.reshape(next_layer.kernel, (-1, output_shape))
    kappa = tf.linalg.diag(kappa)
    # See Eq. 10. 
    W_fused = tf.matmul(kappa, W)
    W_fused = tf.reshape(W_fused, tf.shape(next_layer.kernel))    
    # See Eq. 9. 
    b_fused = next_layer.bias + tf.reduce_sum(tf.matmul(tf.linalg.diag(b_term), W), axis=0)
    layer = copy_layer(next_layer)
    if 'conv' in next_layer.name:
        # Set BN flag to 1 and BN_before_ReLU to 0.
        layer.set_weights([W_fused, np.array([1]), np.array([0])])
        # Create bias which will have 9 different values. 
        # Those values are generated by subtracting from b_fused the terms which come through padded input.
        # The obtained results is in Eq. 11.
        layer.set_bias(bias=b_fused, W=next_layer.kernel, b_term=b_term)
    else:
        layer.set_weights([W_fused, b_fused])
    fused_model.add(layer)
    fused_model.add(copy_layer(model.layers[i+2]))   
    return i+2

def dump_cifar10_outputs_for_plotting(model, data, logging_dir):
    """
    Store the layer outputs of the scaled ReLU network. Used to plot Fig. 4.
    """
    outputs=[]
    for layer in model.layers:
        if 'conv' in layer.name or 'dense' in layer.name:
            extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
            extractor.compile()
            layer_output = extractor.predict(data.x_test[:1])
            # For the output layer (dense_1), store values which are obtained without applying softmax function. 
            if layer.name!= 'dense_1': layer_output = (tf.nn.relu(layer_output)).numpy()
            outputs.append(layer_output)
    pkl.dump(outputs, open(logging_dir + '/plot_outputs.pkl', 'wb'))



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
