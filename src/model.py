import pickle as pkl
import logging
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten 
from tensorflow.keras.models import Model
from utils import *
tf.keras.backend.set_floatx('float64')
    
class SpikingDense(tf.keras.layers.Layer):
    def __init__(self, units, name, X_n=1, outputLayer=False, robustness_params={}, input_dim=None):
        self.units = units
        self.B_n = (1 + robustness_params['zeta']) * X_n
        self.outputLayer=outputLayer
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.noise=robustness_params['noise']
        self.threshold=tf.zeros((units, ), dtype=tf.float64)
        # Alpha has fixed noise with given standard deviation.
        self.alpha_noise = tf.random.normal((units, ), stddev=robustness_params['alpha_noise'], dtype=tf.dtypes.float64)
        self.alpha = tf.cast(tf.fill((units, ), 1), dtype=tf.float64) 
        self.input_dim=input_dim
        super(SpikingDense, self).__init__(name=name)
    
    def build(self, input_dim):
        # In case this is the first dense layer after Flatten layer.
        if input_dim[-1] is None: input_dim=(None, self.input_dim)
        self.kernel = self.add_weight(shape=(input_dim[-1], self.units), name='kernel')
        self.b = self.add_weight(shape=(self.units), initializer=tf.constant_initializer(0), name='bias')
        self.built = True
    
    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev, self.t_min, self.t_max = t_min_prev, t_min, t_min+self.B_n
        if not self.outputLayer:
            # This branch corresponds to a ReLU layer.
            W_sum = tf.math.reduce_sum(self.kernel, axis=0)
            # Sets J_ij.
            self.kernel.assign(self.alpha*self.kernel/(1-W_sum))
            J_sum = tf.math.reduce_sum(self.kernel, axis=0)
            D_i = (self.B_n * J_sum) - (self.alpha + J_sum)*self.b
            # Sets vartheta_i (threshold). 
            self.threshold = self.alpha * (self.t_max-self.t_min_prev) + D_i
        else:
            # This corresponds to the softmax (output) layer. In this case there is no threshold and the weights are the same as in the ReLU network. 
            self.alpha = self.b/(self.t_min-self.t_min_prev)
        return self.t_min, self.t_max
            
    def call(self, tj):
        """
        Input spiking times tj, output spiking times ti or the value of membrane potential in case of output layer. 
        """
        # Call layer which corresponds to ReLU first.
        output = call_spiking(tj, self.kernel, self.alpha + self.alpha_noise, self.threshold, self.t_min_prev, self.t_max, noise=self.noise)
        # In case of the output layer a simple integration is applied without spiking. 
        if self.outputLayer:
            # Read out the value of membrane potential at time t_min.
            W_mult_x = tf.matmul(self.t_min-tj, self.kernel)
            output = self.alpha * (self.t_min - self.t_min_prev) + W_mult_x    
        return output
    
    
class SpikingConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, name, X_n=1, padding='same', kernel_size=(3,3), robustness_params={}):
        self.filters=filters
        self.kernel_size=kernel_size
        self.padding=padding
        self.B_n = (1 + robustness_params['zeta']) * X_n
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.noise=robustness_params['noise']
        # When fusing a batch normalization layer with the next convolutional layer where padding='same', some of the biases in scaled ReLU network are changed, leading to 9 different values. This yields different thresholds in the SNN.
        self.threshold=[tf.zeros((filters, ), dtype=tf.float64)]*9
        # Alpha has fixed noise with given standard deviation.
        self.alpha_noise = tf.random.normal((filters, ), stddev=robustness_params['alpha_noise'], dtype=tf.dtypes.float64)
        self.alpha = tf.cast(tf.fill((filters, ), 1), dtype=tf.float64) 
        super(SpikingConv2D, self).__init__(name=name)
    
    def build(self, input_dim):
        self.kernel = self.add_weight(shape=(self.kernel_size[0], self.kernel_size[1], input_dim[-1], self.filters), name='kernel')
        # Depending on whether there is fusion with batch normalization layer and its position with respect to ReLU activation function the processing in spiking convolutional layer can be different.
        self.BN=tf.Variable(tf.constant([0]), name='BN')
        self.BN_before_ReLU=tf.Variable(tf.constant([0]), name='BN_before_ReLU')
        # When fusing a batch normalization layer with the next convolutional layer where padding=='same', some of the biases in scaled ReLU network are changed, leading to 9 different values.
        self.b = self.add_weight(shape=(9, self.filters), initializer=tf.constant_initializer(0), name='bias')
        self.built = True
    
    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev, self.t_min, self.t_max = t_min_prev, t_min, t_min+self.B_n
        W_sum = tf.math.reduce_sum(self.kernel, axis=(0, 1, 2))
        # Sets J_ij.
        self.kernel.assign(self.alpha*self.kernel/(1-W_sum))
        J_sum = tf.math.reduce_sum(self.kernel, axis=(0, 1, 2))
        for i in range(9):
            # Sets 9 different thresholds for 9 different biases. 
            D_i = (self.B_n * J_sum) - (self.alpha + J_sum)*(self.b[i])
            self.threshold[i] = self.alpha * (self.t_max-self.t_min_prev) + D_i
            # When padding=='valid' or there is no batch normalization layer, or the batch normalization layer is fused with the previous convolutional layer, all locations have the same threshold and we break from the for loop. 
            if self.padding=='valid' or self.BN!=1 or self.BN_before_ReLU==1: break
        return self.t_min, self.t_max
    
    def call(self, tj):
        """
        Input spiking times tj, output spiking times ti. 
        """
        # Image size in case of padding='same' or padding='valid'.
        padding_size, image_same_size = int(self.padding=='same')*(self.kernel_size[0]//2), tf.shape(tj)[1] 
        image_valid_size = image_same_size - self.kernel_size[0]+1
        # Pad input with t_min value, which is equivalent with 0 in ReLU network.
        tj=tf.pad(tj, tf.constant([[0, 0], [padding_size, padding_size,], [padding_size, padding_size], [0, 0]]), constant_values=self.t_min)
        # Extract image patches of size (kernel_size, kernel_size). call_spiking function will be called for different patches in parallel.  
        tj = tf.image.extract_patches(tj, sizes=[1, self.kernel_size[0], self.kernel_size[1], 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        # We reshape input and weights in order to utilize the same function as for the fully-connected layer.
        J = tf.reshape(self.kernel, (-1, self.filters))
        if self.padding=='valid' or self.BN!=1 or self.BN_before_ReLU==1: 
            # In this case the threshold is the same for whole input image.
            tj = tf.reshape(tj, (-1, tf.shape(J)[0]))
            ti = call_spiking(tj, J, self.alpha+self.alpha_noise, self.threshold[0], self.t_min_prev, self.t_max, noise=self.noise)   
            # Layer output is reshaped back.
            if self.padding=='valid':
                ti = tf.reshape(ti, (-1, image_valid_size, image_valid_size, self.filters))
            else:
                ti = tf.reshape(ti, (-1, image_same_size, image_same_size, self.filters))
        else:
            # In this case there are 9 different thresholds for 9 different image partitions.
            tj_partitioned = [tj[:, 1:-1, 1:-1, :], tj[:, :1, :1, :], tj[:, :1, 1:-1, :], tj[:, :1, -1:, :], tj[:, 1:-1, -1:, :], tj[:, -1:, -1:, :] , tj[:, -1:, 1:-1, :], tj[:, -1:, :1, :], tj[:, 1:-1, :1, :]]
            ti_partitioned=[]
            for i, tj_part in enumerate(tj_partitioned):
                # Iterate over 9 different partitions and call call_spiking with different threshold value.
                tj_part = tf.reshape(tj_part, (-1, tf.shape(J)[0]))
                ti_part = call_spiking(tj_part, J, self.alpha+self.alpha_noise, self.threshold[i], self.t_min_prev, self.t_max, noise=self.noise) 
                # Partitions are reshaped back.
                if i==0: ti_part=tf.reshape(ti_part, (-1, image_valid_size, image_valid_size, self.filters))
                if i in [1, 3, 5, 7]: ti_part=tf.reshape(ti_part, (-1, 1, 1, self.filters))
                if i in [2, 6]: ti_part=tf.reshape(ti_part, (-1, 1, image_valid_size, self.filters))
                if i in [4, 8]: ti_part=tf.reshape(ti_part, (-1, image_valid_size, 1, self.filters))
                ti_partitioned.append(ti_part) 
            # Partitions are concatenated to create a complete output.
            if image_valid_size!=0:
                ti_top_row = tf.concat([ti_partitioned[1], ti_partitioned[2], ti_partitioned[3]], axis=2)
                ti_middle = tf.concat([ti_partitioned[8], ti_partitioned[0], ti_partitioned[4]], axis=2)
                ti_bottom_row = tf.concat([ti_partitioned[7], ti_partitioned[6], ti_partitioned[5]], axis=2)
                ti = tf.concat([ti_top_row, ti_middle, ti_bottom_row], axis=1)         
            else:
                ti_top_row = tf.concat([ti_partitioned[1], ti_partitioned[3]], axis=2)
                ti_bottom_row = tf.concat([ti_partitioned[7], ti_partitioned[5]], axis=2)
                ti = tf.concat([ti_top_row, ti_bottom_row], axis=1)   
        return ti

def create_fc_model_SNN(logging_dir, model_name, X_n=None, robustness_params={}):
    """
    Create 2-layer fully connected network. Tested on MNIST dataset.
    """
    # Set X_n to the value which is obtained from the scaled ReLU network. 
    if X_n==None: X_n=pkl.load(open(logging_dir + model_name + '_X_n.pkl', 'rb'))
    tj = Input(shape=784)
    ti = SpikingDense(600, 'dense', X_n[0], robustness_params=robustness_params)(tj)
    outputs = SpikingDense(10, 'dense_1', outputLayer=True, robustness_params=robustness_params)(ti)
    model = Model (inputs=tj, outputs=outputs)  
    model.compile(metrics=['accuracy'], loss="categorical_crossentropy") 
    print (model.summary())
    return model

def create_lenet_model_SNN(logging_dir, model_name, robustness_params={}):
    """
    Create 5-layer LeNet5 network. Tested on MNIST dataset.
    """
    # Set X_n to the value which is obtained from the scaled ReLU network.
    X_n=pkl.load(open(logging_dir + model_name + '_X_n.pkl', 'rb'))
    tj = Input(shape=(28, 28, 1))
    ti = SpikingConv2D(6, 'conv2d', X_n[0], padding='same', kernel_size=(5,5), robustness_params=robustness_params)(tj)
    # Finding maximum in ReLU network is equivalent to finding maximum of (-ti) which is the minimum spiking time in SNN. 
    ti=-MaxMinPool2D()(-ti)
    ti = SpikingConv2D(16, 'conv2d_1', X_n[1], padding='valid', kernel_size=(5,5), robustness_params=robustness_params)(ti)
    ti=-MaxMinPool2D()(-ti)
    ti = SpikingConv2D(120, 'conv2d_2', X_n[2], padding='valid', kernel_size=(5,5), robustness_params=robustness_params)(ti)
    ti=Flatten()(ti)
    ti=SpikingDense(84, 'dense', X_n[3], robustness_params=robustness_params, input_dim=120)(ti)
    outputs = SpikingDense(10, 'dense_1', outputLayer=True, robustness_params=robustness_params)(ti)
    model = Model (inputs=tj, outputs=outputs)  
    model.compile(metrics=['accuracy'], loss="categorical_crossentropy") 
    print (model.summary())
    return model
    
def create_vgg_model_SNN(layers2D, kernel_size, layers1D, data, logging_dir, model_name, robustness_params={}):
    """
    Create VGG-like network. Tested on various datasets.
    """
    # Set X_n to the value which is obtained from the scaled ReLU network.
    X_n=pkl.load(open(logging_dir + model_name + '_X_n.pkl', 'rb'))
    tj = Input(shape=data.input_shape) 
    ti = SpikingConv2D(layers2D[0], 'conv2d', X_n[0], padding='same', kernel_size=kernel_size, robustness_params=robustness_params)(tj)
    j, image_size =1, data.input_shape[0]
    for f in layers2D[1:]:
        if f!='pool':
            ti = SpikingConv2D(f, 'conv2d_' +str(j), X_n[j], padding='same', kernel_size=kernel_size, robustness_params=robustness_params)(ti)
            j=j+1
        else:
            ti, image_size=-MaxMinPool2D()(-ti), image_size//2
    ti=Flatten()(ti)
    ti=SpikingDense(layers1D[0], 'dense', X_n[j], robustness_params=robustness_params, input_dim=(image_size**2)*layers2D[-2])(ti)
    j, k=j+1, -1
    for k, d in enumerate(layers1D[1:]):
        ti=SpikingDense(d, 'dense_'+str(k+1), X_n[j], robustness_params=robustness_params)(ti)
        j+=1
    outputs=SpikingDense(data.num_of_classes, 'dense_'+str(k+2), outputLayer=True, robustness_params=robustness_params)(ti)
    model = Model (inputs=tj, outputs=outputs)  
    model.compile(metrics=['accuracy'], loss="categorical_crossentropy") 
    print (model.summary())
    return model


def  call_spiking(tj, J, alpha, threshold, t_min_prev, t_max, noise):
        """
        Calculates spiking times from which ReLU functionality can be recovered.
        """
        # Sort input spiking times tj and their coresponding weights.
        tj_arg_sorted = tf.argsort(tj, -1)
        tj_sorted = tf.sort(tj)[:, :, tf.newaxis]        
        J_sorted = tf.gather(J, tj_arg_sorted, axis=0)               
        J_mult_tj = tf.math.cumsum(tf.math.multiply(J_sorted, tj_sorted), axis=1)
        J_sum = tf.math.cumsum(J_sorted, axis=1)    
        # Calculate the value of potential spiking time ti after each input spike tj is received.
        ti = (J_mult_tj + threshold + alpha*t_min_prev)/(J_sum + alpha) 
        # Spiking happens after input tj if the threshold is reached before the next input spike tj arrives and if the slope of the trajectory is positive.
        # In general, spike ti can happen at any time, however for the threshold we set, the spiking will happen only after all input spikes tj are received (guaranteeing the equivalence with ReLU).  
        tj_last = tf.cast(tf.fill([tf.shape(tj)[0], 1, 1], 1000000.0), dtype=tf.float64)
        tj_next_spike = tf.concat([tj_sorted, tj_last], axis=1)
        tj_next_spike = tf.slice(tj_next_spike, [0, 1, 0], tf.shape(tj_sorted))
        before_next_spike = tf.cast(ti < tj_next_spike, dtype=tf.float64)
        positive_slope = tf.cast((J_sum + alpha) > 0, dtype=tf.float64)     
        # Outputs the first index where both conditions are satisfied.
        spike_index = tf.argmax(before_next_spike * positive_slope, axis=1)[:, :, tf.newaxis] 
        ti = tf.transpose(ti, perm=[0, 2, 1])
        # Find earliest spiking time which satisfies both conditons.
        ti = tf.squeeze(tf.gather(ti, spike_index, batch_dims=2, axis=2), axis=2) 
        # Spike cannot happen after t_max.
        ti = tf.where(ti <= t_max, ti, t_max)
        # Add noise to the spiking time when needed.
        ti = ti + tf.random.normal(tf.shape(ti), stddev=noise, dtype=tf.dtypes.float64)
        return ti



     

        
        